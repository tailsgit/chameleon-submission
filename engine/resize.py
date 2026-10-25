"""Chameleon engine v1: analyse -> generate layout candidates -> realise -> select.

See engine/analyze.py, recipes.py, fit.py and select.py. resize() never raises:
when the pipeline fails it falls back to a repaired uniform scale, and failing
that, to a repaired stretch. When the aspect ratio barely changes, the uniform
scale is also the preferred candidate: the designer's layout already fits.

Enabled mode uses a fixed realization budget and no elapsed-time selection gates.
Generator proposals are opt-in with GEN_ON=1; GEN_OFF=1 overrides it. Both retain the guard
that drops retrieved layouts which shrink the headline or main image excessively.

CHAMELEON_CRITIC=1 adds a hill-climbing search (engine.search) after selection, judged by the trained
critic (engine.critic_score; the hand scorer when no model is installed). Off by default.
"""

import copy
import logging
import os
from itertools import zip_longest
from typing import Any

from engine.analyze import analyze
from engine.clock import now
from engine.finish import variants
from engine.fit import realize
from engine.recipes import generate
from engine.select import _passes, _tier, better_variant, fallback, repair, select
from eval.score import _area, _drawn_box, key_elements, score

SAME_ASPECT = 0.02  # relative aspect-ratio change below which the source layout is kept as is
BUDGET_S = 14.5     # seconds per case, everything included (the brief's reference: 15 s)
GEN_SHARE = 0.5     # plans are generated in the first half of it...
SELECT_COST = 5     # ...and realising stops while the rest still covers judging every candidate made: the guard
                    # and select check, score and test legibility of each, about this many times (measured as we go)
GUARD = 0.7         # a retrieved layout must give the headline and main image at least this share of the best recipe's
SEARCH_END_S = 12.5  # CHAMELEON_CRITIC=1: the search stops this long after the start (the brief's budget: 13.5 s)
RECIPE_CANDIDATES = 1  # guaranteed representative per applicable recipe family
RETRIEVAL_CANDIDATES = 7  # stratified representatives across the existing top 30
CONTRAST_CANDIDATES = 3
FIXED_CANDIDATES = 4  # shared extra realizations after all path minima
SEARCH_STARTS = 3    # ...starting from the chosen layout, this many other best candidates and the best preserve layout
log = logging.getLogger(__name__)


def resize(source: dict[str, Any], target_canvas: dict[str, Any]) -> dict[str, Any]:
    target = {"width": int(target_canvas["width"]), "height": int(target_canvas["height"])}
    generator_on = os.environ.get("GEN_ON") == "1" and os.environ.get("GEN_OFF") != "1"
    deadline = None
    try:
        # tight groups held rigid (engine.analyze) and, when that changed the units, also free: the scorer picks
        tight, plain = analyze(source), analyze(source, tight=False)
        pool = [tight] + ([plain] if plain.units.keys() != tight.units.keys() else [])
        outputs, recipes, made_by = [], [], []
        from engine.candidate_budget import schedule
        for a, plan, f, k in schedule(pool, target, recipe_min=RECIPE_CANDIDATES,
                                      retrieval_min=RETRIEVAL_CANDIDATES,
                                      extra=FIXED_CANDIDATES):
            try:
                outputs.append(realize(a, plan, target["width"], target["height"], focal=f, keep_on_panel=k))
                recipes.append(plan.recipe)
                made_by.append(a)
            except Exception:
                log.exception("recipe %s %s failed", plan.recipe, plan.knobs)
        if generator_on:
            from engine.gen_candidates import candidates, LAST, layout_distance
            try:
                generated = candidates(tight, target)
            except Exception:
                log.exception("generator failed; retaining existing candidates")
                generated = []
            LAST["new_unique"] = sum(all(layout_distance(g, o) >= .08 for o in outputs) for g in generated)
            outputs.extend(generated)
            recipes.extend(["generator"] * len(generated))
            made_by.extend([tight] * len(generated))
        preferred = None
        sa = source["canvas"]["width"] / source["canvas"]["height"]
        if abs(target["width"] / target["height"] / sa - 1) <= SAME_ASPECT:
            preferred = fallback(source, target)
        made_by = {id(o): x for o, x in zip(outputs, made_by)}
        outputs, recipes = _guard(source, target, outputs, recipes)
        if outputs or preferred is not None:
            best = select(source, target, outputs, preferred, deadline, render_limit=CONTRAST_CANDIDATES)
            if generator_on:
                from engine.gen_candidates import LAST
                LAST["won_selection"] = any(o is best for o in generated)
            a = made_by.get(id(best), tight)
            if any(o is best for o in outputs):  # the chosen layout pinned to the corners and/or grown into its free space
                extra = variants(a, best, target["width"], target["height"])
                win = better_variant(source, target, best, extra)
                if any(o is win for o in extra):
                    outputs, recipes = outputs + [win], recipes + ["finished"]
                best = win
            if os.environ.get("CHAMELEON_CRITIC") == "1" and any(o is best for o in outputs):
                best = _critic_search(source, target, a, outputs, recipes, best, None)
                if not any(o is best for o in outputs):
                    outputs, recipes = outputs + [best], recipes + ["searched"]
            log.info("winner: %s", next((r for o, r in zip(outputs, recipes) if o is best),
                                        "uniform" if best is preferred else "repaired"))
            return best
        log.warning("no candidates; using fallback")
    except Exception:
        log.exception("engine failed; using fallback")
    try:
        return fallback(source, target)
    except Exception:
        log.exception("fallback failed; using repaired stretch")
        return _stretch(source, target)


def _plans(a, target, until, *, recipes=RECIPE_CANDIDATES, retrieval=RETRIEVAL_CANDIDATES):
    plans = generate(a, target["width"], target["height"], until,
                     recipe_limit=recipes, retrieval_limit=retrieval)
    if until is None:
        # Fixed round-robin across recipe families retains preserve/retrieval.
        # Solver settings, plans themselves and the scorer are unchanged.
        from collections import OrderedDict
        families = OrderedDict()
        for plan in plans:
            families.setdefault(plan.recipe, []).append(plan)
        plans = [p for row in zip_longest(*families.values()) for p in row if p is not None]
    for plan in plans:
        yield a, plan


def _alternate(gens):
    """Walk the generators in step and keep the i-th item of generator i % n (then the rest of the longest), so
    several analyses together make as many candidates as one: each covers its share of every recipe's knobs."""
    for i, items in enumerate(zip_longest(*gens)):
        mine = items[i % len(items)]
        yield mine if mine is not None else next(x for x in items if x is not None)


def _critic_search(source, target, a, outputs, recipes, best, deadline):
    """The chosen layout, or the best layout engine.search climbs to from it and the best other candidates, when that
    passes check(), is in no worse a safety tier (engine.select._tier) and the critic scores it higher."""
    from engine.search import search
    ranked = sorted(((score(source, target, o)["total"], k) for k, o in enumerate(outputs) if o is not best), reverse=True)
    others = [outputs[k] for _s, k in ranked if recipes[k] != "preserve"][:SEARCH_STARTS]
    others += [outputs[k] for _s, k in ranked if recipes[k] == "preserve"][:1]
    starts = [best] + [o for o in others if _passes(source, target, o)]
    try:
        found = search(source, target, a, starts, deadline)
    except Exception:
        log.exception("critic search failed; keeping the chosen layout")
        return best
    from engine.critic_score import scores
    tb, base = _tier(source, target, best), scores(source, target, [best], a)[0]
    for s, o in found:  # best first; each passed check() in the search
        if s <= base:
            break
        if _tier(source, target, o) >= tb:
            log.info("critic search: %.3f over the chosen layout's %.3f", s, base)
            return o
    return best


def _guard(source, target, outputs, recipes):
    """Drop retrieved layouts that shrink the headline's or main image's share of the canvas
    below GUARD x what the best-scoring passing recipe layout gives it."""
    if "retrieved" not in recipes:
        return outputs, recipes
    ranked = [(score(source, target, o)["total"], o) for o, r in zip(outputs, recipes) if r != "retrieved" and _passes(source, target, o)]
    if not ranked:
        return outputs, recipes
    keys = key_elements(source)
    ref = _shares(max(ranked, key=lambda t: t[0])[1], keys)
    keep = [(o, r) for o, r in zip(outputs, recipes)
            if r != "retrieved" or all(s >= GUARD * b for s, b in zip(_shares(o, keys), ref) if b)]
    if len(keep) < len(outputs):
        log.info("guard dropped %d retrieved layouts", len(outputs) - len(keep))
    return [o for o, _r in keep], [r for _o, r in keep]


def _shares(output, keys):
    """Each key element's drawn area as a share of the canvas (None when there is no such element)."""
    els = {e["id"]: e for e in output["elements"]}
    area = output["canvas"]["width"] * output["canvas"]["height"]
    return [_area(_drawn_box(els[k])) / area if k in els else None for k in keys]


def _stretch(source, target):
    sx = target["width"] / source["canvas"]["width"]
    sy = target["height"] / source["canvas"]["height"]
    out = {"canvas": dict(target), "elements": copy.deepcopy(source["elements"])}
    for el in out["elements"]:
        el.update(x=el["x"] * sx, y=el["y"] * sy, width=el["width"] * sx, height=el["height"] * sy)
        if el["type"] == "text":
            el["font_size"] = max(1, int(el["font_size"] * min(sx, sy)))
    try:
        return repair(source, target, out)
    except Exception:
        return out
