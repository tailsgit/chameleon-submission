"""Path coverage before optional work; counts, never machine-speed cutoffs.

Each recipe family gets a feasible representative, preservation keeps every
solver-returned analysis/variant, retrieval spans the existing top-30 ranks.
Each applicable realization mode is represented before a shared extra quota.
No design IDs, target-case special cases, scoring or solver changes.
"""
from __future__ import annotations
import math
import os
from itertools import zip_longest
from typing import Any, Iterator
from engine import recipes
from engine.fit import Sizer
from engine.preserve import preserve_plans
from engine.retrieve import retrieved

LAST: dict[str, Any] = {}
FULL_REP_MAX_ASPECT = 2.5  # width/height at and above which a family's representative stays the grid's first corner


def schedule(analyses: list[Any], target: dict, *, recipe_min: int = 1,
             retrieval_min: int = 7, extra: int = 4) -> Iterator[tuple]:
    if recipe_min < 1 or retrieval_min < 1 or extra < 0:
        raise ValueError('path minima must be positive and shared extra nonnegative')
    tw, th = target['width'], target['height']
    # CHAMELEON_FULL_REP=0/1 forces it; by default on below FULL_REP_MAX_ASPECT (wide banners lost with it in the labelled A/B)
    full_rep = os.environ.get('CHAMELEON_FULL_REP', '1' if tw / th < FULL_REP_MAX_ASPECT else '0') == '1'
    sizers = [Sizer(a, tw, th) for a in analyses]
    tasks = [recipes.generate(a, tw, th, tasks_only=True) for a in analyses]
    families = list(dict.fromkeys(f for tasks_by_family in tasks for f in tasks_by_family))
    required = []
    counts = {}
    # Flow splits may rewrap independent plain units; other recipes retain tight
    # source relationships. Fall back to the other analysis when inapplicable.
    for family in families:
        preferred = 1 if family == 'split' and len(analyses) > 1 else 0
        order = [preferred] + [i for i in range(len(analyses)) if i != preferred]
        made = 0
        for ai in order:
            candidates = tasks[ai].get(family, [])
            if full_rep:
                # representative = the family's unshrunk hero and larger scale cap, not the grid's first (smallest) corner
                candidates = sorted(candidates, key=lambda task: (abs(math.log(task[2].get('img_mult', 1.0))), -task[2].get('cap', 0)))
            if family == 'split':
                candidates = sorted(candidates, key=lambda task: task[2].get('vjust') != 'center')
            for fn, name, knobs in candidates:
                p = fn(analyses[ai], sizers[ai], tw, th, **knobs)
                if p is None:
                    continue
                p.recipe, p.knobs, p.sizer = name, knobs, sizers[ai]
                required.append((ai, p)); made += 1
                if made >= recipe_min:
                    break
            if made >= recipe_min:
                break
        counts[family] = made
    # All existing feasible preserve variants survive; CP-SAT settings unchanged.
    for ai, (a, sizer) in enumerate(zip(analyses, sizers)):
        preserved = preserve_plans(a, sizer, tw, th)
        required.extend((ai, p) for p in preserved)
        counts[f'preserve:{ai}'] = len(preserved)
    # Tight analysis covers the retrieval rank strata. A plain-analysis head
    # representative also survives when grouping differs; rank strata cannot
    # silently change analysis when the nearby library is shorter.
    retrieval = [retrieved(a, sizer, tw, th, limit=retrieval_min if ai == 0 else 1,
                           spread=ai == 0)
                 for ai, (a, sizer) in enumerate(zip(analyses, sizers))]
    retrieval_count = 0
    for ai, plans in enumerate(retrieval):
        required.extend((ai, p) for p in plans)
        retrieval_count += len(plans)
    counts['retrieved'] = retrieval_count
    mandatory, optional = [], []
    modes = {}
    for ai, p in required:
        a = analyses[ai]
        focal = (True, False) if any(a.units[u].emphasis for u in p.floating) else (True,)
        variants = [(f, k) for f in focal for k in ((True, False) if a.panels else (True,))]
        mandatory.append((a, p, *variants[0]))
        for f, k in variants[1:]:
            modes.setdefault((f, k), []).append((a, p, f, k))
    # A branch such as keep_on_panel=False cannot be starved by earlier plans.
    for variants in modes.values():
        mandatory.append(variants[0]); optional.append(variants[1:])
    extras = [item for row in zip_longest(*optional) for item in row if item is not None][:extra]
    LAST.clear(); LAST.update(families=counts, guaranteed=len(mandatory), extra=len(extras),
                             total=len(mandatory) + len(extras), retrieval_ranks=[p.knobs['rank'] for _, p in required if p.recipe == 'retrieved'])
    yield from mandatory
    yield from extras
