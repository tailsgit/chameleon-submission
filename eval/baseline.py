"""The starter repo's uniform-stretch resize(), kept as the benchmark floor and as
a known-bad input for the checker and scorer tests."""


def stretch(source, target_canvas):
    sx = target_canvas["width"] / source["canvas"]["width"]
    sy = target_canvas["height"] / source["canvas"]["height"]
    out = []
    for el in source["elements"]:
        el = dict(el)
        el.update(x=el["x"] * sx, y=el["y"] * sy, width=el["width"] * sx, height=el["height"] * sy)
        if el["type"] == "text":
            el["font_size"] = max(1.0, el["font_size"] * min(sx, sy))
        out.append(el)
    return {"canvas": {"width": target_canvas["width"], "height": target_canvas["height"]}, "elements": out}
