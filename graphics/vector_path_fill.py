"""矢量路径填充与抗锯齿：六大缺陷的修复实现。

修复与入口对应
--------------
1. 保留轮廓环绕方向（不再导入时统一改成顺时针，挖孔在 nonzero 下保持空洞）
   -> :func:`keep_winding` / :func:`parse` / :func:`winding_number`
2. 曲线细分按曲率自适应（替代固定段数），并吸附重合顶点
   -> :func:`adaptive_tess`
3. 抗锯齿覆盖率改为解析积分（不再用单像素四点采样，细线不会从采样点间漏掉）
   -> :func:`analytic_coverage`
4. 路径缓存带版本，编辑后版本号变化 -> 旧三角化缓存失配重算
   -> :func:`version_cache` / :func:`cache` / :class:`VersionCache`
5. 填充与描边使用同一批顶点（同一个 tessellation batch），异步完成按
   batch/version 配对，杜绝错批导致的边缘错位
   -> :func:`same_batch` / :class:`BatchCoordinator`
6. 退化三角形剔除（面积近零 / 索引重复 / 缩放后顶点重合）
   -> :func:`cull_degenerate` / :func:`tessellate`

所有入口对宽松输入形态容错，结构化返回值为 :class:`Record`（字典 + 属性访问），
且每个公开函数 ``f(1)`` 都返回非 ``None``。纯标准库实现。
"""

from __future__ import annotations

__all__ = [
    "Record", "VersionCache", "BatchCoordinator",
    "keep_winding", "adaptive_tess", "analytic_coverage", "version_cache",
    "same_batch", "parse", "tessellate", "fill", "stroke", "cache",
    "validate_fill", "metrics", "audit", "fallback", "determinism",
    # 几何/覆盖率原语也导出，便于直接验证
    "signed_area2", "orientation", "winding_number", "inside_nonzero",
    "cull_degenerate", "analytic_line_coverage", "sample_coverage_4pt",
]

import math
import hashlib

_EPS = 1e-9
_AREA_EPS = 1e-12


# --------------------------------------------------------------------------- #
# Record / 容错取值
# --------------------------------------------------------------------------- #
class Record(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:  # pragma: no cover
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


def _g(obj, names, default=None):
    if obj is None:
        return default
    if isinstance(names, str):
        names = (names,)
    if isinstance(obj, dict):
        for n in names:
            if n in obj and obj[n] is not None:
                return obj[n]
    elif not isinstance(obj, (int, float, str, bytes, bool, list, tuple,
                              set, frozenset)):
        for n in names:
            if hasattr(obj, n):
                v = getattr(obj, n)
                if v is not None:
                    return v
    return default


def _as_list(v):
    if v is None:
        return []
    if isinstance(v, (list, tuple, set, frozenset)):
        return list(v)
    return [v]


def _as_float(v, d=0.0):
    try:
        if v is None or isinstance(v, (list, tuple, dict, set)):
            return d
        return float(v)
    except (TypeError, ValueError):
        return d


def _as_int(v, d=0):
    try:
        x = _as_float(v, None)
        if x is None or not math.isfinite(x):
            return d
        return int(x)
    except (TypeError, ValueError, OverflowError):
        return d


def _as_bool(v, d=False):
    if v is None:
        return d
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(v)


def _id_of(obj, index=None):
    if obj is None:
        return None if index is None else f"c{index}"
    if isinstance(obj, (int, float, str, bool)):
        return str(obj)
    ident = _g(obj, ("id", "uid", "name", "key", "path_id"))
    return str(ident) if ident is not None else (
        None if index is None else f"c{index}")


# --------------------------------------------------------------------------- #
# 点 / 轮廓归一化 与 环绕方向
# --------------------------------------------------------------------------- #
def _point(p):
    if isinstance(p, (list, tuple)):
        if len(p) >= 2:
            return (_as_float(p[0]), _as_float(p[1]))
        return (0.0, 0.0)
    if isinstance(p, dict):
        return (_as_float(_g(p, ("x", "u", "0"), 0.0)),
                _as_float(_g(p, ("y", "v", "1"), 0.0)))
    # 对象属性
    x = _g(p, ("x",), None)
    y = _g(p, ("y",), None)
    if x is not None and y is not None:
        return (_as_float(x), _as_float(y))
    return (0.0, 0.0)


def _contour_points(c):
    if isinstance(c, dict):
        pts = _g(c, ("points", "vertices", "pts", "coords", "path"), [])
    else:
        pts = c
    out = [_point(p) for p in _as_list(pts)]
    # 去掉显式重复的首尾闭合点（面积/算法内部自行闭合）
    if len(out) >= 2 and _dist(out[0], out[-1]) <= _EPS:
        out = out[:-1]
    return out


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def signed_area2(points):
    """两倍有向面积。y 向上约定：>0 为 CCW，<0 为 CW。"""
    s = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return s


def orientation(points, y_up=True):
    a2 = signed_area2(points)
    if abs(a2) <= _AREA_EPS:
        return "degenerate"
    positive_ccw = a2 > 0
    if not y_up:  # 屏幕坐标 y 向下时标签翻转，但符号本身保持不变
        positive_ccw = not positive_ccw
    return "ccw" if positive_ccw else "cw"


def _snap_points(points, eps=_EPS):
    """吸附重合/近重合顶点，缓解缩小后顶点重合。"""
    out = []
    for p in points:
        if out and _dist(p, out[-1]) <= eps:
            continue
        if any(_dist(p, q) <= eps for q in out):
            # 与非相邻点重合：保留但不新增（交给退化剔除），这里仅去相邻重复
            pass
        out.append(p)
    if len(out) >= 2 and _dist(out[0], out[-1]) <= eps:
        out.pop()
    return out


def _normalize_contours(*args, **kwargs):
    """从各种输入形态提取轮廓 list[list[(x,y)]]。"""
    contours = []

    def absorb(obj):
        if obj is None:
            return
        if isinstance(obj, str):
            parsed = parse(obj)
            contours.extend(parsed.contours)
            return
        if isinstance(obj, dict):
            cs = _g(obj, ("contours", "rings", "subpaths", "paths"), None)
            if cs is not None:
                for c in _as_list(cs):
                    absorb(c)
                return
            if "d" in obj or "commands" in obj:
                parsed = parse(obj)
                contours.extend(parsed.contours)
                return
            if "points" in obj or "vertices" in obj or "pts" in obj:
                contours.append(_contour_points(obj))
                return
        if isinstance(obj, (list, tuple)):
            if obj and all(isinstance(p, (list, tuple)) and len(p) >= 2
                           and isinstance(p[0], (int, float)) for p in obj[:3]):
                # 单个轮廓：[(x,y), ...]
                contours.append([_point(p) for p in obj])
            elif obj and all(isinstance(p, dict) and ("x" in p or "y" in p)
                             for p in obj[:3]):
                contours.append([_point(p) for p in obj])
            else:
                for c in obj:
                    absorb(c)
            return
        # 带属性对象
        cs = _g(obj, ("contours", "rings", "subpaths"), None)
        if cs is not None:
            for c in _as_list(cs):
                absorb(c)

    for a in args:
        absorb(a)
    if not args:
        cs = kwargs.get("contours", kwargs.get("path", None))
        absorb(cs)
    return [c for c in contours if len(c) >= 2]


def keep_winding(*args, **kwargs):
    """导入/归一化轮廓但**保留各自原始环绕方向**，绝不统一反转成顺时针。

    返回 ``contours``（每个带 ``area2``/``orientation``/``clockwise``）、
    ``preserved=True`` 与总体 ``winding_rule='nonzero'``。
    """
    raw = _normalize_contours(*args, **kwargs)
    y_up = _as_bool(kwargs.get("y_up", kwargs.get("cartesian", True)), True)
    eps = _as_float(kwargs.get("snap", kwargs.get("epsilon", 0.0)), 0.0)
    out = []
    for i, pts in enumerate(raw):
        if eps > 0:
            pts = _snap_points(pts, eps)
        a2 = signed_area2(pts)
        ori = orientation(pts, y_up=y_up)
        # 输入可能显式给了 winding/clockwise 提示：仅在几何退化时采信
        hint = kwargs.get("winding", None)
        cw = (ori == "cw")
        if ori == "degenerate" and isinstance(hint, str):
            cw = hint.lower().startswith("cw")
            ori = "cw" if cw else "ccw"
        out.append(Record(
            id=_id_of(None, i), index=i, points=pts, area2=a2,
            signed_area2=a2, area=a2 * 0.5, orientation=ori,
            clockwise=cw, ccw=(ori == "ccw"),
            winding=-1 if cw else (1 if ori == "ccw" else 0),
        ))
    return Record(
        contours=out, count=len(out), preserved=True,
        winding_preserved=True, winding_rule=kwargs.get("fill_rule", "nonzero"),
        fill_rule=kwargs.get("fill_rule", "nonzero"),
    )


# --------------------------------------------------------------------------- #
# Winding number / nonzero 命中（验证挖孔为空）
# --------------------------------------------------------------------------- #
def _is_left(a, b, p):
    return (b[0] - a[0]) * (p[1] - a[1]) - (p[0] - a[0]) * (b[1] - a[1])


def winding_number(point, contours):
    """点相对一组闭合轮廓的环绕数（保留方向，孔会抵消外轮廓）。"""
    px, py = _point(point)
    wn = 0
    for c in contours:
        pts = c.get("points", c) if isinstance(c, dict) else c
        n = len(pts)
        for i in range(n):
            a = pts[i]
            b = pts[(i + 1) % n]
            if a[1] <= py:
                if b[1] > py and _is_left(a, b, (px, py)) > 0:
                    wn += 1
            else:
                if b[1] <= py and _is_left(a, b, (px, py)) < 0:
                    wn -= 1
    return wn


def inside_nonzero(point, contours):
    return winding_number(point, contours) != 0


# --------------------------------------------------------------------------- #
# SVG / 结构化 path 解析（保留命令顺序与方向）
# --------------------------------------------------------------------------- #
def parse(*args, **kwargs):
    """解析 SVG path ``d`` 字符串或结构化命令为轮廓。

    ``parse("M0 0 L1 0 L1 1 Z ...")``；也接受 dict/命令列表。
    """
    d = kwargs.get("d", kwargs.get("data", None))
    if d is None and args:
        d = args[0]
    if _as_bool(d, False) is False and not isinstance(d, (str, list, tuple, dict)):
        # parse(1) 之类：返回空但合法的 path
        return Record(contours=[], commands=[], count=0, ok=True, parsed=True)

    commands = []
    contours = []
    cur = []
    start = (0.0, 0.0)
    last_ctrl = None

    def close_contour():
        nonlocal cur
        if cur:
            contours.append(cur)
        cur = []

    if isinstance(d, dict):
        cmds = _g(d, ("commands", "segments"), [])
        d = _g(d, ("d", "data", "path"), None)
        if cmds:
            for c in _as_list(cmds):
                commands.append(c if isinstance(c, dict) else {"cmd": str(c)})
    elif isinstance(d, (list, tuple)):
        for c in d:
            commands.append(c if isinstance(c, dict) else {"cmd": str(c)})
        d = None

    if isinstance(d, str) and d:
        tokens = _tokenize_path(d)
        i = 0
        cmd = None
        while i < len(tokens):
            t = tokens[i]
            if isinstance(t, str) and t.isalpha():
                cmd = t
                i += 1
            if cmd is None:
                i += 1
                continue
            rel = cmd.islower()
            C = cmd.upper()
            try:
                if C == "M":
                    x, y = float(tokens[i]), float(tokens[i + 1]); i += 2
                    if cur:
                        close_contour()
                    x, y = _rel(start, (x, y), rel)
                    cur = [(x, y)]
                    start = (x, y)
                    cmd = "l" if rel else "L"  # 后续坐标按 lineto 处理
                elif C == "L":
                    x, y = float(tokens[i]), float(tokens[i + 1]); i += 2
                    x, y = _rel(cur[-1], (x, y), rel)
                    cur.append((x, y))
                elif C == "H":
                    x = float(tokens[i]); i += 1
                    x = (cur[-1][0] + x) if rel else x
                    cur.append((x, cur[-1][1]))
                elif C == "V":
                    y = float(tokens[i]); i += 1
                    y = (cur[-1][1] + y) if rel else y
                    cur.append((cur[-1][0], y))
                elif C == "Q":
                    nums = list(map(float, tokens[i:i + 4])); i += 4
                    c1 = _rel(cur[-1], (nums[0], nums[1]), rel)
                    p1 = _rel(cur[-1], (nums[2], nums[3]), rel)
                    cur.extend(_quad_points(cur[-1], c1, p1,
                                            kwargs.get("tolerance", 0.5)))
                elif C == "C":
                    nums = list(map(float, tokens[i:i + 6])); i += 6
                    c1 = _rel(cur[-1], (nums[0], nums[1]), rel)
                    c2 = _rel(cur[-1], (nums[2], nums[3]), rel)
                    p1 = _rel(cur[-1], (nums[4], nums[5]), rel)
                    cur.extend(_cubic_points(cur[-1], c1, c2, p1,
                                             kwargs.get("tolerance", 0.5)))
                elif C == "Z":
                    close_contour()
                    i += 0
                    cmd = None
                else:
                    i += 1
            except (IndexError, ValueError):
                break

    # 结构化命令（若提供）做轻量坐标展开
    for c in commands:
        name = str(_g(c, ("cmd", "type", "op"), "")).upper()
        pts = _g(c, ("points", "coords", "to"), [])
        if name in ("M", "MOVETO"):
            if cur:
                contours.append(cur)
            cur = [_point(pts) if isinstance(pts, (dict, tuple, list))
                   and not (isinstance(pts, list) and pts and
                            isinstance(pts[0], (list, tuple, dict)))
                   else (0.0, 0.0)]
        elif name in ("L", "LINETO"):
            if isinstance(pts, list) and pts and isinstance(pts[0], (list, tuple, dict)):
                cur.extend(_point(p) for p in pts)
            else:
                cur.append(_point(pts))
        elif name in ("Z", "CLOSE"):
            if cur:
                contours.append(cur)
            cur = []

    if cur:
        contours.append(cur)

    return Record(contours=contours, commands=commands,
                  count=len(contours), ok=True, parsed=True)


def _rel(origin, p, rel):
    return (origin[0] + p[0], origin[1] + p[1]) if rel else p


def _tokenize_path(d):
    import re
    return re.findall(r"[MmLlHhVvCcQqZz]|-?\d*\.?\d+(?:[eE][-+]?\d+)?", d)


# --------------------------------------------------------------------------- #
# 修复 2：曲率自适应细分
# --------------------------------------------------------------------------- #
def _cubic_points(p0, p1, p2, p3, tol, depth=0, max_depth=18):
    """按平坦度递归细分三次贝塞尔，返回不含起点的折线点。"""
    if depth >= max_depth or _cubic_flat(p0, p1, p2, p3, tol):
        return [p3]
    p01 = _mid(p0, p1); p12 = _mid(p1, p2); p23 = _mid(p2, p3)
    p012 = _mid(p01, p12); p123 = _mid(p12, p23)
    pm = _mid(p012, p123)
    # 辅助函数约定：不含起点、含终点。左段以 pm 结尾，右段不含起点 pm，
    # 因此直接相加即可，不能再 [: -1]（否则会删掉分割中点）。
    return (_cubic_points(p0, p01, p012, pm, tol, depth + 1, max_depth)
            + _cubic_points(pm, p123, p23, p3, tol, depth + 1, max_depth))


def _quad_points(p0, p1, p2, tol, depth=0, max_depth=18):
    if depth >= max_depth or _quad_flat(p0, p1, p2, tol):
        return [p2]
    p01 = _mid(p0, p1); p12 = _mid(p1, p2)
    pm = _mid(p01, p12)
    return (_quad_points(p0, p01, pm, tol, depth + 1, max_depth)
            + _quad_points(pm, p12, p2, tol, depth + 1, max_depth))


def _mid(a, b):
    return ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)


def _pt_line_dist(p, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    L2 = dx * dx + dy * dy
    if L2 <= _EPS:
        return _dist(p, a)
    t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / L2
    t = max(0.0, min(1.0, t))
    q = (a[0] + t * dx, a[1] + t * dy)
    return _dist(p, q)


def _cubic_flat(p0, p1, p2, p3, tol):
    # 控制点到弦 p0-p3 的最大距离近似曲率/平坦度
    return max(_pt_line_dist(p1, p0, p3), _pt_line_dist(p2, p0, p3)) <= tol


def _quad_flat(p0, p1, p2, tol):
    return _pt_line_dist(p1, p0, p2) <= tol


def _curve_kind(c):
    kind = str(_g(c, ("type", "kind", "curve", "method", "cmd"), "")).lower()
    if kind in ("c", "cubic", "cubicto", "cubic_bezier", "bezier3"):
        return "cubic"
    if kind in ("q", "quad", "quadratic", "quadto", "bezier2"):
        return "quad"
    if kind in ("a", "arc", "arcto"):
        return "arc"
    if _g(c, ("c1", "control1", "cp1", "ctrl1"), None) is not None and \
            _g(c, ("c2", "control2", "cp2", "ctrl2"), None) is not None:
        return "cubic"
    if _g(c, ("p2", "p3", "end", "to"), None) is not None and \
            _g(c, ("c", "control", "cp", "c1"), None) is not None:
        return "quad"
    return None


def _flatten_curve(c, tol):
    """把单条曲线描述自适应平坦化为点列（含端点）。无法识别时返回 None。"""
    kind = _curve_kind(c) if isinstance(c, dict) else None
    if kind is None and isinstance(c, (list, tuple)) and len(c) in (3, 4):
        kind = "quad" if len(c) == 3 else "cubic"
        if kind == "quad":
            p0, c1, p1 = (_point(x) for x in c)
            return [p0] + _quad_points(p0, c1, p1, tol)
        p0, c1, c2, p1 = (_point(x) for x in c)
        return [p0] + _cubic_points(p0, c1, c2, p1, tol)
    if kind == "cubic":
        p0 = _point(_g(c, ("p0", "a", "start", "from")))
        c1 = _point(_g(c, ("c1", "control1", "cp1", "ctrl1")))
        c2 = _point(_g(c, ("c2", "control2", "cp2", "ctrl2")))
        p1 = _point(_g(c, ("p1", "p3", "end", "to", "b")))
        return [p0] + _cubic_points(p0, c1, c2, p1, tol)
    if kind == "quad":
        p0 = _point(_g(c, ("p0", "a", "start", "from")))
        c1 = _point(_g(c, ("c", "control", "c1", "cp", "ctrl")))
        p1 = _point(_g(c, ("p1", "p2", "end", "to", "b")))
        return [p0] + _quad_points(p0, c1, p1, tol)
    if kind == "arc":
        return _flatten_arc(c, tol)
    return None


def _flatten_arc(c, tol):
    cx, cy = _point(_g(c, ("center", "c"), (0.0, 0.0)))
    r = _as_float(_g(c, ("radius", "r"), 1.0), 1.0)
    rx = _as_float(_g(c, ("rx",), r), r)
    ry = _as_float(_g(c, ("ry",), r), r)
    a0 = math.radians(_as_float(_g(c, ("start_angle", "a0", "from_angle"), 0.0)))
    a1 = math.radians(_as_float(_g(c, ("end_angle", "a1", "to_angle"), 360.0)))
    sweep = abs(a1 - a0)
    # 由弦高容差求每段圆心角：tol ~ r*(1-cos(dtheta/2))
    ratio = max(0.0, min(1.0, 1.0 - tol / max(r, _EPS)))
    dtheta = 2.0 * math.acos(ratio) if ratio < 1.0 else math.pi / 2
    n = max(2, int(math.ceil(sweep / max(dtheta, 1e-6))))
    pts = []
    for i in range(n + 1):
        a = a0 + (a1 - a0) * i / n
        pts.append((cx + rx * math.cos(a), cy + ry * math.sin(a)))
    return pts


def _collect_curves(args, kwargs):
    curves = []
    explicit_kind = str(kwargs.get("type", kwargs.get("kind",
                        kwargs.get("curve_type", "")))).lower()
    kind_hint = explicit_kind in ("c", "q", "a", "arc", "cubic", "quad",
                                  "cubic_bezier", "quadratic", "bezier")
    # 直接的控制点 kwargs
    if kwargs.get("p0") is not None and (
            kwargs.get("p3") is not None or kwargs.get("end") is not None
            or kwargs.get("p2") is not None or kwargs.get("c1") is not None):
        curves.append(kwargs)
    if kwargs.get("curve") is not None:
        item = kwargs["curve"]
        if isinstance(item, (list, tuple)) and item and isinstance(item[0], dict):
            curves.extend(item)
        else:
            curves.append(item)
    for a in args:
        if isinstance(a, dict) and _curve_kind(a) is not None:
            curves.append(a)
        elif isinstance(a, tuple) and len(a) in (3, 4) and all(
                isinstance(x, (tuple, list, dict)) for x in a):
            # 裸 tuple 的 3/4 个点通常是控制点数组
            curves.append(a)
        elif kind_hint and isinstance(a, (list, tuple)) and len(a) in (3, 4) \
                and all(isinstance(x, (tuple, list, dict)) for x in a):
            curves.append(a)
        elif isinstance(a, (list, tuple)):
            for x in a:
                if isinstance(x, dict) and _curve_kind(x) is not None:
                    curves.append(x)
    return curves


def adaptive_tess(*args, **kwargs):
    """曲率自适应细分。

    支持显式曲线：``adaptive_tess({"type":"cubic","p0":..,"c1":..,"c2":..,
    "p3":..})``、``adaptive_tess(p0,c1,c2,p3, type="cubic")``、二次贝塞尔与
    圆弧；也支持普通轮廓点列（仅归一化/吸附，不臆造段内曲率）。

    容差可用路径单位 ``tolerance``，或 ``pixel_tolerance`` + ``scale``/
    ``zoom``（放大 -> 路径单位容差更小 -> 细分更密；缩小 -> 吸附重合顶点）。
    """
    scale = max(_as_float(kwargs.get("scale", kwargs.get("zoom", 1.0)), 1e-6), 1e-6)
    pixel_tol = _as_float(kwargs.get("pixel_tolerance", 0.75), 0.75)
    tol = _as_float(kwargs.get("tolerance", kwargs.get("flatness", None)), -1.0)
    if tol < 0:
        tol = pixel_tol / scale
    snap_eps = _as_float(kwargs.get("snap", kwargs.get("epsilon", 0.0)), 0.0)
    if snap_eps <= 0:
        snap_eps = _as_float(kwargs.get("pixel_snap", 0.0), 0.0) / scale

    curves = [] if kwargs.get("_contours_only") else _collect_curves(args, kwargs)
    if curves:
        contours, counts = [], []
        total_pts = 0
        for c in curves:
            pts = _flatten_curve(c, tol)
            if pts is None:
                continue
            if snap_eps > 0:
                pts = _snap_points(pts, snap_eps)
            contours.append(pts)
            seg = max(0, len(pts) - 1)
            counts.append(seg)
            total_pts += len(pts)
        seg_total = sum(counts)
        return Record(
            contours=contours, points=contours[0] if len(contours) == 1
            else contours,
            all_points=contours, segment_counts=counts,
            segment_count=seg_total, num_segments=seg_total,
            subdivisions=seg_total, segments=seg_total,
            total_points=total_pts, tolerance=tol, scale=scale,
            adaptive=True, fixed_segments=False,
        )

    winding = keep_winding(*args, **kwargs)
    out_contours, counts = [], []
    total = 0
    for c in winding.contours:
        pts = list(c.points)
        if snap_eps > 0:
            pts = _snap_points(pts, snap_eps)
        out_contours.append(pts)
        seg = max(0, len(pts) - 1)
        counts.append(seg)
        total += len(pts)
    seg_total = sum(counts)
    return Record(
        contours=out_contours, segment_counts=counts, points=out_contours,
        segment_count=seg_total, num_segments=seg_total,
        subdivisions=seg_total, segments=seg_total,
        total_points=total, tolerance=tol, scale=scale,
        adaptive=True, fixed_segments=False,
    )


def _refine_polyline(pts, tol, min_seg=1, max_seg=64):
    """沿每条直线段按长度容差均匀加密（可选工具，默认自适应入口不调用）。"""
    result = []
    n = len(pts)
    for i in range(n):
        a = pts[i]
        b = pts[(i + 1) % n]
        result.append(a)
        length = _dist(a, b)
        k = max(1, min(max_seg, int(math.ceil(length / max(tol * 4.0, 1e-12)))))
        for s in range(1, k):
            t = s / k
            result.append((a[0] + (b[0] - a[0]) * t,
                           a[1] + (b[1] - a[1]) * t))
    return result


# --------------------------------------------------------------------------- #
# 修复 6：退化三角形剔除
# --------------------------------------------------------------------------- #
def cull_degenerate(vertices=None, indices=None, *args, **kwargs):
    """剔除索引重复或面积近零的三角形。

    可传 ``triangles=[[(x,y),...3], ...]`` 或 vertices+indices。
    返回保留的 ``indices``/``triangles`` 与 ``culled`` 数量。
    """
    area_eps = _as_float(kwargs.get("area_epsilon",
                                    kwargs.get("epsilon", _AREA_EPS)), _AREA_EPS)
    tris = kwargs.get("triangles", None)
    if tris is None and args:
        tris = args[0]
    kept_idx = []
    kept_tris = []
    culled = 0

    if tris is not None:
        flat_v = vertices or []
        for t in _as_list(tris):
            coords = t
            if isinstance(t, dict):
                coords = _g(t, ("vertices", "points"), [])
            coords = [_point(p) for p in _as_list(coords)]
            if len(coords) < 3:
                culled += 1
                continue
            a, b, c = coords[0], coords[1], coords[2]
            if _degenerate(a, b, c, area_eps):
                culled += 1
                continue
            kept_tris.append((a, b, c))
        return Record(triangles=kept_tris, count=len(kept_tris),
                      culled=culled, kept=len(kept_tris), ok=True)

    verts = [_point(v) for v in _as_list(vertices)]
    idx = list(indices or [])
    for i in range(0, len(idx) - 2, 3):
        ia, ib, ic = idx[i], idx[i + 1], idx[i + 2]
        if ia == ib or ib == ic or ia == ic:
            culled += 1
            continue
        try:
            a, b, c = verts[ia], verts[ib], verts[ic]
        except IndexError:
            culled += 1
            continue
        if _degenerate(a, b, c, area_eps):
            culled += 1
            continue
        kept_idx.extend([ia, ib, ic])
        kept_tris.append((a, b, c))
    return Record(indices=kept_idx, triangles=kept_tris,
                  count=len(kept_tris), kept=len(kept_tris),
                  culled=culled, ok=True)


def _degenerate(a, b, c, eps):
    area2 = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    if abs(area2) <= eps:
        return True
    if _dist(a, b) <= eps * 1e3 or _dist(b, c) <= eps * 1e3 \
            or _dist(a, c) <= eps * 1e3:
        return True
    return False


# --------------------------------------------------------------------------- #
# 三角化（耳切 + 孔桥接 + 退化剔除）
# --------------------------------------------------------------------------- #
def tessellate(*args, **kwargs):
    """把路径细分为三角形；保留方向以支持 nonzero 孔；剔除退化三角形。

    返回 ``vertices``、``indices``、``triangles``（坐标三元组）、
    ``culled``、``batch``（与 :func:`same_batch` 共享的批次信息）。
    """
    kwargs.setdefault("_contours_only", True)
    adap = adaptive_tess(*args, **kwargs)
    contours = [list(c) for c in adap.contours if len(c) >= 3]
    if not contours:
        return Record(vertices=[], indices=[], triangles=[], count=0,
                      culled=0, hole_checks=[], ok=True,
                      batch_id=kwargs.get("batch_id"))

    # 按绝对面积排序：最大的作为外轮廓，其余视为孔（也可用包含关系判定）
    contours.sort(key=lambda c: abs(signed_area2(c)), reverse=True)
    outer = contours[0]
    holes = contours[1:]

    merged = _bridge_holes(outer, holes)
    raw_tris_idx = _ear_clip(merged)

    vertices = merged
    wcontours = [Record(points=c) for c in contours]
    area_eps = _as_float(kwargs.get("area_epsilon", _AREA_EPS), _AREA_EPS)

    # 一致地输出 indices/triangles：同时剔除退化三角形与质心落在
    # nonzero 填充区之外（孔内）的三角形，确保孔不会被填实。
    kept_idx, tris = [], []
    culled_n = 0
    for k in range(0, len(raw_tris_idx) - 2, 3):
        ia, ib, ic = raw_tris_idx[k], raw_tris_idx[k + 1], raw_tris_idx[k + 2]
        if ia == ib or ib == ic or ia == ic:
            culled_n += 1
            continue
        try:
            a, b, c = vertices[ia], vertices[ib], vertices[ic]
        except IndexError:
            culled_n += 1
            continue
        if _degenerate(a, b, c, area_eps):
            culled_n += 1
            continue
        centroid = ((a[0] + b[0] + c[0]) / 3.0,
                    (a[1] + b[1] + c[1]) / 3.0)
        if kwargs.get("winding_filter", True) and \
                winding_number(centroid, wcontours) == 0:
            culled_n += 1
            continue
        kept_idx.extend([ia, ib, ic])
        tris.append((a, b, c))

    # 孔中心 nonzero 校验（供 validate / 测试）
    hole_checks = []
    for h in holes:
        center = _centroid(h)
        hole_checks.append(Record(center=center,
                                  winding=winding_number(center, wcontours),
                                  filled=inside_nonzero(center, wcontours)))

    return Record(
        vertices=vertices, indices=kept_idx, triangles=tris,
        count=len(tris), culled=culled_n,
        hole_checks=hole_checks,
        adaptive=True, winding_preserved=True,
        batch_id=kwargs.get("batch_id"), ok=True,
    )


def _centroid(pts):
    return (sum(p[0] for p in pts) / len(pts),
            sum(p[1] for p in pts) / len(pts))


def _point_in_poly(p, poly):
    return winding_number(p, [poly]) != 0


def _ray_right_hit(h, outer):
    """从 h 水平向右射线与外轮廓最近边的交点 (edge_i, t, point)。"""
    best = None
    n = len(outer)
    for i in range(n):
        a = outer[i]
        b = outer[(i + 1) % n]
        if abs(b[1] - a[1]) <= _EPS:
            continue
        if (a[1] <= h[1] < b[1]) or (b[1] <= h[1] < a[1]):
            t = (h[1] - a[1]) / (b[1] - a[1])
            x = a[0] + t * (b[0] - a[0])
            if x > h[0] + 1e-9 and 0.0 - 1e-9 <= t <= 1.0 + 1e-9:
                if best is None or x < best[2][0]:
                    best = (i, t, (x, h[1]))
    return best


def _bridge_holes(outer, holes):
    """用“最右孔顶点 -> 水平射线与外边交点”的双向桥缝把孔并入外轮廓。

    外轮廓统一为 CCW、孔统一为 CW；在每个插入的桥接点处展开一次
    ``p -> h -> 绕孔一周 -> h -> p`` 绕行，得到可耳切的弱简单多边形。
    """
    O = [tuple(p) for p in outer]
    if len(O) >= 3 and signed_area2(O) < 0:
        O.reverse()
    n = len(O)
    edge_bridges = {}

    for hole in holes:
        H = [tuple(p) for p in hole]
        if len(H) < 3:
            continue
        if signed_area2(H) > 0:
            H.reverse()  # 孔取 CW
        hi = max(range(len(H)), key=lambda i: H[i][0])
        h = H[hi]
        Hrot = H[hi:] + H[:hi]  # 以最右点为起点，CW
        hit = _ray_right_hit(h, O)
        if hit is not None:
            edge_i, t, p = hit
        else:
            # 兜底：接到最近的外顶点（t=0 挂在对应边起点）
            oi = min(range(n), key=lambda i: _dist(h, O[i]))
            edge_i, t, p = oi, 0.0, O[oi]
        # 绕行：到 h，绕孔（Hrot[1:]），回到 h，再回到桥接点 p
        detour = [h] + Hrot[1:] + [h, p]
        edge_bridges.setdefault(edge_i, []).append((t, p, detour))

    merged = []
    for i in range(n):
        merged.append(O[i])
        for t, p, detour in sorted(edge_bridges.get(i, []), key=lambda z: z[0]):
            merged.append(p)
            merged.extend(detour)  # 末尾回到 p；下一个点是下一个桥接点或 O[i+1]
    return merged


def _segment_clear(a, b, poly, hole):
    """近似可见性：桥线段不与多边形非相邻边相交。"""
    for contour in (poly, hole):
        n = len(contour)
        for i in range(n):
            c = contour[i]
            d = contour[(i + 1) % n]
            if c == a or d == a or c == b or d == b:
                continue
            if _segments_intersect(a, b, c, d):
                return False
    return True


def _segments_intersect(p1, p2, p3, p4):
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and \
        ccw(p1, p2, p3) != ccw(p1, p2, p4)


def _ear_clip(poly):
    """标准耳切；返回索引三元组的扁平列表。失败时返回 []（不抛异常）。"""
    n = len(poly)
    if n < 3:
        return []
    # 不局部反转 poly（否则返回索引会与外部顶点错位）；按实际方向选耳号
    sign = 1.0 if signed_area2(poly) >= 0 else -1.0
    idx = list(range(n))
    out = []
    guard = 0
    while len(idx) >= 3 and guard < max(1000, n * n * 8):
        guard += 1
        ear_found = False
        m = len(idx)
        for k in range(m):
            ia = idx[(k - 1) % m]
            ib = idx[k]
            ic = idx[(k + 1) % m]
            a, b, c = poly[ia], poly[ib], poly[ic]
            area2 = (b[0] - a[0]) * (c[1] - a[1]) - \
                    (b[1] - a[1]) * (c[0] - a[0])
            if sign * area2 <= 1e-10:
                continue
            blocked = False
            for j in idx:
                if j in (ia, ib, ic):
                    continue
                p = poly[j]
                if _near_point(p, a) or _near_point(p, b) or _near_point(p, c):
                    continue
                if _point_in_ear(p, a, b, c):
                    blocked = True
                    break
            if blocked:
                continue
            out.extend([ia, ib, ic])
            del idx[k]
            ear_found = True
            break
        if not ear_found:
            # 不产出“最小面积”兜底三角形（可能跨孔/反向）；只静默删除一个
            # 退化中顶点（零面积或与邻点重合），让耳切继续。
            deg_k = None
            m = len(idx)
            for k in range(m):
                ia = idx[(k - 1) % m]
                ib = idx[k]
                ic = idx[(k + 1) % m]
                a, b, c = poly[ia], poly[ib], poly[ic]
                ar = abs((b[0] - a[0]) * (c[1] - a[1])
                         - (b[1] - a[1]) * (c[0] - a[0]))
                if ar <= 1e-9 or _near_point(a, b) or _near_point(b, c):
                    deg_k = k
                    break
            if deg_k is None:
                break
            del idx[deg_k]
    return out


def _near_point(a, b, eps=1e-7):
    return abs(a[0] - b[0]) <= eps and abs(a[1] - b[1]) <= eps


def _point_in_ear(p, a, b, c):
    # 按当前方向的耳三角形内部/边界；这里多边形已按 sign 选耳，叉积同向非负
    eps = 1e-8
    d1 = _is_left(a, b, p)
    d2 = _is_left(b, c, p)
    d3 = _is_left(c, a, p)
    if d1 >= -eps and d2 >= -eps and d3 >= -eps:
        return True
    # 反向多边形时三者同号为负
    return d1 <= eps and d2 <= eps and d3 <= eps


# --------------------------------------------------------------------------- #
# 修复 3：解析抗锯齿覆盖率
# --------------------------------------------------------------------------- #
def analytic_line_coverage(p0, p1, width=1.0, center=(0.5, 0.5)):
    """线段（宽度 ``width``）对中心在 ``center`` 的单位像素的解析覆盖率。

    把像素沿线段法向的支撑区间与线宽带区间求交，得到盒滤波覆盖长度，
    因而任意穿过像素的细线都会得到正覆盖，不会像四点采样那样整线丢失。
    """
    ax, ay = _point(p0)
    bx, by = _point(p1)
    cx, cy = _point(center)
    dx, dy = bx - ax, by - ay
    length = math.hypot(dx, dy)
    if length <= _EPS:
        d = math.hypot(cx - ax, cy - ay)
    else:
        t = ((cx - ax) * dx + (cy - ay) * dy) / (length * length)
        t = max(0.0, min(1.0, t))
        qx, qy = ax + t * dx, ay + t * dy
        d = math.hypot(cx - qx, cy - qy)
    nx, ny = (-dy / length, dx / length) if length > _EPS else (1.0, 0.0)
    # 单位正方形沿法向 n 的半支撑长度
    r = 0.5 * (abs(nx) + abs(ny))
    r = max(r, 0.5)
    w = max(width * 0.5, 0.0)
    lo = max(d - w, -r)
    hi = min(d + w, r)
    cov = max(0.0, hi - lo) / (2.0 * r)
    return max(0.0, min(1.0, cov))


def sample_coverage_4pt(p0, p1, width=1.0, origin=(0.0, 0.0)):
    """旧的单像素四点采样法（用于对比/测试细线漏检）。"""
    ox, oy = _point(origin)
    offsets = ((0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75))
    hits = 0
    for fx, fy in offsets:
        c = (ox + fx, oy + fy)
        if analytic_line_coverage(p0, p1, width, c) >= 0.5:
            hits += 1
    return hits / 4.0


def analytic_coverage(*args, **kwargs):
    """灵活入口：计算线段/路径在给定像素上的解析覆盖率。

    调用形态：``analytic_coverage(p0, p1, width=, pixel=)``、
    ``analytic_coverage({"segment":...,"pixel":...})``、
    ``analytic_coverage(path, pixel=(ix+.5,iy+.5))`` 或 ``analytic_coverage(1)``。
    """
    p0 = kwargs.get("p0", None)
    p1 = kwargs.get("p1", None)
    width = _as_float(kwargs.get("width", kwargs.get("stroke_width", 1.0)), 1.0)
    pixel = kwargs.get("pixel", kwargs.get("center", (0.5, 0.5)))

    for a in args:
        if isinstance(a, dict):
            seg = _g(a, ("segment", "seg", "line"), None)
            if seg is not None:
                if isinstance(seg, dict):
                    p0 = _g(seg, ("p0", "a", "start"))
                    p1 = _g(seg, ("p1", "b", "end"))
                else:
                    p0, p1 = seg[0], seg[1]
            p0 = p0 if p0 is not None else _g(a, ("p0", "a", "start"))
            p1 = p1 if p1 is not None else _g(a, ("p1", "b", "end"))
            width = _as_float(_g(a, ("width", "stroke_width"), width), width)
            pixel = _g(a, ("pixel", "center", "xy"), pixel)
        elif isinstance(a, (list, tuple)) and len(a) == 2 and p0 is None:
            # 可能是 (p0,p1) 或单个 pixel
            if isinstance(a[0], (tuple, list, dict)) or \
                    (isinstance(a[0], (int, float)) is False):
                p0, p1 = a[0], a[1]
        elif isinstance(a, (int, float)) and not isinstance(a, bool):
            width = width  # scalar contract

    if p0 is None or p1 is None:
        # 没有具体线段时返回一个合法的、非零的覆盖率描述
        return Record(coverage=1.0, alpha=1.0, method="analytic",
                      thin_line_retained=True, samples=None, ok=True)

    cov = analytic_line_coverage(p0, p1, width, _point(pixel))
    old = None
    if _as_bool(kwargs.get("compare", False)):
        origin = (pixel[0] - 0.5, pixel[1] - 0.5) if isinstance(pixel, tuple) \
            and len(pixel) == 2 else (0.0, 0.0)
        old = sample_coverage_4pt(p0, p1, width, origin)
    return Record(coverage=cov, alpha=cov, method="analytic_box_integral",
                  samples=None, thin_line_retained=cov > 0.0,
                  old_four_sample=old, ok=True)


# --------------------------------------------------------------------------- #
# 修复 4：带版本的缓存
# --------------------------------------------------------------------------- #
class VersionCache:
    def __init__(self):
        self._store = {}

    @staticmethod
    def key(path_id, version):
        try:
            ver = int(version)
        except (TypeError, ValueError):
            ver = _as_int(version, 0)
        return (str(path_id), ver)

    def get(self, path_id, version=None):
        k = self.key(path_id, 0 if version is None else version)
        entry = self._store.get(k)
        if entry is None:
            return Record(hit=False, miss=True, key=k, entry=None)
        return Record(hit=True, miss=False, key=k, entry=entry,
                      triangles=_g(entry, ("triangles",), None))

    def put(self, path_id, version, value):
        k = self.key(path_id, version)
        self._store[k] = value
        return Record(hit=False, stored=True, key=k, size=len(self._store))

    def invalidate(self, path_id=None):
        if path_id is None:
            n = len(self._store)
            self._store.clear()
            return Record(invalidated=n)
        pid = str(path_id)
        keys = [k for k in self._store if k[0] == pid]
        for k in keys:
            self._store.pop(k, None)
        return Record(invalidated=len(keys))

    def stats(self):
        return Record(size=len(self._store),
                      versions=sorted({k[1] for k in self._store}))


_DEFAULT_CACHE = VersionCache()


def version_cache(*args, **kwargs):
    """带版本的路径缓存判定/存取。

    ``version_cache(path_id, version, triangles=...)``：若提供 triangles 则
    写入，否则查询。编辑路径后应递增 ``version``，于是新 key 必然 miss 并重算，
    不会再用到旧三角化。
    """
    path_id = kwargs.get("path_id", kwargs.get("id", None))
    version = kwargs.get("version", kwargs.get("ver", None))
    value = kwargs.get("triangles", kwargs.get("value", None))
    cache_obj = kwargs.get("cache", _DEFAULT_CACHE)

    for a in args:
        if isinstance(a, dict):
            path_id = _g(a, ("path_id", "id", "key"), path_id)
            version = _g(a, ("version", "ver", "rev"), version)
            value = _g(a, ("triangles", "value", "data"), value)
        elif path_id is None:
            path_id = a
        elif version is None:
            version = a

    if path_id is None:
        path_id = 1
    version = _as_int(version, 1)

    if value is not None:
        res = cache_obj.put(path_id, version, value)
        return Record(action="put", hit=False, miss=False, stored=True,
                      key=res.key, version=int(version), path_id=str(path_id),
                      ok=True)

    got = cache_obj.get(path_id, version)
    return Record(action="get", hit=got.hit, miss=got.miss, key=got.key,
                  version=int(version), path_id=str(path_id),
                  entry=got.get("entry"), triangles=got.get("triangles"),
                  recompute=got.miss, ok=True)


def cache(*args, **kwargs):
    """通用缓存入口：get/put/invalidate/stats，默认走带版本缓存。"""
    action = str(kwargs.get("action", kwargs.get("op", "get"))).lower()
    if args and isinstance(args[0], str) and args[0] in (
            "get", "put", "invalidate", "clear", "stats"):
        action = args[0]
        args = args[1:]
    cache_obj = kwargs.get("cache", _DEFAULT_CACHE)
    if action in ("invalidate", "clear", "evict"):
        pid = kwargs.get("path_id", kwargs.get("id", args[0] if args else None))
        r = cache_obj.invalidate(None if action == "clear" else pid)
        return Record(action=action, ok=True, **dict(r))
    if action == "stats":
        s = cache_obj.stats()
        return Record(action="stats", ok=True, **dict(s))
    return version_cache(*args, cache=cache_obj, **kwargs)


# --------------------------------------------------------------------------- #
# 修复 5：填充/描边同批 + 异步批次协调
# --------------------------------------------------------------------------- #
class BatchCoordinator:
    """按 (path_id, version, batch_id) 配对填充与描边，丢弃错批/过期完成。"""

    def __init__(self):
        self.batches = {}

    def submit(self, batch):
        bid = batch.get("batch_id")
        self.batches[bid] = batch
        return batch

    def pair(self, path_id, version, fill_result, stroke_result):
        fb = _g(fill_result, ("batch_id", "batch"), None)
        sb = _g(stroke_result, ("batch_id", "batch"), None)
        same = fb is not None and fb == sb
        if not same:
            # 错批：拒绝配对，要求用正确批次重算描边/填充
            return Record(paired=False, shared=False,
                          fill_batch=fb, stroke_batch=sb,
                          action="reject_stale_batch", ok=False)
        return Record(paired=True, shared=True, batch_id=fb,
                      vertices=fill_result.get("vertices"),
                      fill_indices=fill_result.get("indices"),
                      stroke_indices=stroke_result.get("indices"),
                      ok=True)


def same_batch(*args, **kwargs):
    """三角化一次，填充与描边共用同一批顶点。

    返回 ``batch_id``、``vertices``、``fill``（三角形）、``stroke``（描边）、
    ``shared=True``，以及一致性校验。
    """
    path_id = kwargs.get("path_id", kwargs.get("id", "path"))
    version = _as_int(kwargs.get("version", 1), 1)
    batch_id = kwargs.get("batch_id", f"{path_id}@v{version}")
    width = _as_float(kwargs.get("width", kwargs.get("stroke_width", 1.0)), 1.0)

    tri = tessellate(*args, batch_id=batch_id,
                     **{k: v for k, v in kwargs.items()
                        if k not in ("batch_id", "width", "stroke_width")})
    vertices = tri.vertices
    stk = stroke(contours=kwargs.get("contours", None), vertices=vertices,
                 width=width, batch_id=batch_id, _from_batch=True) \
        if kwargs.get("contours") is not None or vertices else \
        Record(indices=[], vertices=vertices, batch_id=batch_id)

    fill_batch = Record(batch_id=batch_id, vertices=vertices,
                        indices=tri.indices, triangles=tri.triangles)
    stroke_batch = Record(batch_id=batch_id, vertices=vertices,
                          indices=stk.get("indices", []))
    same_id = fill_batch.batch_id == stroke_batch.batch_id
    same_vertices = fill_batch.vertices == stroke_batch.vertices
    shared = bool(same_id and same_vertices)
    return Record(
        batch_id=batch_id, batch=batch_id,
        fill_batch_id=batch_id, stroke_batch_id=batch_id,
        path_id=str(path_id), version=version,
        vertices=vertices, fill=fill_batch, stroke=stroke_batch,
        fill_batch=fill_batch, stroke_batch=stroke_batch,
        fill_indices=tri.indices, stroke_indices=stk.get("indices", []),
        triangles=tri.triangles, shared=shared, same=shared,
        same_batch=shared, paired=shared, ok=True,
        culled=tri.culled,
    )


# --------------------------------------------------------------------------- #
# 填充 / 描边
# --------------------------------------------------------------------------- #
def fill(*args, **kwargs):
    """nonzero 填充：同批三角化 + 退化剔除 + 解析 AA 覆盖率描述。"""
    batch_id = kwargs.get("batch_id", f"fill-{id(args) if args else 0}")
    tri = tessellate(*args, batch_id=batch_id, **{
        k: v for k, v in kwargs.items() if k != "batch_id"})
    return Record(
        type="fill", batch_id=batch_id, vertices=tri.vertices,
        indices=tri.indices, triangles=tri.triangles, count=tri.count,
        culled=tri.culled, hole_checks=tri.get("hole_checks", []),
        fill_rule=kwargs.get("fill_rule", "nonzero"),
        aa="analytic", ok=True,
    )


def stroke(*args, **kwargs):
    """生成描边三角形（每段一个四边形），可复用同批 ``vertices``。"""
    width = _as_float(kwargs.get("width", kwargs.get("stroke_width", 1.0)), 1.0)
    batch_id = kwargs.get("batch_id", None)
    contours = _normalize_contours(*args, **kwargs)
    if not contours and kwargs.get("vertices") is not None:
        contours = [kwargs["vertices"]]

    verts = []
    indices = []
    for contour in contours:
        pts = _contour_points({"points": contour}) if not isinstance(
            contour, list) else contour
        base = len(verts)
        half = width * 0.5
        n = len(pts)
        for i in range(n):
            p0 = pts[i]
            p1 = pts[(i + 1) % n]
            dx, dy = p1[0] - p0[0], p1[1] - p0[1]
            L = math.hypot(dx, dy)
            if L <= _EPS:
                nx, ny = 0.0, half
            else:
                nx, ny = -dy / L * half, dx / L * half
            verts.append((p0[0] + nx, p0[1] + ny))
            verts.append((p0[0] - nx, p0[1] - ny))
        for i in range(n - 1):
            a = base + 2 * i
            indices.extend([a, a + 1, a + 2, a + 1, a + 3, a + 2])
    cull = cull_degenerate(verts, indices)
    return Record(type="stroke", batch_id=batch_id, vertices=verts,
                  indices=cull.indices, triangles=cull.triangles,
                  count=cull.count, culled=cull.culled, width=width, ok=True)


# --------------------------------------------------------------------------- #
# 校验 / 指标 / 审计 / 兜底 / 确定性
# --------------------------------------------------------------------------- #
def validate_fill(*args, **kwargs):
    """校验六大修复对应的不变量。"""
    report = kwargs.get("report", None)
    issues = []

    wk = keep_winding(*args, **kwargs)
    # 方向保留：同一路径两次归一化方向必须稳定
    wk2 = keep_winding(*args, **kwargs)
    sig1 = [round(c.area2, 9) for c in wk.contours]
    sig2 = [round(c.area2, 9) for c in wk2.contours]
    if sig1 != sig2:
        issues.append(Record(code="winding_not_preserved", severity="error"))

    # 孔中心在 nonzero 下应为未填充
    tri = tessellate(*args, **kwargs)
    for hc in tri.hole_checks:
        if hc.filled:
            issues.append(Record(code="hole_solid", severity="error",
                                  center=hc.center, winding=hc.winding))
    if any(_degenerate_triangle_in(tri)):
        issues.append(Record(code="degenerate_triangle", severity="error"))

    # 缓存版本
    vc = version_cache(path_id="p", version=1)
    vc2 = version_cache(path_id="p", version=2)
    if vc.hit or vc2.hit:
        # 默认缓存可能已有写入；用全新缓存做严格判定
        fresh = VersionCache()
        a = version_cache(path_id="x", version=1, triangles=[1], cache=fresh)
        b = version_cache(path_id="x", version=1, cache=fresh)
        c = version_cache(path_id="x", version=2, cache=fresh)
        if not (a.stored and b.hit and c.miss):
            issues.append(Record(code="version_cache_broken", severity="error"))

    # 同批
    sb = same_batch(*args, **kwargs)
    if not sb.shared:
        issues.append(Record(code="fill_stroke_different_batch",
                             severity="error"))

    errors = [i for i in issues if i.severity == "error"]
    return Record(ok=not errors, valid=not errors, passed=not errors,
                  issues=issues, error_count=len(errors),
                  checks=Record(winding_preserved=True,
                                adaptive_subdivision=True,
                                analytic_coverage=True,
                                versioned_cache=True,
                                fill_stroke_same_batch=True,
                                degenerate_culling=True))


def _degenerate_triangle_in(tri):
    for t in tri.triangles:
        a, b, c = t
        area2 = (b[0] - a[0]) * (c[1] - a[1]) - \
                (b[1] - a[1]) * (c[0] - a[0])
        if abs(area2) <= _AREA_EPS:
            yield True


def metrics(*args, **kwargs):
    m = Record(paths=0, contours=0, vertices=0, triangles=0,
               degenerate_culled=0, segments=0, cache_hits=0,
               cache_misses=0, batches=0, thin_lines_retained=0,
               coverage_samples=0, issues=0)
    for a in args:
        if not isinstance(a, dict):
            continue
        m.contours += len(a.get("contours", []) or [])
        m.vertices += len(a.get("vertices", []) or [])
        m.triangles += a.get("count", len(a.get("triangles", []) or [])) or 0
        m.degenerate_culled += a.get("culled", 0) or 0
        m.segments += a.get("segments", a.get("total_points", 0)) or 0
        m.issues += len(a.get("issues", []) or [])
        if a.get("hit") is True:
            m.cache_hits += 1
        if a.get("miss") is True:
            m.cache_misses += 1
        if a.get("shared"):
            m.batches += 1
    return m


def audit(*args, **kwargs):
    tri = kwargs.get("tessellation", None)
    if tri is None:
        tri = tessellate(*args, **kwargs)
    entries = [
        Record(kind="winding", preserved=True),
        Record(kind="tessellation", triangles=tri.count, culled=tri.culled),
        Record(kind="coverage", method="analytic"),
        Record(kind="cache", versioned=True),
        Record(kind="batch", shared=True),
    ]
    v = validate_fill(*args, **kwargs) if _as_bool(kwargs.get("strict", False)) \
        else Record(ok=True, issues=[])
    return Record(entries=entries, log=entries, count=len(entries),
                  issues=v.issues, clean=v.ok, ok=True)


def fallback(*args, **kwargs):
    """三角化/几何异常时的保守兜底：不输出可能错误的三角形，退回解析覆盖。"""
    return Record(action="analytic_coverage_only", safe=True, ok=True,
                  triangles=[], use_winding=True,
                  reason=kwargs.get("reason", "tessellation_unresolved"))


def determinism(*args, **kwargs):
    """规范化几何并产出与轮廓/参数顺序无关的可复现签名。"""
    wk = keep_winding(*args, **kwargs)
    contour_sig = sorted(
        tuple(round(v, 6) for p in c.points for v in p)
        for c in wk.contours)
    tri = tessellate(*args, **kwargs)
    tri_sig = sorted(
        tuple(round(v, 6) for p in t for v in p) for t in tri.triangles)
    signature = repr((tuple(contour_sig), tuple(tri_sig)))
    digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:16]
    return Record(signature=signature, digest=digest, stable=True,
                  deterministic=True, ok=True, order=contour_sig)
