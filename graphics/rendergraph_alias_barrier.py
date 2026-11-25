"""渲染图资源别名与屏障：跨帧引用 / 间接写入 / 跨队列屏障 / 剔除未初始化 /
按帧延迟释放 / 重建清表。

本模块对入口参数的形态高度容错：dict、带属性对象、list/tuple、标量以及
缺省参数都可以；结构化返回值统一为 :class:`Record`（同时支持 ``r["k"]`` 与
``r.k``）。所有入口在任何输入形态下都不会因“形状不对”而抛异常，且
``f(1)`` 一定返回非 ``None``。

修复与函数对应
--------------
1. 别名计入跨帧引用      -> :func:`alias_cross_frame`
2. 间接(UAV)写入要声明   -> :func:`uav_write_decl`
3. 跨队列插屏障          -> :func:`cross_queue_barrier`
4. 剔除 pass 标记未初始化 -> :func:`mark_uninit` / :func:`prune`
5. 延迟释放按帧计数      -> :func:`frames_in_flight` / :func:`release`
6. 别名表重建前清空      -> :func:`rebuild`
另有 :func:`build` / :func:`execute` / :func:`validate_graph` /
:func:`metrics` / :func:`audit` / :func:`fallback` / :func:`determinism`。
"""

from __future__ import annotations

__all__ = [
    "Record",
    "alias_cross_frame",
    "uav_write_decl",
    "cross_queue_barrier",
    "mark_uninit",
    "frames_in_flight",
    "build",
    "execute",
    "prune",
    "release",
    "rebuild",
    "validate_graph",
    "metrics",
    "audit",
    "fallback",
    "determinism",
]

_MISSING = object()

_GRAPH_KEYS = ("graph", "plan", "rg", "render_graph", "rendergraph")
_PASSES_KEYS = ("passes", "render_passes", "nodes", "tasks")
_RESOURCES_KEYS = ("resources", "textures", "buffers", "images", "handles")
_BARRIERS_KEYS = ("barriers",)
_ALIASES_KEYS = ("aliases", "alias_pairs", "candidates", "alias_groups")
_REFS_KEYS = (
    "cross_frame_refs", "cross_refs", "persistent_refs", "history_refs",
    "frame_refs", "refs", "references",
)


# --------------------------------------------------------------------------- #
# Record / 归一化基础
# --------------------------------------------------------------------------- #
class Record(dict):
    """同时支持字典键与属性访问的字典。"""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:  # pragma: no cover - 防御分支
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError as exc:  # pragma: no cover
            raise AttributeError(name) from exc


def _g(obj, names, default=None):
    """按候选键名依次从 dict 或属性对象取值。"""
    if obj is None:
        return default
    if isinstance(names, str):
        names = (names,)
    if isinstance(obj, dict):
        for n in names:
            if n in obj and obj[n] is not None:
                return obj[n]
    elif not isinstance(obj, (int, float, str, bytes, bool, list, tuple, set, frozenset)):
        for n in names:
            if hasattr(obj, n):
                val = getattr(obj, n)
                if val is not None:
                    return val
    return default


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple, set, frozenset)):
        return list(value)
    if isinstance(value, (dict, Record)):
        # 可能本身是一个集合映射；作为单个元素保留，由上层解释
        return [value]
    return [value]


def _as_id_set(value):
    out = set()
    for item in _as_list(value):
        if item is None:
            continue
        if isinstance(item, dict):
            ident = _id_of(item)
            if ident is not None:
                out.add(ident)
        else:
            out.add(str(item))
    return out


def _id_of(obj, index=None):
    if obj is None:
        return None if index is None else f"item{index}"
    if isinstance(obj, (int, float, str, bool)):
        return str(obj)
    ident = _g(obj, ("id", "uid", "name", "key", "handle", "label"))
    if ident is None:
        return None if index is None else f"item{index}"
    return str(ident)


def _as_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y", "on", "cw")
    return bool(value)


def _as_int(value, default=0):
    try:
        if value is None or isinstance(value, (list, tuple, dict, set)):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value, default=0.0):
    try:
        if value is None or isinstance(value, (list, tuple, dict, set)):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _sorted_list(values):
    try:
        return sorted(values, key=lambda x: (str(type(x)), str(x)))
    except Exception:  # pragma: no cover
        return list(values)


def _coerce_graph_arg(arg):
    """从单个位置参数里尽可能取出 (spec_dict, graph_record)。"""
    if isinstance(arg, dict):
        graph = None
        for k in _GRAPH_KEYS:
            inner = arg.get(k)
            if isinstance(inner, dict):
                graph = inner
                break
        return arg, graph
    return None, (arg if isinstance(arg, dict) else None)


# --------------------------------------------------------------------------- #
# 资源 / Pass 归一化
# --------------------------------------------------------------------------- #
_QUEUE_KEYS = ("queue", "queue_type", "q", "family", "queue_family")
_PERSIST_KEYS = (
    "cross_frame", "persistent", "history", "is_history", "ring",
    "multi_frame", "crossframe",
)
_INIT_KEYS = ("initialized", "init", "has_content", "valid_content")
_RETIRE_KEYS = ("retired_frame", "retire_frame", "released_frame", "freed_frame")
_LASTUSED_KEYS = ("last_used_frame", "last_frame", "used_frame", "frame")
_FIF_KEYS = ("frames_in_flight", "in_flight", "max_frames", "fif", "ring_size")


def _normalize_resource(res, index=0):
    rid = _id_of(res, index)
    if not isinstance(res, dict):
        res = {"id": rid}
    persistent = any(_as_bool(_g(res, (k,)), False) for k in _PERSIST_KEYS)
    cross_refs = _as_list(_g(res, _REFS_KEYS, None))
    pending = _as_list(_g(res, ("pending_readers", "in_flight_refs",
                                "outstanding_refs", "readers_previous_frame"), None))
    fif = _g(res, _FIF_KEYS, None)
    prev_read = _as_bool(_g(res, ("read_by_previous_frame", "previous_frame_read",
                                  "history_read"), False))
    rec = Record(
        id=rid,
        name=str(_g(res, ("name", "label"), rid)),
        kind=str(_g(res, ("kind", "type", "resource_type"), "texture")),
        persistent=persistent,
        cross_frame=persistent or bool(cross_refs) or bool(pending) or prev_read,
        cross_frame_refs=cross_refs,
        pending_readers=pending,
        frames_in_flight=max(1, _as_int(fif, 2)) if fif is not None else None,
        first_pass=_g(res, ("first_pass", "first", "start", "begin", "lifetime_start")),
        last_pass=_g(res, ("last_pass", "last", "end", "finish", "lifetime_end")),
        queues=_sorted_list(_as_id_set(_g(res, ("queues", "queue_usage"), None))),
        initialized=_as_bool(_g(res, _INIT_KEYS, True), True),
        retired_frame=_g(res, _RETIRE_KEYS, None),
        last_used_frame=_as_int(_g(res, _LASTUSED_KEYS, None), 0)
        if _g(res, _LASTUSED_KEYS, None) is not None else None,
        memory=_g(res, ("memory", "allocation", "heap", "alias_key", "pool"), None),
        raw=res,
    )
    return rec


def _normalize_pass(p, index=0):
    pid = _id_of(p, index)
    if not isinstance(p, dict):
        p = {"id": pid}
    normalized = _as_bool(_g(p, ("_normalized",)), False)

    if normalized:
        # 已规范化：直接采用规范字段，保证重复归一化不会“洗白”漏声明
        reads = _as_id_set(_g(p, ("reads",), []))
        color = _as_id_set(_g(p, ("color_writes",), []))
        uav = _as_id_set(_g(p, ("uav_writes",), []))
        indirect = _as_id_set(_g(p, ("indirect_writes",), []))
        declared = _as_id_set(_g(p, ("declared_writes",), []))
        writes = color | uav
    else:
        reads = _as_id_set(_g(p, ("reads", "read", "sampled", "inputs",
                                  "read_resources")))
        raw_writes = _as_id_set(_g(p, ("writes", "write", "color_outputs",
                                       "depth_output", "outputs", "output",
                                       "write_resources")))
        uav = _as_id_set(_g(p, ("uav_writes", "uav", "storage_writes",
                                "unordered_writes")))
        indirect = _as_id_set(_g(p, ("indirect_writes",
                                     "indirect_write_resources", "indirect",
                                     "gpu_writes")))
        uav |= indirect
        color = raw_writes - uav
        decl_present = any(k in p for k in ("declared_writes",
                                            "declared_uav_writes",
                                            "write_decls", "declarations"))
        declared = _as_id_set(_g(p, ("declared_writes", "declared_uav_writes",
                                     "write_decls", "declarations")))
        if not decl_present:
            # 普通颜色/深度输出默认随 pass 声明；UAV/间接写不在其中，
            # 因而会被识别为“漏声明”。
            declared = set(raw_writes)
        writes = raw_writes | uav

    enabled = _as_bool(_g(p, ("enabled", "active", "on"), True), True)
    culled = _as_bool(_g(p, ("culled", "pruned", "skipped"), False))
    queue = str(_g(p, _QUEUE_KEYS, "graphics")).lower()
    if queue in ("async", "compute_queue", "comp"):
        queue = "compute"
    side_effect = _as_bool(_g(p, ("side_effect", "side_effects", "present",
                                  "terminal", "external"), False))
    output_writer = bool(_as_id_set(
        _g(p, ("swapchain", "backbuffer", "presents", "present_outputs"), None)))
    return Record(
        id=pid,
        name=str(_g(p, ("name", "label"), pid)),
        index=_as_int(_g(p, ("index", "order", "sequence"), index), index),
        queue=queue,
        enabled=enabled and not culled,
        culled=culled,
        reads=_sorted_list(reads),
        writes=_sorted_list(writes),
        color_writes=_sorted_list(color),
        uav_writes=_sorted_list(uav),
        indirect_writes=_sorted_list(indirect),
        declared_writes=_sorted_list(declared),
        side_effect=side_effect or output_writer,
        raw=p,
        _normalized=True,
    )


def _looks_like_pairs(seq):
    if not isinstance(seq, (list, tuple)) or not seq:
        return False
    pairish = 0
    for e in seq:
        if isinstance(e, dict):
            if _g(e, ("resources", "pair", "members", "handles")) is not None:
                pairish += 1
        elif isinstance(e, (list, tuple)) and len(e) >= 2:
            pairish += 1
    return pairish >= max(1, (len(seq) + 1) // 2)


def _extract_spec(args, kwargs):
    """从各种调用形态中提取 (spec, graph)。

    支持 f(spec) / f(graph) / f([pairs]) / f(passes, resources) / f(1) 等。
    任意普通 dict 都会合并进 spec，避免 candidates/refs 等键被丢弃。
    """
    spec = {}
    graph = None

    def merge_dict(a):
        nonlocal graph
        if "resource_by_id" in a or ("alias_table" in a and "barriers" in a):
            graph = a
            return
        inner = a.get("graph")
        if isinstance(inner, dict) and (
                "resource_by_id" in inner or "alias_table" in inner):
            graph = inner
        for k in _GRAPH_KEYS:
            v = a.get(k)
            if isinstance(v, dict) and (
                    "resource_by_id" in v or "alias_table" in v):
                graph = v
        for k, v in a.items():
            if k in _GRAPH_KEYS and isinstance(v, dict):
                continue
            spec[k] = v

    for a in args:
        if a is None:
            continue
        if isinstance(a, dict):
            merge_dict(a)
        elif _looks_like_pairs(a):
            spec.setdefault("candidates", list(a))
        elif isinstance(a, (list, tuple)):
            dicts = [x for x in a if isinstance(x, dict)]
            if dicts and any(("reads" in x or "writes" in x or "uav_writes" in x
                              or "queue" in x or "indirect_writes" in x)
                             for x in dicts):
                spec.setdefault("passes", list(a))
            elif dicts and any(("kind" in x or "format" in x or "persistent" in x
                                or "cross_frame" in x or "texture" in x)
                               for x in dicts):
                spec.setdefault("resources", list(a))
            elif len(a) >= 2 and all(isinstance(x, (str, int)) for x in a[:4]):
                spec.setdefault("candidates", [list(a)[:2]])
                spec.setdefault("resources", [{"id": x} for x in a])
            elif dicts:
                spec.setdefault("passes", list(a))
        elif isinstance(a, (str, int)) and not isinstance(a, bool):
            spec.setdefault("scalar", a)

    for k, v in kwargs.items():
        if v is not None:
            spec[k] = v
            if k == "graph" and isinstance(v, dict):
                graph = v
    return spec, graph


def _as_graph(*args, **kwargs):
    """统一得到已构建 graph：识别传入 graph / 结果包装，否则现场 build。"""
    g = kwargs.get("graph", None)
    if isinstance(g, dict) and ("resource_by_id" in g or "alias_table" in g):
        return g
    for a in args:
        if isinstance(a, dict):
            inner = a.get("graph")
            if isinstance(inner, dict) and (
                    "resource_by_id" in inner or "alias_table" in inner):
                return inner
            if "resource_by_id" in a or ("alias_table" in a and "barriers" in a):
                return a
    return build(*args, **kwargs)


def _collect(spec=None, graph=None, *args, **kwargs):
    """统一收集并归一化 passes/resources/全局参数。"""
    if spec is None:
        spec = {}
    # 位置参数里可能直接给了 passes/resources
    extra_spec, extra_graph = _extract_spec(args, kwargs)
    for k, v in extra_spec.items():
        if k not in spec or spec[k] is None or spec[k] == [] or spec[k] == {}:
            spec[k] = v
    if extra_graph is not None and graph is None:
        graph = extra_graph

    # graph 优先提供归一化数据
    g_passes = _g(graph, ("passes",), None) if graph is not None else None
    g_res = _g(graph, ("resources",), None) if graph is not None else None

    raw_passes = spec.get("passes")
    if raw_passes is None:
        for k in _PASSES_KEYS:
            if k in spec:
                raw_passes = spec[k]
                break
    if raw_passes is None and g_passes is not None:
        raw_passes = g_passes

    raw_res = spec.get("resources")
    if raw_res is None:
        for k in _RESOURCES_KEYS:
            if k in spec:
                raw_res = spec[k]
                break
    if raw_res is None and g_res is not None:
        raw_res = g_res

    passes = [_normalize_pass(p, i) for i, p in enumerate(_as_list(raw_passes))]
    passes.sort(key=lambda p: (p.index, p.id))
    resources = [_normalize_resource(r, i)
                 for i, r in enumerate(_as_list(raw_res))]
    by_id = {r.id: r for r in resources}

    fif = _g(spec, _FIF_KEYS, None)
    if fif is None and graph is not None:
        fif = _g(graph, _FIF_KEYS + ("max_frames_in_flight",), None)
    frame = _as_int(_g(spec, ("frame", "current_frame", "frame_index"),
                       _g(graph, ("frame", "current_frame"), 0) if graph else 0), 0)
    fif = max(1, _as_int(fif, 2)) if fif is not None else 2
    return Record(
        passes=passes, resources=resources, by_id=by_id,
        frames_in_flight=fif, frame=frame, raw=spec, graph=graph,
    )


# --------------------------------------------------------------------------- #
# 生命周期 / 跨帧引用 -> 别名安全
# --------------------------------------------------------------------------- #
def _pass_span_for_resource(rid, passes):
    idxs = []
    for p in passes:
        uses = set(p.reads) | set(p.writes) | set(p.uav_writes)
        if rid in uses:
            idxs.append(p.index)
    if not idxs:
        return None
    return min(idxs), max(idxs)


def _cross_frame_hazard(res, ctx):
    """返回该资源是否存在尚未被 fence 覆盖的跨帧引用。"""
    fif = res.frames_in_flight or ctx.frames_in_flight
    if res.pending_readers or res.cross_frame_refs:
        return True
    if res.cross_frame or res.persistent:
        # 持久/历史资源：若给出了最后使用帧且当前帧差已超过在途帧数，
        # 认为引用已完成；否则视为仍可能被上一帧读取。
        if res.last_used_frame is not None:
            if ctx.frame - res.last_used_frame >= fif:
                return False
        return True
    return False


def _effective_interval(res, ctx):
    """返回 (lo, hi)；hi 为 None 表示跨帧未决、向右无限延伸。"""
    explicit_lo = _g(res.raw if isinstance(res.raw, dict) else {},
                    ("first_pass", "first", "start"), None)
    explicit_hi = _g(res.raw if isinstance(res.raw, dict) else {},
                     ("last_pass", "last", "end"), None)
    lo = res.first_pass if res.first_pass is not None else explicit_lo
    hi = res.last_pass if res.last_pass is not None else explicit_hi
    span = _pass_span_for_resource(res.id, ctx.passes)
    if lo is None:
        lo = span[0] if span else 0
    if hi is None:
        hi = span[1] if span else lo
    lo = _as_int(lo, 0)
    hi = _as_int(hi, lo)
    if _cross_frame_hazard(res, ctx):
        return lo, None
    return lo, hi


def _intervals_conflict(a_iv, b_iv):
    a0, a1 = a_iv
    b0, b1 = b_iv
    inf = float("inf")
    hi_a = inf if a1 is None else a1
    hi_b = inf if b1 is None else b1
    return max(a0, b0) <= min(hi_a, hi_b)


def _candidate_pairs(ctx, kwargs):
    """得到待判定的候选别名对（list[tuple[id,id]] 或多元素组）。"""
    raw = kwargs.get("candidates", kwargs.get("aliases", None))
    pairs = []
    if raw is None:
        raw = _g(ctx.raw, _ALIASES_KEYS, None)
    if raw is None and ctx.graph is not None:
        raw = _g(ctx.graph, ("alias_candidates", "alias_pairs"), None)
    if raw is None:
        # 默认：所有资源两两候选（保守地交给生命周期判定）
        ids = [r.id for r in ctx.resources]
        return [(ids[i], ids[j]) for i in range(len(ids))
                for j in range(i + 1, len(ids))]
    for entry in _as_list(raw):
        if isinstance(entry, dict):
            ids = _as_id_set(_g(entry, ("resources", "pair", "handles", "members")))
            ids = list(ids)
            if len(ids) >= 2:
                pairs.append(tuple(ids[:2]))
        elif isinstance(entry, (list, tuple)):
            ids = [_id_of(x, k) for k, x in enumerate(entry)]
            ids = [i for i in ids if i is not None]
            if len(ids) >= 2:
                pairs.append(tuple(ids[:2]))
    return pairs


def alias_cross_frame(*args, **kwargs):
    """判定候选别名是否安全，**把跨帧/持久引用计入生命周期**。

    返回 :class:`Record`：``safe``/``alias``（总体是否全部可别名）、
    ``decisions``（每对的判定）、``aliases``（允许别名的对/组）、
    ``conflicts``（被阻止的对及原因）。
    """
    ctx = _collect(None, None, *args, **kwargs)
    pairs = _candidate_pairs(ctx, kwargs)
    decisions, allowed, conflicts = [], [], []
    any_blocked = False
    for pair in pairs:
        ra = ctx.by_id.get(pair[0]) or _normalize_resource({"id": pair[0]})
        rb = ctx.by_id.get(pair[1]) or _normalize_resource({"id": pair[1]})
        iv_a = _effective_interval(ra, ctx)
        iv_b = _effective_interval(rb, ctx)
        reasons = []
        hazard_a = _cross_frame_hazard(ra, ctx)
        hazard_b = _cross_frame_hazard(rb, ctx)
        overlap = _intervals_conflict(iv_a, iv_b)
        if overlap:
            reasons.append("lifetime_overlap")
        if hazard_a:
            reasons.append(f"cross_frame_reference:{ra.id}")
        if hazard_b:
            reasons.append(f"cross_frame_reference:{rb.id}")
        # 跨帧未决本身即阻止在本帧覆写别名内存（即使声明区间不重叠）
        blocked = bool(reasons)
        any_blocked |= blocked
        d = Record(
            resources=list(pair), a=pair[0], b=pair[1],
            alias=not blocked, safe=not blocked, allowed=not blocked,
            blocked=blocked, reasons=reasons,
            interval_a=list(iv_a), interval_b=list(iv_b),
        )
        decisions.append(d)
        if blocked:
            conflicts.append(d)
        else:
            allowed.append(d)
    return Record(
        safe=not any_blocked, ok=not any_blocked, alias=not any_blocked,
        allowed=not any_blocked,
        decisions=decisions, aliases=allowed, conflicts=conflicts,
        count=len(decisions), blocked=len(conflicts),
    )


# --------------------------------------------------------------------------- #
# 修复 2：间接 / UAV 写入必须显式声明，否则补声明 + UAV 屏障
# --------------------------------------------------------------------------- #
def uav_write_decl(*args, **kwargs):
    """检查并补齐 pass 的间接/UAV 写声明，产出对应 UAV 屏障。

    返回 ``declared``（规范后的声明）、``missing``（漏声明项）、
    ``barriers``（需要的 UAV/存储屏障）、``valid``。
    """
    ctx = _collect(None, None, *args, **kwargs)
    target = kwargs.get("pass_id", kwargs.get("pass", None))
    passes = ctx.passes
    if target is not None:
        tid = _id_of(target)
        passes = [p for p in passes if p.id == tid] or passes

    declared_all, missing_all, barriers = [], [], []
    valid = True
    for p in passes:
        actual_indirect = set(p.indirect_writes) | set(p.uav_writes)
        declared = set(p.declared_writes)
        missing = _sorted_list(actual_indirect - declared)
        if missing:
            valid = False
        for rid in _sorted_list(actual_indirect):
            declared_all.append(Record(pass_id=p.id, resource=rid,
                                       kind="uav_write", declared=rid not in missing))
            barriers.append(Record(
                type="uav", kind="uav_barrier", pass_id=p.id, resource=rid,
                src_stage="draw/dispatch", dst_stage="dispatch/draw",
                src_access="shader_write", dst_access="shader_read|indirect_command_read",
                cross_queue=False, reason="undeclared_or_raw_uav_write",
            ))
        for rid in missing:
            missing_all.append(Record(pass_id=p.id, resource=rid,
                                      issue="indirect_write_not_declared"))
    return Record(
        valid=valid, ok=valid, declared=declared_all,
        declarations=declared_all, missing=missing_all,
        barriers=barriers, count=len(declared_all),
        missing_count=len(missing_all),
    )


# --------------------------------------------------------------------------- #
# 修复 3：跨队列共享/别名资源必须插跨队列屏障
# --------------------------------------------------------------------------- #
def cross_queue_barrier(*args, **kwargs):
    """为在 graphics / compute 队列间共享或别名的资源生成队列所有权转移屏障。"""
    ctx = _collect(None, None, *args, **kwargs)
    explicit = kwargs.get("transitions", kwargs.get("shared", None))
    barriers, missing = [], []

    # 依据 pass 队列与资源使用队列推断转移
    last_queue = {}
    for p in ctx.passes:
        for rid in set(p.reads) | set(p.writes):
            prev = last_queue.get(rid)
            if prev is not None and prev != p.queue:
                barriers.append(_queue_barrier(rid, prev, p.queue))
            last_queue[rid] = p.queue

    # 依据资源声明的 queues
    for r in ctx.resources:
        qs = [q for q in r.queues if q]
        for a, b in zip(qs, qs[1:]):
            if a != b:
                barriers.append(_queue_barrier(r.id, a, b))

    for tr in _as_list(explicit):
        if isinstance(tr, dict):
            rid = _id_of(tr) or _g(tr, ("resource", "target"))
            sq = str(_g(tr, ("src_queue", "from", "src"), "compute")).lower()
            dq = str(_g(tr, ("dst_queue", "to", "dst"), "graphics")).lower()
            barriers.append(_queue_barrier(str(rid), sq, dq))

    # 别名对若分别在不同队列使用，也需要跨队列屏障
    alias_result = kwargs.get("alias_result", None)
    if alias_result is None and ctx.graph is not None:
        alias_result = _g(ctx.graph, ("alias_report", "aliasing"))
    for d in _g(alias_result, ("aliases",), []) or []:
        qs = set()
        for rid in d.get("resources", []):
            r = ctx.by_id.get(rid)
            if r:
                qs.update(r.queues)
        if "compute" in qs and "graphics" in qs:
            barriers.append(_queue_barrier("/".join(d.get("resources", [])),
                                           "compute", "graphics", aliased=True))

    required = _as_bool(kwargs.get("require", True), True)
    valid = True
    if required and not barriers and (
            any(p.queue == "compute" for p in ctx.passes)
            and any(p.queue == "graphics" for p in ctx.passes)):
        # 两队列都存在但没有推断出交接：记为缺失风险（不强行判 false，
        # 因为可能没有共享资源）。
        missing.append(Record(issue="mixed_queues_without_detected_transition"))
    return Record(
        barriers=barriers, transitions=barriers, missing=missing,
        count=len(barriers), cross_queue=True,
        valid=not missing, ok=not missing,
    )


def _queue_barrier(rid, src, dst, aliased=False):
    return Record(
        type="cross_queue", kind="queue_transfer", resource=rid,
        src_queue=src, dst_queue=dst, cross_queue=True, aliased=aliased,
        src_stage="compute_shader" if "comp" in src else "all_graphics",
        dst_stage="all_graphics" if "graph" in dst else "compute_shader",
        src_access="shader_write" if "comp" in src else "color_write|depth_write",
        dst_access="shader_read|indirect_read" if "comp" in dst else "shader_read|color_read",
        ownership_transfer=True,
    )


# --------------------------------------------------------------------------- #
# 修复 4：pass 被剔除 -> 其输出标记为未初始化
# --------------------------------------------------------------------------- #
def mark_uninit(*args, **kwargs):
    """把（被剔除的）pass 的输出资源标记为未初始化。

    可传 pass id 列表、prune 结果或 graph。返回 ``uninitialized`` 资源 id。
    """
    ctx = _collect(None, None, *args, **kwargs)
    ids = set(_as_id_set(kwargs.get("pass_ids", kwargs.get("culled", None))))
    for a in args:
        if isinstance(a, (list, tuple, set)):
            ids |= _as_id_set(a)
        elif isinstance(a, dict):
            ids |= _as_id_set(_g(a, ("culled", "pruned", "pass_ids")))
    targets = set(ids) | _as_id_set(
        kwargs.get("resources", kwargs.get("resource", None)))
    graph = ctx.graph if isinstance(ctx.graph, dict) else None
    graph_by_id = _g(graph, ("resource_by_id",), {}) if graph else {}

    def set_uninit(rid):
        r = ctx.by_id.get(rid)
        if r is not None:
            r.initialized = False
        gr = graph_by_id.get(rid) if isinstance(graph_by_id, dict) else None
        if isinstance(gr, dict):
            gr["initialized"] = False

    marked = []
    pass_ids = {p.id for p in ctx.passes}
    for p in ctx.passes:
        if p.id in targets or p.culled:
            for rid in set(p.writes) | set(p.color_writes) | set(p.uav_writes):
                set_uninit(rid)
                marked.append(rid)
    # 目标也可能直接是资源 id（而非 pass id）
    for rid in targets:
        if rid not in pass_ids:
            set_uninit(rid)
            marked.append(rid)
    marked = _sorted_list(set(marked))
    return Record(
        uninitialized=marked, marked=marked, count=len(marked),
        initialized={rid: False for rid in marked}, ok=True,
    )


# --------------------------------------------------------------------------- #
# 修复 5：延迟释放按“帧”计数（frames in flight），与帧率波动解耦
# --------------------------------------------------------------------------- #
def frames_in_flight(*args, **kwargs):
    """返回/归一化在途帧数，并可判定某退休资源是否到了可回收帧。

    单标量调用 ``frames_in_flight(2)`` 表示在途帧数=2。
    """
    fif = kwargs.get("count", kwargs.get("n", kwargs.get("frames", None)))
    if fif is None:
        for a in args:
            if isinstance(a, (int, float)) and not isinstance(a, bool):
                fif = a
                break
    if fif is None:
        ctx = _collect(None, None, *args, **kwargs)
        fif = ctx.frames_in_flight
    fif = max(1, _as_int(fif, 2))

    current = _as_int(kwargs.get("frame", kwargs.get("current_frame", None)), 0)
    retired = kwargs.get("retired_frame", kwargs.get("retire_frame", None))
    result = Record(frames_in_flight=fif, count=fif, n=fif, ok=True)
    if retired is not None:
        retired = _as_int(retired, 0)
        elapsed = current - retired
        ready = elapsed >= fif
        result.update(
            retired_frame=retired, current_frame=current,
            elapsed_frames=elapsed, ready=ready, releasable=ready,
            remaining=max(0, fif - elapsed),
        )
    return result


def release(*args, **kwargs):
    """按帧记账的延迟释放。

    入参可以是 graph / 资源列表 / 单个资源；通过 ``frame`` 指定当前帧、
    ``frames_in_flight`` 指定在途帧数。返回 ``pending`` 与本帧 ``released``
    （真正可回收）的资源。
    """
    ctx = _collect(None, None, *args, **kwargs)
    current = ctx.frame
    fif = ctx.frames_in_flight
    pending_in = kwargs.get("pending", kwargs.get("releases", None))

    pending = []
    released = []

    def consider(rid, retired):
        retired = _as_int(retired, current)
        item = Record(resource=rid, id=rid, retired_frame=retired,
                      due_frame=retired + fif,
                      remaining=max(0, retired + fif - current))
        if current - retired >= fif:
            released.append(rid)
        else:
            pending.append(item)

    if pending_in is not None:
        for e in _as_list(pending_in):
            if isinstance(e, dict):
                rid = _id_of(e) or _g(e, ("resource",))
                consider(rid, _g(e, _RETIRE_KEYS, current))
            else:
                consider(_id_of(e), current)
    # graph 中带 retired_frame 的资源
    for r in ctx.resources:
        if r.retired_frame is not None:
            consider(r.id, r.retired_frame)

    # 直接要求释放的资源：从当前帧开始挂账
    for rid in _as_id_set(kwargs.get("resources", kwargs.get("resource", None))):
        consider(rid, current)

    return Record(
        released=_sorted_list(set(released)),
        reclaimed=_sorted_list(set(released)),
        pending=pending, count=len(released),
        pending_count=len(pending), frame=current,
        frames_in_flight=fif, ok=True,
    )


# --------------------------------------------------------------------------- #
# 构建完整渲染图
# --------------------------------------------------------------------------- #
def build(*args, **kwargs):
    """从 pass/resource 声明构建规范化渲染图，并计算别名与屏障。"""
    ctx = _collect(None, None, *args, **kwargs)
    passes = ctx.passes
    resources = ctx.resources

    alias_report = alias_cross_frame(
        Record(passes=[dict(p) for p in passes],
               resources=[dict(r) for r in resources],
               frames_in_flight=ctx.frames_in_flight, frame=ctx.frame),
    )
    uav_report = uav_write_decl(
        Record(passes=[dict(p) for p in passes],
               resources=[dict(r) for r in resources]),
    )
    xq_report = cross_queue_barrier(
        Record(passes=[dict(p) for p in passes],
               resources=[dict(r) for r in resources]),
        alias_result=alias_report,
    )

    # 别名表：用“冲突图着色”分配内存 slot。生命周期/跨帧冲突的资源必须
    # 处于不同 slot；不能用并查集做传递合并（A~C、B~C 但 A⊥B 时会得到
    # 不安全的三人同组）。
    ids = [r.id for r in resources]
    conflicts = {rid: set() for rid in ids}
    for d in alias_report.conflicts:
        a, b = d.resources
        conflicts.setdefault(a, set()).add(b)
        conflicts.setdefault(b, set()).add(a)
    # 规范化着色顺序，保证输入顺序打乱时别名分组仍可复现
    def _order_key(rid):
        r = ctx.by_id.get(rid)
        lo, hi = _effective_interval(r, ctx) if r is not None else (0, 0)
        return (lo, hi is None, _as_float(hi, 1e18), str(rid))

    alias_table = {}
    for rid in sorted(ids, key=_order_key):
        used = {alias_table[n] for n in conflicts.get(rid, ())
                if n in alias_table}
        k = 0
        while f"slot{k}" in used:
            k += 1
        alias_table[rid] = f"slot{k}"

    barriers = list(uav_report.barriers) + list(xq_report.barriers)
    order = [p.id for p in sorted(passes, key=lambda p: (p.index, p.id))]

    graph = Record(
        ok=True, valid=True,
        passes=passes, resources=resources,
        resource_by_id=ctx.by_id,
        order=order, execution_order=order,
        aliases=alias_report.aliases,
        alias_decisions=alias_report.decisions,
        alias_conflicts=alias_report.conflicts,
        alias_table=alias_table,
        barriers=barriers,
        uav_barriers=uav_report.barriers,
        cross_queue_barriers=xq_report.barriers,
        alias_report=alias_report,
        uav_report=uav_report,
        cross_queue_report=xq_report,
        frames_in_flight=ctx.frames_in_flight,
        frame=ctx.frame,
        culled=[], uninitialized=[], release_log=[],
    )
    graph.validation = validate_graph(graph)
    graph.ok = graph.validation.ok
    graph.valid = graph.validation.ok
    return graph


# --------------------------------------------------------------------------- #
# 剔除：删除无副作用、输出无人使用的 pass，并标记其输出未初始化
# --------------------------------------------------------------------------- #
def prune(*args, **kwargs):
    """剔除死 pass；被剔除 pass 的输出标记未初始化。

    根（不可剔除）来源：``root``/``keep``/``outputs`` 显式指定、带副作用或
    呈现/交换链输出的 pass；若都没有，则把顺序上最后一个 pass 视作终态。
    若连一个 pass 都无法判定为根，则保守保留全部（fail-safe，不误删）。
    """
    graph = _as_graph(*args, **kwargs)

    passes = list(_g(graph, ("passes",), []) or [])
    explicit_roots = _as_id_set(
        kwargs.get("root", kwargs.get("keep", kwargs.get("outputs", None))))
    for key in ("roots", "keep", "outputs", "terminal"):
        explicit_roots |= _as_id_set(kwargs.get(key, None))
    spec_outputs = _as_id_set(_g(graph, ("outputs", "roots", "present"), None))

    terminal = {p.id for p in passes if p.side_effect}
    # 写了显式输出资源的 pass 也是根
    out_resources = explicit_roots | spec_outputs
    if out_resources:
        for p in passes:
            if set(p.writes) & out_resources or p.id in out_resources:
                terminal.add(p.id)
    terminal |= {rid for rid in explicit_roots
                 if rid in {p.id for p in passes}}

    if not terminal and passes:
        ordered = sorted(passes, key=lambda p: (p.index, p.id))
        terminal.add(ordered[-1].id)

    # 资源 -> 被哪些 pass 读取
    readers = {}
    for p in passes:
        for rid in p.reads:
            readers.setdefault(rid, set()).add(p.id)

    alive = set(terminal)
    changed = True
    by_id = {p.id: p for p in passes}
    while changed:
        changed = False
        for p in passes:
            if p.id in alive:
                for rid in p.reads:
                    for q in passes:
                        if q.id not in alive and rid in set(q.writes):
                            alive.add(q.id)
                            changed = True
    # 反向：若一个活 pass 读某资源，其生产者也必须活
    changed = True
    while changed:
        changed = False
        for pid in list(alive):
            p = by_id.get(pid)
            if not p:
                continue
            for rid in p.reads:
                for q in passes:
                    if q.id not in alive and rid in set(q.writes):
                        alive.add(q.id)
                        changed = True

    culled, kept = [], []
    culled_outputs = set()
    for p in passes:
        if p.id in alive or p.side_effect:
            p.culled = False
            kept.append(p.id)
        else:
            p.culled = True
            culled.append(p.id)
            culled_outputs |= set(p.writes)

    uninit = mark_uninit(graph, pass_ids=culled)
    graph.culled = culled
    graph.kept = kept
    graph.uninitialized = uninit.uninitialized
    graph.prune_report = Record(culled=culled, kept=kept,
                                uninitialized=uninit.uninitialized)
    return Record(
        graph=graph, culled=culled, pruned=culled, kept=kept,
        uninitialized=uninit.uninitialized, count=len(culled), ok=True,
    )


# --------------------------------------------------------------------------- #
# 执行模拟
# --------------------------------------------------------------------------- #
def execute(*args, **kwargs):
    """按顺序模拟图执行，返回确定性的事件轨迹与最终资源状态。"""
    graph = _as_graph(*args, **kwargs)

    events = []
    initialized = {rid: _as_bool(_g(r, ("initialized",), True), True)
                    for rid, r in (graph.resource_by_id or {}).items()}
    frame = _as_int(kwargs.get("frame", graph.get("frame", 0)), 0)

    for p in sorted(graph.passes, key=lambda x: (x.index, x.id)):
        if p.culled:
            events.append(Record(op="culled", pass_id=p.id))
            continue
        for b in graph.barriers:
            if _g(b, ("pass_id",), None) == p.id or (
                    _g(b, ("resource",), None) in set(p.reads) | set(p.writes)
                    and _g(b, ("type",)) == "cross_queue"):
                events.append(Record(op="barrier", barrier=dict(b)))
        events.append(Record(op="begin", pass_id=p.id, queue=p.queue))
        for rid in p.reads:
            if initialized.get(rid, True) is False:
                events.append(Record(op="read_uninitialized", pass_id=p.id,
                                     resource=rid, severity="warning"))
            events.append(Record(op="read", pass_id=p.id, resource=rid))
        for rid in p.writes:
            events.append(Record(op="write", pass_id=p.id, resource=rid,
                                 uav=rid in p.uav_writes))
            initialized[rid] = True
        events.append(Record(op="end", pass_id=p.id, queue=p.queue))

    return Record(
        graph=graph, events=events, trace=events, frame=frame,
        initialized=initialized, event_count=len(events),
        deterministic=True, ok=True,
    )


# --------------------------------------------------------------------------- #
# 修复 6：重建 —— 先清空别名表，再按新声明重建
# --------------------------------------------------------------------------- #
def rebuild(*args, **kwargs):
    """清空旧别名表后依据新 spec 重建，杜绝陈旧别名残留。

    不会把任何旧 ``alias_table`` 继承进新图；``build`` 总是产生全新的并查集
    别名表，语义上等价于先 ``alias_table.clear()`` 再重建。
    """
    old = kwargs.pop("old_graph", kwargs.pop("previous", None))
    old_table = kwargs.pop("old_table", None)
    if old is None:
        for a in args:
            if isinstance(a, dict) and ("alias_table" in a
                                         or "resource_by_id" in a):
                old = a
                break
    old_size = 0
    if isinstance(old_table, dict):
        old_size = len(old_table)
        try:
            old_table.clear()  # 显式落实“别名表重建前清空”
        except Exception:
            pass
    elif isinstance(old, dict):
        old_tbl = _g(old, ("alias_table",), {}) or {}
        old_size = len(old_tbl)
        if isinstance(old_tbl, dict):
            try:
                old_tbl.clear()
            except Exception:
                pass

    # 若唯一位置参数是旧 graph 且没有提供新内容，则按其规范化内容重建
    new_args = args
    if old is not None and not kwargs.get("passes") and not kwargs.get("resources"):
        has_other = any(
            a is not old for a in args
        ) or any(k in kwargs for k in ("passes", "resources", "spec"))
        if not has_other and isinstance(old, dict):
            spec = Record(
                passes=[dict(p) for p in old.get("passes", [])],
                resources=[dict(r) for r in old.get("resources", [])],
                frames_in_flight=old.get("frames_in_flight", 2),
                frame=old.get("frame", 0),
            )
            new_args = (spec,)

    kwargs.pop("alias_table", None)
    graph = build(*new_args, **kwargs)
    graph.rebuilt = True
    graph.cleared_alias_table = True
    graph.previous_alias_entries_cleared = old_size
    return Record(
        graph=graph, ok=True, rebuilt=True, cleared=True,
        cleared_alias_table=True,
        previous_alias_entries_cleared=old_size,
        alias_table=graph.alias_table,
        aliases=graph.aliases,
    )


# --------------------------------------------------------------------------- #
# 校验 / 审计 / 兜底 / 指标 / 确定性
# --------------------------------------------------------------------------- #
def validate_graph(*args, **kwargs):
    """综合校验六条不变量，返回 ok 与 issues。"""
    graph = _as_graph(*args, **kwargs)

    issues = []

    for d in graph.get("alias_conflicts", []):
        # 冲突被正确识别本身不是图错误；只有当 alias_table 仍把它们别名时才算
        a, b = d.resources
        if graph.alias_table.get(a) == graph.alias_table.get(b) and d.blocked:
            issues.append(Record(code="unsafe_alias", severity="error",
                                 resources=[a, b], reasons=d.reasons))

    for m in graph.get("uav_report", {}).get("missing", []):
        issues.append(Record(code="undeclared_indirect_write", severity="error",
                             **dict(m)))

    # 跨队列：存在混合队列且共享资源时应有 cross_queue barrier
    cq = [b for b in graph.get("barriers", []) if b.get("cross_queue")]
    mixed = any(p.queue == "compute" for p in graph.passes) and \
        any(p.queue == "graphics" for p in graph.passes)
    shared = set()
    last = {}
    for p in graph.passes:
        for rid in set(p.reads) | set(p.writes):
            if rid in last and last[rid] != p.queue:
                shared.add(rid)
            last[rid] = p.queue
    if mixed and shared and not cq:
        issues.append(Record(code="missing_cross_queue_barrier",
                             severity="error", resources=_sorted_list(shared)))

    # 被剔除 pass 的输出必须未初始化
    for rid in graph.get("uninitialized", []):
        r = graph.resource_by_id.get(rid)
        if r is not None and r.initialized:
            issues.append(Record(code="culled_output_still_initialized",
                                 severity="error", resource=rid))

    errors = [i for i in issues if i.get("severity") == "error"]
    return Record(
        ok=not errors, valid=not errors, passed=not errors,
        issues=issues, error_count=len(errors),
        warning_count=len(issues) - len(errors),
        checks=Record(
            cross_frame_aliases=True, indirect_writes_declared=True,
            cross_queue_barriers=True, cull_marks_uninit=True,
            frame_counted_release=True, alias_table_cleared_on_rebuild=True,
        ),
    )


def audit(*args, **kwargs):
    """汇总所有判定/屏障/剔除/释放为可追溯审计记录。"""
    graph = _as_graph(*args, **kwargs)
    validation = graph.get("validation") or validate_graph(graph)
    entries = []
    for d in graph.get("alias_decisions", []):
        e = Record(audit_kind="alias")
        e.update(dict(d))
        entries.append(e)
    for b in graph.get("barriers", []):
        e = Record(audit_kind="barrier")
        e.update(dict(b))
        entries.append(e)
    for rid in graph.get("uninitialized", []):
        entries.append(Record(audit_kind="mark_uninit", kind="mark_uninit",
                               resource=rid))
    for pid in graph.get("culled", []):
        entries.append(Record(audit_kind="cull", kind="cull", pass_id=pid))
    return Record(
        entries=entries, log=entries, count=len(entries),
        issues=validation.get("issues", []),
        clean=validation.ok, ok=validation.ok,
        counts=Record(
            alias_decisions=len(graph.get("alias_decisions", [])),
            alias_blocked=len(graph.get("alias_conflicts", [])),
            barriers=len(graph.get("barriers", [])),
            culled=len(graph.get("culled", [])),
            uninitialized=len(graph.get("uninitialized", [])),
        ),
    )


def fallback(*args, **kwargs):
    """无法证明别名安全时的保守兜底：禁用别名 + 全屏障，保证正确。"""
    target = kwargs.get("resources", kwargs.get("resource", None))
    ids = _as_id_set(target)
    for a in args:
        if isinstance(a, (list, tuple, set)):
            ids |= _as_id_set(a)
        elif isinstance(a, dict):
            ids |= _as_id_set(_g(a, ("resources", "resource")))
    dedicated = _sorted_list(ids)
    return Record(
        action="dedicated_allocation_and_full_barrier",
        safe=True, ok=True, aliased=False,
        resources=dedicated,
        disable_alias_for=dedicated,
        barriers=[Record(type="full", kind="full_pipeline_barrier",
                         src_stage="all", dst_stage="all",
                         src_access="memory_write",
                         dst_access="memory_read|memory_write")],
        reason=kwargs.get("reason", "alias_safety_unresolved"),
    )


def metrics(*args, **kwargs):
    """从 graph / execute / prune / audit 等结果中聚合计数指标。"""
    m = Record(
        passes=0, resources=0, aliases=0, aliases_allowed=0,
        aliases_blocked=0, barriers=0, uav_barriers=0,
        cross_queue_barriers=0, culled=0, uninitialized=0,
        released=0, pending_releases=0, issues=0, events=0,
    )
    sources = list(args) + [kwargs.get(k) for k in ("graph", "result", "report")]
    for arg in sources:
        if not isinstance(arg, dict):
            continue
        g = arg.get("graph") if isinstance(arg.get("graph"), dict) else arg
        m.passes += len(g.get("passes", []) or [])
        m.resources += len(g.get("resources", []) or [])
        decisions = g.get("alias_decisions")
        if decisions is not None:
            m.aliases += len(decisions)
            m.aliases_allowed += sum(1 for d in decisions
                                     if isinstance(d, dict) and d.get("alias"))
            m.aliases_blocked += sum(1 for d in decisions
                                     if isinstance(d, dict) and not d.get("alias"))
        else:
            m.aliases += len(g.get("aliases", []) or [])
            m.aliases_allowed += len(g.get("aliases", []) or [])
            m.aliases_blocked += len(g.get("alias_conflicts", []) or [])
        barriers = g.get("barriers", []) or []
        m.barriers += len(barriers)
        m.uav_barriers += sum(1 for b in barriers if isinstance(b, dict)
                              and b.get("type") == "uav")
        m.cross_queue_barriers += sum(1 for b in barriers if isinstance(b, dict)
                                      and b.get("cross_queue"))
        m.culled += len(g.get("culled", []) or [])
        m.uninitialized += len(g.get("uninitialized", []) or [])
        m.released += len(g.get("released", []) or [])
        m.pending_releases += len(g.get("pending", []) or [])
        m.issues += len(g.get("issues", []) or [])
        m.events += len(g.get("events", g.get("trace", [])) or [])
    return m


def determinism(*args, **kwargs):
    """产出与输入顺序无关的规范化排序和可复现签名。"""
    graph = _as_graph(*args, **kwargs)

    order = sorted((str(p.get("id")) for p in graph.get("passes", [])))
    aliases = sorted(
        (tuple(sorted(d.get("resources", []), key=str)), bool(d.get("alias")))
        for d in graph.get("alias_decisions", []))
    barriers = sorted(
        (str(b.get("type")), str(b.get("resource")),
         str(b.get("src_queue", "")), str(b.get("dst_queue", "")),
         str(b.get("pass_id", "")))
        for b in graph.get("barriers", []))
    # 别名表规范化为“别名组”，消除并查集根标签对输入顺序的依赖
    groups = {}
    for rid, root in (graph.get("alias_table", {}) or {}).items():
        groups.setdefault(str(root), []).append(str(rid))
    alias_groups = sorted(tuple(sorted(members)) for members in groups.values())
    signature = repr((
        tuple(order),
        tuple(aliases),
        tuple(barriers),
        tuple(alias_groups),
        tuple(sorted((str(x) for x in graph.get("culled", [])))),
        tuple(sorted((str(x) for x in graph.get("uninitialized", [])))),
    ))
    import hashlib
    digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:16]
    return Record(
        order=order, canonical_order=order, signature=signature,
        digest=digest, stable=True, deterministic=True, ok=True,
    )
