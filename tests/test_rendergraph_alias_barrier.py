import pytest
import graphics.rendergraph_alias_barrier as mod

CASES = [
    ("alias_cross_frame", (1,)),
    ("uav_write_decl", (1,)),
    ("cross_queue_barrier", (1,)),
    ("mark_uninit", (1,)),
    ("frames_in_flight", (1,)),
    ("build", (1,)),
    ("execute", (1,)),
    ("prune", (1,)),
    ("release", (1,)),
    ("rebuild", (1,)),
    ("validate_graph", (1,)),
    ("metrics", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("determinism", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
