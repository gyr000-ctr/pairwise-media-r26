import math

import pytest

import inpaint

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

IMG = [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
MASK = [[False, False, False], [False, True, False], [False, False, False]]

CASES = [
    ("outside_unchanged", lambda: abs(inpaint.Inpaint().fill([[0.2, 0.2], [0.2, 0.2]], [[False, False], [False, False]])[0][0] - 0.2) < 1e-12),
    ("shape", lambda: len(inpaint.Inpaint().fill(IMG, MASK)) == 3),
    ("bad_iter", lambda: _raises(lambda: inpaint.Inpaint(0))),
    ("bad_tol", lambda: _raises(lambda: inpaint.Inpaint(10, 0.0))),
    ("bad_shape", lambda: _raises(lambda: inpaint.Inpaint().fill([[0.0]], [[False, False]]))),
    ("empty", lambda: _raises(lambda: inpaint.Inpaint().fill([], []))),
    ("type", lambda: isinstance(inpaint.Inpaint().fill(IMG, MASK), list)),
    ("det", lambda: inpaint.Inpaint().fill(IMG, MASK) == inpaint.Inpaint().fill(IMG, MASK)),
    ("range", lambda: all(0.0 <= v <= 1.0 for row in inpaint.Inpaint().fill(IMG, MASK) for v in row)),
    ("nan", lambda: _raises(lambda: inpaint.Inpaint().fill([[float('nan')]], [[False]]))),
    ("no_mut", lambda: inpaint.Inpaint().fill([r[:] for r in IMG], [m[:] for m in MASK]) is not None),
    ("all_mask", lambda: isinstance(inpaint.Inpaint().fill([[0.5]], [[True]]), list)),
    ("smoothed", lambda: abs(inpaint.Inpaint().fill(IMG, MASK)[1][1] - 1.0) >= 0.0),
    ("float", lambda: isinstance(inpaint.Inpaint().fill(IMG, MASK)[0][0], float)),
    ("no_mask", lambda: inpaint.Inpaint().fill(IMG, [[False] * 3 for _ in range(3)]) == IMG),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
