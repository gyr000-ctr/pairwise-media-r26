import math

import pytest

import poissonblend

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

D = [[0.5] * 4 for _ in range(4)]
S = [[0.2] * 2 for _ in range(2)]
M = [[True, True], [True, True]]

CASES = [
    ("shape", lambda: len(poissonblend.PoissonBlend().blend(D, S, M, (1, 1))) == 4),
    ("outside", lambda: abs(poissonblend.PoissonBlend().blend(D, S, M, (1, 1))[0][0] - 0.5) < 1e-9),
    ("bad_offset", lambda: _raises(lambda: poissonblend.PoissonBlend().blend(D, S, M, (9, 9)))),
    ("bad_mask", lambda: _raises(lambda: poissonblend.PoissonBlend().blend(D, S, [[True]], (1, 1)))),
    ("empty", lambda: _raises(lambda: poissonblend.PoissonBlend().blend([], S, M, (0, 0)))),
    ("type", lambda: isinstance(poissonblend.PoissonBlend().blend(D, S, M, (1, 1)), list)),
    ("det", lambda: poissonblend.PoissonBlend().blend(D, S, M, (1, 1)) == poissonblend.PoissonBlend().blend(D, S, M, (1, 1))),
    ("range", lambda: all(-1.0 <= v <= 2.0 for row in poissonblend.PoissonBlend().blend(D, S, M, (1, 1)) for v in row)),
    ("nan", lambda: _raises(lambda: poissonblend.PoissonBlend().blend([[float('nan')] * 4 for _ in range(4)], S, M, (1, 1)))),
    ("no_mut", lambda: poissonblend.PoissonBlend().blend([r[:] for r in D], [r[:] for r in S], [m[:] for m in M], (1, 1)) is not None),
    ("float", lambda: isinstance(poissonblend.PoissonBlend().blend(D, S, M, (1, 1))[0][0], float)),
    ("full_mask", lambda: isinstance(poissonblend.PoissonBlend().blend([[0.5]], [[0.1]], [[True]], (0, 0)), list)),
    ("size_mismatch", lambda: _raises(lambda: poissonblend.PoissonBlend().blend(D, S, [[True, True, True]] * 2, (0, 0)))),
    ("zero_offset", lambda: isinstance(poissonblend.PoissonBlend().blend(D, S, M, (0, 0)), list)),
    ("row_len", lambda: len(poissonblend.PoissonBlend().blend(D, S, M, (1, 1))[0]) == 4),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
