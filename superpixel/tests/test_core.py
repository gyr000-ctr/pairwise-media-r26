import math

import pytest

import superpixel

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

IMG = [[0.1, 0.2], [0.3, 0.4]]

CASES = [
    ("labels_shape", lambda: len(superpixel.Slic(2).segment(IMG)) == 2),
    ("label_start", lambda: min(v for row in superpixel.Slic(2).segment(IMG) for v in row) == 0),
    ("bad_k", lambda: _raises(lambda: superpixel.Slic(0))),
    ("bad_iters", lambda: _raises(lambda: superpixel.Slic(2, 0))),
    ("bad_compact", lambda: _raises(lambda: superpixel.Slic(2, 10, 0.0))),
    ("empty", lambda: _raises(lambda: superpixel.Slic(2).segment([]))),
    ("centers_len", lambda: len(superpixel.Slic(2).centers()) <= 2),
    ("type", lambda: isinstance(superpixel.Slic(2).segment(IMG), list)),
    ("det", lambda: superpixel.Slic(2).segment(IMG) == superpixel.Slic(2).segment(IMG)),
    ("label_int", lambda: all(isinstance(v, int) for row in superpixel.Slic(2).segment(IMG) for v in row)),
    ("nan", lambda: _raises(lambda: superpixel.Slic(2).segment([[float('nan')]]))),
    ("no_mut", lambda: superpixel.Slic(2).segment([r[:] for r in IMG]) is not None),
    ("centers_type", lambda: isinstance(superpixel.Slic(2).segment(IMG) and superpixel.Slic(2).centers(), list)),
    ("labels_range", lambda: all(0 <= v < 4 for row in superpixel.Slic(2).segment(IMG) for v in row)),
    ("compact_effect", lambda: superpixel.Slic(2, 10, 1.0).segment(IMG) != superpixel.Slic(2, 10, 20.0).segment(IMG)),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
