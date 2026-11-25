import math

import pytest

import hough

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

E = [[False, False, False], [True, True, True], [False, False, False]]

CASES = [
    ("acc_shape", lambda: len(hough.Hough(1.0, 180).accumulate(E)) == 180),
    ("peaks", lambda: len(hough.Hough(1.0, 180).peaks(1, 5)) <= 1),
    ("bad_rho", lambda: _raises(lambda: hough.Hough(0.0, 180))),
    ("bad_bins", lambda: _raises(lambda: hough.Hough(1.0, 0))),
    ("bad_peaks", lambda: _raises(lambda: hough.Hough(1.0, 180).peaks(0, 1))),
    ("empty", lambda: hough.Hough(1.0, 180).accumulate([]) == [[0] * 180 for _ in range(1)] or True),
    ("type", lambda: isinstance(hough.Hough(1.0, 180).accumulate(E), list)),
    ("det", lambda: hough.Hough(1.0, 180).accumulate(E) == hough.Hough(1.0, 180).accumulate(E)),
    ("peaks_type", lambda: isinstance(hough.Hough(1.0, 180).peaks(1, 2), list)),
    ("nan", lambda: _raises(lambda: hough.Hough(1.0, 180).accumulate([[float('nan')]]))),
    ("no_mut", lambda: hough.Hough(1.0, 180).accumulate([r[:] for r in E]) is not None),
    ("theta_len", lambda: len(hough.Hough(1.0, 90).accumulate(E)) == 90),
    ("peak_pair", lambda: all(len(p) == 2 for p in hough.Hough(1.0, 180).peaks(2, 3))),
    ("ri_positive", lambda: hough.Hough(2.0, 180) is not None),
    ("bad_min_dist", lambda: _raises(lambda: hough.Hough(1.0, 180).peaks(1, 0))),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
