import math

import pytest

import template

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

IMG = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
T = [[0.4, 0.5], [0.7, 0.8]]

CASES = [
    ("self_one", lambda: abs(template.Ncc().score(T, T) - 1.0) < 1e-9),
    ("range", lambda: -1.0 <= template.Ncc().score(T, [[0.1, 0.1], [0.1, 0.1]]) <= 1.0),
    ("shape", lambda: len(template.Ncc().match(IMG, T)) == 2),
    ("bad_size", lambda: _raises(lambda: template.Ncc().match(IMG, [[0.1] * 5]))),
    ("flat_template", lambda: _raises(lambda: template.Ncc().score(T, [[0.5, 0.5], [0.5, 0.5]]))),
    ("empty", lambda: _raises(lambda: template.Ncc().match([], T))),
    ("type", lambda: isinstance(template.Ncc().match(IMG, T), list)),
    ("det", lambda: template.Ncc().match(IMG, T) == template.Ncc().match(IMG, T)),
    ("score_type", lambda: isinstance(template.Ncc().score(T, T), float)),
    ("nan", lambda: _raises(lambda: template.Ncc().score(T, [[float('nan'), 0.0], [0.0, 0.0]]))),
    ("no_mut", lambda: template.Ncc().match([r[:] for r in IMG], [r[:] for r in T]) is not None),
    ("antisym", lambda: abs(template.Ncc().score(T, [[0.8, 0.7], [0.5, 0.4]]) + template.Ncc().score([[0.8, 0.7], [0.5, 0.4]], T)) < 1e-9),
    ("match_range", lambda: all(-1.0 <= v <= 1.0 for row in template.Ncc().match(IMG, T) for v in row)),
    ("float", lambda: isinstance(template.Ncc().match(IMG, T)[0][0], float)),
    ("one_by_one", lambda: isinstance(template.Ncc().match(IMG, [[0.5]]), list)),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
