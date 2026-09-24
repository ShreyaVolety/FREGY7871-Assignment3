import numpy as np
from src.estimators import estimate_pair


def test_w1_closed_form():
    xh = np.array([1., -2., 3., -1.])
    xl = np.array([.2, -.1, .3, -.2])
    yh = 2*xh + np.array([.1, .2, -.1, .1])
    yl = 2*xl + np.array([.02, -.01, .01, 0.])
    result = estimate_pair(xh, xl, yh, yl)
    expected = ((xh@yh)/4 - (xl@yl)/4) / ((xh@xh)/4 - (xl@xl)/4)
    assert np.isclose(result["w1_beta"], expected)


def test_equal_length_required():
    try:
        estimate_pair([1, 2], [1], [2, 4], [2])
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")
