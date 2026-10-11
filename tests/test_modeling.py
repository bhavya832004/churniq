import numpy as np
from churniq.modeling import choose_threshold

def test_threshold_range():
    t = choose_threshold(np.array([0,0,1,1]), np.array([0.1,0.4,0.6,0.9]))
    assert 0 <= t <= 1
