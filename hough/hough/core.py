"""hough 核心 API（初始未实现）。"""


class Hough:
    def __init__(self, rho_res, theta_bins):
        """构造"""
        raise NotImplementedError

    def accumulate(self, edges):
        """投票"""
        raise NotImplementedError

    def peaks(self, n, min_dist):
        """峰值"""
        raise NotImplementedError
