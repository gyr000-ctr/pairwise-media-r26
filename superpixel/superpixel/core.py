"""superpixel 核心 API（初始未实现）。"""


class Slic:
    def __init__(self, k, iters=10, compactness=10.0):
        """构造"""
        raise NotImplementedError

    def segment(self, image):
        """分割"""
        raise NotImplementedError

    def centers(self):
        """簇中心"""
        raise NotImplementedError
