"""template 核心 API（初始未实现）。"""


class Ncc:
    def match(self, image, tmpl):
        """匹配图"""
        raise NotImplementedError

    def score(self, a, b):
        """相似度"""
        raise NotImplementedError
