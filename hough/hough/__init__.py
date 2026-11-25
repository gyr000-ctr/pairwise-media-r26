"""hough：极坐标霍夫直线检测（亚像素峰值 + 共线峰值合并线段）。

这里把 :mod:`hough.core` 的全部公共名字（以及常用的下划线内部辅助，例如
``_line_diff``）重新导出到包命名空间，因此 ``import hough`` 后无论是
``hough.Hough``、``hough.fit_line`` 还是各类异常都可直接访问。
"""

from .core import *  # noqa: F401,F403
from . import core  # noqa: F401

# 把 core 中定义/导入的名字（含单下划线辅助）绑定为 hough.<name>，
# 最大化对不同调用/导入习惯的兼容性；跳过双下划线元信息以免覆盖包属性。
for _name in dir(core):
    if not _name.startswith("__"):
        globals()[_name] = getattr(core, _name)

# 对外 star-export 所有非下划线公共名。
__all__ = [_n for _n in dir(core) if not _n.startswith("_")]

del _name  # noqa: F821
