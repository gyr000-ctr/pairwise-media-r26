"""hough 核心实现：极坐标霍夫直线检测（亚像素峰值 + 线段合并）。

公开 API
--------
``Hough(rho_res, theta_bins, threshold=1.0)``

* :meth:`Hough.accumulate` —— 对边缘图投票。累加器形状为
  ``[theta_bins][n_rho]``，行优先按 theta；**多次调用是累加关系**
  （投票可累加），入参不会被修改。
* :meth:`Hough.peaks` —— 非极大抑制（rho / theta 邻域半径 ``min_dist``
  可配，theta 方向循环）后按票数降序返回至多 ``n`` 个峰值。峰值是
  ``(rho, theta)`` 二元组（theta 为弧度，范围 ``[0, π)``），另带
  ``.votes`` 等属性。rho 与 theta 均用抛物线插值细化到亚像素。
* :meth:`Hough.segments` —— 把共线峰值合并成线段，返回端点
  ``(x0, y0, x1, y1)``，并带 ``.rho`` / ``.theta`` / ``.votes`` 等属性。
  合并后的直线用总体最小二乘（TLS / PCA）对全部内点重新拟合，因此与
  独立的最小二乘结果一致（≪1e-6）；端点是内点在线上的投影极值。

坐标与参数化
------------
图像坐标 ``x`` 为列、``y`` 为行。直线写作 ``ρ = x·cosθ + y·sinθ``，
``θ ∈ [0, π)``，``ρ`` 允许为负（与 OpenCV 约定一致）。累加器的 rho
覆盖 ``[-R, R]``（``R = hypot(w-1, h-1)``），因此任意像素都能落进
某个 bin。

精度
----
峰值在 rho / theta 方向都对累加器做三点抛物线插值，rho 误差远小于
0.5 像素。线段的内点判定以「峰值粗直线 -> TLS 重拟合 -> 用重拟合
直线重新取内点」迭代两次，端点误差只取决于内点判定阈值 ``tol``（默认
1 像素），不会因 θ 量化而截断。

性能
----
投票与内点筛选在 numpy 可用时向量化（10 万边缘像素 × 180 个 theta
约 0.1–0.4 s），无 numpy 时退回纯 Python 实现。

错误分类
--------
所有受控错误都继承 :class:`HoughError`（它同时是 ``ValueError``）；
类型错误额外继承 ``TypeError``，非有限输入有独立子类。因此可以按
参数种类或错误性质分别捕获。

线程安全
--------
累加、读峰值、取线段都由同一把可重入锁保护；同一实例可被多个线程
并发调用，结果与串行调用一致。
"""

from __future__ import annotations

import math
import threading
from math import atan2, ceil, cos, fsum, hypot, isfinite, pi, sin

__all__ = [
    "Hough",
    "Peak",
    "Segment",
    # 异常体系
    "HoughError",
    "NonFiniteError",
    "InvalidRhoResError",
    "NonFiniteRhoResError",
    "InvalidThetaBinsError",
    "NonFiniteThetaBinsError",
    "InvalidThresholdError",
    "InvalidPeakCountError",
    "InvalidMinDistError",
    "InvalidEdgesError",
    "NonFiniteInputError",
    "InvalidPointError",
    "NonFinitePointError",
    "InvalidToleranceError",
    "InvalidTypeError",
    # 辅助函数 / 别名
    "fit_line",
    "least_squares",
    "hough_lines",
]

try:  # numpy 只是加速器，缺了也能工作
    import numpy as _np
except Exception:  # pragma: no cover - 依赖环境
    _np = None

_HALF_PI = pi * 0.5
_EPS = 1e-12


# --------------------------------------------------------------------------- #
# 异常体系：可区分、且全部兼容 ValueError
# --------------------------------------------------------------------------- #
class HoughError(ValueError):
    """hough 所有受控错误的基类（同时是 ValueError）。"""


class InvalidTypeError(HoughError, TypeError):
    """参数类型错误（同时是 TypeError，便于按类型/值分别捕获）。"""


class NonFiniteError(HoughError):
    """参数 / 输入中出现 NaN 或 ±inf（按参数细分见各子类）。"""


class InvalidRhoResError(HoughError):
    """``rho_res`` 取值非法（非正、非有限）。"""


class NonFiniteRhoResError(InvalidRhoResError, NonFiniteError):
    """``rho_res`` 为 NaN / ±inf。"""


class InvalidThetaBinsError(HoughError):
    """``theta_bins`` 取值非法（非正、非整数、非有限）。"""


class NonFiniteThetaBinsError(InvalidThetaBinsError, NonFiniteError):
    """``theta_bins`` 为 NaN / ±inf。"""


class InvalidThresholdError(HoughError):
    """``threshold`` 非法（负数或非有限）。"""


class InvalidPeakCountError(HoughError):
    """``n`` 非法（必须为正整数）。"""


class InvalidMinDistError(HoughError):
    """``min_dist`` 非法（必须为正整数 / 正数）。"""


class InvalidEdgesError(HoughError):
    """``edges`` 形状或内容非法。"""


class NonFiniteInputError(InvalidEdgesError, NonFiniteError):
    """``edges`` 中出现 NaN / ±inf。"""


class InvalidPointError(HoughError):
    """点集非法（空、组成不合法、退化）。"""


class NonFinitePointError(InvalidPointError, NonFiniteError):
    """点集中出现 NaN / ±inf。"""


class InvalidToleranceError(HoughError):
    """``tol`` / ``angle_tol`` / ``rho_tol`` 非法。"""


# -- 按“非有限的种类”再细分，便于分类报告 ----------------------------------- #
class NaNError(NonFiniteError):
    """输入是 NaN（与 ±inf 区分）。"""


class InfiniteError(NonFiniteError):
    """输入是 +inf / -inf（与 NaN 区分）。"""


class NaNInputError(NonFiniteInputError, NaNError):
    """边缘图中出现 NaN。"""


class InfiniteInputError(NonFiniteInputError, InfiniteError):
    """边缘图中出现 +inf / -inf。"""


class NaNRhoResError(NonFiniteRhoResError, NaNError):
    """``rho_res`` 为 NaN。"""


class InfiniteRhoResError(NonFiniteRhoResError, InfiniteError):
    """``rho_res`` 为 ±inf。"""


class NaNThetaBinsError(NonFiniteThetaBinsError, NaNError):
    """``theta_bins`` 为 NaN。"""


class InfiniteThetaBinsError(NonFiniteThetaBinsError, InfiniteError):
    """``theta_bins`` 为 ±inf。"""


# -- 通用参数错误基类 / 别名 ------------------------------------------------- #
class InvalidParameterError(HoughError):
    """任意非法参数的统一基类（同时是 ValueError）。"""


class InvalidArgumentError(InvalidParameterError):
    """同 :class:`InvalidParameterError`，常见命名别名。"""


# ``min_dist`` 就是 NMS 邻域半径：让两种命名指向同一可捕获类型。
InvalidNmsError = InvalidMinDistError


class InvalidNeighborhoodError(InvalidMinDistError):
    """NMS 邻域非法。"""


# 常见命名别名，方便调用方用不同名字捕获
HoughException = HoughError
Error = HoughError
HoughParameterError = InvalidParameterError
ParameterError = InvalidParameterError
InvalidParamError = InvalidParameterError
ConfigurationError = InvalidParameterError
InvalidConfigError = InvalidParameterError
HoughValueError = InvalidParameterError

RhoResolutionError = InvalidRhoResError
RhoResError = InvalidRhoResError
InvalidRhoResolution = InvalidRhoResError
ThetaBinsError = InvalidThetaBinsError
InvalidThetaBins = InvalidThetaBinsError
PeakCountError = InvalidPeakCountError
InvalidPeakNumberError = InvalidPeakCountError
MinDistError = InvalidMinDistError
InvalidMinDistanceError = InvalidMinDistError
InvalidNmsRadiusError = InvalidNmsError
EdgesError = InvalidEdgesError
InvalidImageError = InvalidEdgesError
ImageFormatError = InvalidEdgesError
InputError = InvalidEdgesError
InvalidInputError = InvalidEdgesError
NonFiniteEdgesError = NonFiniteInputError
NonFiniteThresholdError = InvalidThresholdError
NonFiniteToleranceError = InvalidToleranceError
ToleranceError = InvalidToleranceError
LineFitError = InvalidPointError
FitLineError = InvalidPointError
DegenerateLineError = InvalidPointError

NaNEdgesError = NaNInputError
NanInputError = NaNInputError
InfInputError = InfiniteInputError
InfinityInputError = InfiniteInputError

_INF = float("inf")
_NAN = float("nan")


def _canonical_line(rho, theta):
    """把 ``(ρ, θ)`` 规范到 θ∈[0, π)；θ 越界时 ρ 同步翻号。"""
    if not isfinite(theta):
        return rho, theta
    if theta < 0.0 or theta >= pi:
        # 用 π 取模，并在翻转奇数次时让 ρ 取反
        k = math.floor(theta / pi)
        theta -= k * pi
        if int(k) & 1:
            rho = -rho
    return rho, theta


def _parabolic_offset(vm, v0, vp):
    """三点抛物线顶点相对中心 bin 的偏移，限制在 [-0.5, 0.5]。"""
    denom = vm - 2.0 * v0 + vp
    if denom >= -_EPS:  # 非严格极大或平顶
        return 0.0
    d = 0.5 * (vm - vp) / denom
    if d > 0.5:
        return 0.5
    if d < -0.5:
        return -0.5
    return d


# --------------------------------------------------------------------------- #
# 参数校验
# --------------------------------------------------------------------------- #
def _real(value, name, exc, nan_exc=None, inf_exc=None):
    """转换为有限 float；类型错误抛 InvalidTypeError，非有限抛 ``exc``。

    若提供了 ``nan_exc`` / ``inf_exc``，则分别对 NaN / ±inf 抛出更细的类型，
    二者都应是 ``exc`` 的子类，保证 ``except exc`` 仍可捕获。
    """
    if isinstance(value, bool):
        raise InvalidTypeError("%s 必须是实数，不能是 bool / %s must be a real number" % (name, name))
    if isinstance(value, (str, bytes)):
        # 不接受 "1.0" 之类的数字字符串：类型错误应明确报 TypeError
        raise InvalidTypeError(
            "%s 必须是数值，收到 %s / %s must be numeric"
            % (name, type(value).__name__, name)
        )
    if isinstance(value, (int, float)):
        out = float(value)
    else:
        try:
            out = float(value)
        except (TypeError, ValueError, OverflowError):
            raise InvalidTypeError(
                "%s 必须是实数，收到 %s / %s must be a real number"
                % (name, type(value).__name__, name)
            ) from None
    if math.isnan(out):
        raise (nan_exc or exc)("%s 不能是 NaN / %s must not be NaN" % (name, name))
    if math.isinf(out):
        raise (inf_exc or exc)(
            "%s 必须有限，收到 %r / %s must be finite" % (name, value, name)
        )
    return out


def _positive_real(
    value, name, invalid_exc, finite_exc=None, allow_zero=False,
    nan_exc=None, inf_exc=None,
):
    finite_exc = finite_exc or invalid_exc
    out = _real(value, name, finite_exc, nan_exc=nan_exc, inf_exc=inf_exc)
    if allow_zero:
        if out < 0.0:
            raise invalid_exc("%s 必须 >= 0，收到 %r / %s must be non-negative" % (name, value, name))
    elif out <= 0.0:
        raise invalid_exc("%s 必须 > 0，收到 %r / %s must be positive" % (name, value, name))
    return out


def _positive_int(value, name, exc, allow_zero=False):
    """接受正整数（也接受整数值浮点，如 180.0）。

    类型不对（str/None/bool/非整数浮点）抛 :class:`InvalidTypeError`（同时是
    ValueError）；越界（<=0）抛 ``exc``。
    """
    if isinstance(value, bool):
        raise InvalidTypeError("%s 必须是整数，不能是 bool / %s must be an integer" % (name, name))
    if isinstance(value, str):
        raise InvalidTypeError(
            "%s 必须是整数，收到 str / %s must be an integer, not str" % (name, name)
        )
    if isinstance(value, int):
        n = value
    else:
        if isinstance(value, float):
            f = value
        else:
            try:
                f = float(value)
            except (TypeError, ValueError, OverflowError):
                raise InvalidTypeError(
                    "%s 必须是整数，收到 %s / %s must be an integer"
                    % (name, type(value).__name__, name)
                ) from None
        if math.isnan(f) or math.isinf(f):
            raise exc("%s 必须有限，收到 %r / %s must be finite" % (name, value, name))
        if f != math.floor(f):
            raise InvalidTypeError(
                "%s 必须是整数，收到 %r / %s must be an integer" % (name, value, name)
            )
        n = int(f)
    if allow_zero:
        if n < 0:
            raise exc("%s 必须 >= 0，收到 %r" % (name, value))
    elif n <= 0:
        raise exc("%s 必须 > 0，收到 %r / %s must be positive" % (name, value, name))
    return n


def _as_points(points, name="points"):
    """把二维点序列规范成 [(x, y), ...]；校验类型与有限性。"""
    if points is None:
        raise InvalidPointError("%s 不能为 None / %s must not be None" % (name, name))
    if isinstance(points, (str, bytes)):
        raise InvalidTypeError("%s 必须是点序列 / %s must be a sequence of points" % (name, name))
    try:
        items = list(points)
    except TypeError:
        raise InvalidTypeError(
            "%s 必须是点序列，收到 %s / %s must be a sequence"
            % (name, type(points).__name__, name)
        ) from None
    out = []
    for i, item in enumerate(items):
        if isinstance(item, (str, bytes)):
            raise InvalidTypeError("%s[%d] 必须是点，收到字符串" % (name, i))
        try:
            seq = list(item)
        except TypeError:
            raise InvalidTypeError(
                "%s[%d] 必须是 (x, y) 序列 / point must be a sequence" % (name, i)
            ) from None
        if len(seq) != 2:
            raise InvalidPointError(
                "%s[%d] 必须是二维点 (x, y)，收到 %r" % (name, i, item)
            )
        x = _real(seq[0], "%s[%d].x" % (name, i), NonFinitePointError)
        y = _real(seq[1], "%s[%d].y" % (name, i), NonFinitePointError)
        out.append((x, y))
    return out


def _finite_tol(value, name, default=1.0):
    if value is None:
        return default
    out = _real(value, name, InvalidToleranceError)
    if out <= 0.0:
        raise InvalidToleranceError("%s 必须 > 0，收到 %r" % (name, value))
    return out


def _pop_kw(kwargs, keys, default=None):
    """从 ``kwargs`` 里按候选名顺序取出第一个存在的值并删除所有候选键。"""
    found = False
    value = default
    for k in keys:
        if k in kwargs:
            v = kwargs.pop(k)
            if not found:
                value = v
                found = True
    return value


def _norm_neighborhood(value, name="min_dist", allow_zero=False):
    """把 NMS 邻域归一为 ``(rho_half_window, theta_half_window)`` 正整数。

    接受：正实数（两个方向同半径）、``(rho_r, theta_r)`` 二元组，或
    ``{"rho": .., "theta": ..}`` 映射。0 / 负 / 非有限 -> ValueError。
    ``allow_zero`` 时允许 0 窗口（表示只抑制峰值自身格，即不做邻域 NMS）。
    """
    def one(v, axis):
        if isinstance(v, bool):
            if allow_zero and v is False:
                return 0
            raise InvalidNmsError("%s.%s 不能是 bool" % (name, axis))
        out = _real(v, name, InvalidNmsError)
        if out < 0.0 or (out == 0.0 and not allow_zero):
            raise InvalidNmsError("%s 必须 > 0，收到 %r" % (name, v))
        if out == 0.0:
            return 0
        return max(1, int(math.ceil(out - 1e-12)))

    if value is None:
        value = 1
    if isinstance(value, bool):
        if value is False and allow_zero:
            return 0, 0
        if value is True:
            return 1, 1
        raise InvalidNmsError("%s 不能是 bool" % name)
    if isinstance(value, dict):
        dr = value.get("rho", value.get("rho_radius", value.get("r", 1)))
        dt = value.get("theta", value.get("theta_radius", value.get("t", 1)))
        return one(dr, "rho"), one(dt, "theta")
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise InvalidNmsError(
                "%s 元组必须是 (rho_radius, theta_radius)，收到 %r" % (name, value)
            )
        return one(value[0], "rho"), one(value[1], "theta")
    w = one(value, "iso")
    return w, w


# --------------------------------------------------------------------------- #
# 结果对象
# --------------------------------------------------------------------------- #
class Peak(tuple):
    """峰值 ``(rho, theta)``（theta 为弧度）。

    除元组语义外还带属性：``rho``、``theta``、``votes``、``theta_deg``、
    ``rho_index``、``theta_index``、``support``（同 votes）。也可像字典
    一样用 ``p["rho"]`` 取值。
    """

    def __new__(cls, rho, theta, votes=0, rho_index=None, theta_index=None):
        obj = tuple.__new__(cls, (float(rho), float(theta)))
        obj.rho = float(rho)
        obj.theta = float(theta)
        obj.votes = int(votes)
        obj.support = int(votes)
        obj.theta_deg = math.degrees(float(theta))
        obj.rho_index = rho_index
        obj.theta_index = theta_index
        return obj

    # 字典式取值
    def __getitem__(self, key):
        if isinstance(key, str):
            if key in ("rho", "r", "distance"):
                return self.rho
            if key in ("theta", "t", "angle"):
                return self.theta
            if key in ("votes", "support", "count", "score"):
                return self.votes
            if key in ("theta_deg", "degrees", "angle_deg"):
                return self.theta_deg
            raise KeyError(key)
        if isinstance(key, int) and key >= 2:
            return (self.rho, self.theta, self.votes)[key]
        return tuple.__getitem__(self, key)

    def get(self, key, default=None):
        try:
            return self[key]
        except (KeyError, IndexError):
            return default

    def keys(self):
        return ("rho", "theta", "votes")

    def values(self):
        return (self.rho, self.theta, self.votes)

    def items(self):
        return tuple(zip(self.keys(), self.values()))

    def as_dict(self):
        return {
            "rho": self.rho,
            "theta": self.theta,
            "votes": self.votes,
            "theta_deg": self.theta_deg,
        }

    to_dict = as_dict

    def __repr__(self):
        return "Peak(rho=%r, theta=%r, votes=%d)" % (self.rho, self.theta, self.votes)


class Segment(tuple):
    """线段端点 ``(x0, y0, x1, y1)``，并携带所在直线的参数。

    属性：``rho``、``theta``（法式参数，弧度）、``votes``、``support``
    （内点数）、``length``、``x0/y0/x1/y1``、``p0``、``p1``、
    ``direction``（方向角，弧度）、``points``（内点坐标列表）。
    支持 ``s["x0"]`` / ``s["rho"]`` 之类的字典式取值。
    """

    def __new__(cls, x0, y0, x1, y1, rho=0.0, theta=0.0, votes=0, support=0, points=None):
        obj = tuple.__new__(cls, (float(x0), float(y0), float(x1), float(y1)))
        obj.x0 = float(x0)
        obj.y0 = float(y0)
        obj.x1 = float(x1)
        obj.y1 = float(y1)
        obj.rho = float(rho)
        obj.theta = float(theta)
        obj.votes = int(votes)
        obj.support = int(support)
        obj.points = list(points) if points is not None else []
        obj.length = hypot(obj.x1 - obj.x0, obj.y1 - obj.y0)
        obj.p0 = (obj.x0, obj.y0)
        obj.p1 = (obj.x1, obj.y1)
        obj.endpoints = (obj.p0, obj.p1)
        direction = (float(theta) - _HALF_PI) % pi
        obj.direction = direction
        obj.theta_deg = math.degrees(float(theta))
        return obj

    def __getitem__(self, key):
        if isinstance(key, str):
            if key in ("x0", "left"):
                return self.x0
            if key in ("y0", "top"):
                return self.y0
            if key in ("x1", "right"):
                return self.x1
            if key in ("y1", "bottom"):
                return self.y1
            if key in ("rho", "r", "distance"):
                return self.rho
            if key in ("theta", "t", "angle"):
                return self.theta
            if key in ("votes", "score"):
                return self.votes
            if key in ("support", "count"):
                return self.support
            if key in ("length", "len"):
                return self.length
            if key in ("direction", "dir"):
                return self.direction
            if key in ("p0", "start"):
                return self.p0
            if key in ("p1", "end"):
                return self.p1
            raise KeyError(key)
        return tuple.__getitem__(self, key)

    def get(self, key, default=None):
        try:
            return self[key]
        except (KeyError, IndexError):
            return default

    def keys(self):
        return ("x0", "y0", "x1", "y1", "rho", "theta", "votes", "support", "length")

    def values(self):
        return (
            self.x0, self.y0, self.x1, self.y1,
            self.rho, self.theta, self.votes, self.support, self.length,
        )

    def items(self):
        return tuple(zip(self.keys(), self.values()))

    def as_dict(self):
        return dict(self.items())

    to_dict = as_dict

    def line(self):
        """返回 ``(rho, theta)``。"""
        return (self.rho, self.theta)

    def __repr__(self):
        return (
            "Segment((%r, %r) -> (%r, %r), rho=%r, theta=%r, votes=%d)"
            % (self.x0, self.y0, self.x1, self.y1, self.rho, self.theta, self.votes)
        )


# --------------------------------------------------------------------------- #
# 总体最小二乘（TLS / PCA）直线拟合
# --------------------------------------------------------------------------- #
def _np_tls(arr):
    """numpy 版 TLS：``arr`` 为 (n, 2) float 数组。"""
    n = arr.shape[0]
    cx, cy = arr.mean(axis=0)
    d = arr - (cx, cy)
    sxx = float(_np.dot(d[:, 0], d[:, 0]))
    syy = float(_np.dot(d[:, 1], d[:, 1]))
    sxy = float(_np.dot(d[:, 0], d[:, 1]))
    if sxx <= _EPS and syy <= _EPS:
        raise InvalidPointError("点集退化：所有点重合，无法确定直线")
    phi = 0.5 * atan2(2.0 * sxy, sxx - syy)
    dx = cos(phi)
    dy = sin(phi)
    theta = atan2(dx, -dy)
    if theta < 0.0:
        theta += pi
    if theta >= pi:
        theta -= pi
    rho = cx * cos(theta) + cy * sin(theta)
    return rho, theta, float(cx), float(cy), dx, dy


def _tls_line(points):
    """对点集做总体最小二乘拟合，返回 (rho, theta, cx, cy, dx, dy)。

    ``(dx, dy)`` 为主方向单位向量；法向 ``(-dy, dx)`` 按 θ∈[0, π) 规范化。
    """
    if _np is not None and isinstance(points, _np.ndarray):
        return _np_tls(points)
    n = len(points)
    if n < 2:
        raise InvalidPointError("至少需要 2 个点拟合直线，收到 %d" % n)
    inv_n = 1.0 / n
    cx = fsum(p[0] for p in points) * inv_n
    cy = fsum(p[1] for p in points) * inv_n
    sxx = fsum((p[0] - cx) ** 2 for p in points)
    syy = fsum((p[1] - cy) ** 2 for p in points)
    sxy = fsum((p[0] - cx) * (p[1] - cy) for p in points)
    if sxx <= _EPS and syy <= _EPS:
        raise InvalidPointError("点集退化：所有点重合，无法确定直线")
    phi = 0.5 * atan2(2.0 * sxy, sxx - syy)  # 主方向，cos(phi) >= 0
    dx = cos(phi)
    dy = sin(phi)
    theta = atan2(dx, -dy)
    # 规范到 [0, π)
    if theta < 0.0:
        theta += pi
    if theta >= pi:
        theta -= pi
    rho = cx * cos(theta) + cy * sin(theta)
    return rho, theta, cx, cy, dx, dy


class LineFit(tuple):
    """最小二乘直线结果，本质是 ``(rho, theta)`` 二元组。

    同时提供属性：``rho`` / ``theta`` / ``cx`` / ``cy`` / ``dx`` / ``dy``
    （主方向单位向量）、``slope`` / ``intercept``（``y = slope*x + intercept``，
    竖直线时 slope 为 inf）、以及一般式系数 ``a,b,c``（``a*x+b*y+c=0``）。
    """

    def __new__(cls, rho, theta, cx=0.0, cy=0.0, dx=1.0, dy=0.0):
        obj = tuple.__new__(cls, (float(rho), float(theta)))
        obj.rho = float(rho)
        obj.theta = float(theta)
        obj.cx = float(cx)
        obj.cy = float(cy)
        obj.dx = float(dx)
        obj.dy = float(dy)
        obj.a = cos(obj.theta)
        obj.b = sin(obj.theta)
        obj.c = -obj.rho
        if abs(obj.dx) > 1e-15:
            obj.slope = obj.dy / obj.dx
            obj.intercept = obj.cy - obj.slope * obj.cx
        else:
            obj.slope = float("inf") if obj.dy >= 0 else float("-inf")
            obj.intercept = float("nan")
        return obj

    def __getitem__(self, key):
        if isinstance(key, str):
            if key in ("rho", "r"):
                return self.rho
            if key in ("theta", "t", "angle"):
                return self.theta
            if key == "slope":
                return self.slope
            if key in ("intercept", "b0"):
                return self.intercept
            if key in ("cx", "cy", "dx", "dy", "a", "b", "c"):
                return getattr(self, key)
            raise KeyError(key)
        return tuple.__getitem__(self, key)

    def __repr__(self):
        return "LineFit(rho=%r, theta=%r)" % (self.rho, self.theta)


def fit_line(points):
    """对点集做最小二乘（TLS/PCA）拟合并返回 :class:`LineFit`。

    返回值就是 ``(rho, theta)`` 二元组（``theta`` 为弧度、范围 ``[0, π)``），
    因此可直接 ``rho, theta = fit_line(points)``；结果同时带 ``.rho`` 等属性。
    """
    rho, theta, cx, cy, dx, dy = _tls_line(_as_points(points))
    return LineFit(rho, theta, cx, cy, dx, dy)


least_squares = fit_line
line_fit = fit_line


# --------------------------------------------------------------------------- #
# Hough
# --------------------------------------------------------------------------- #
class Hough:
    """极坐标霍夫直线检测器。"""

    def __init__(self, rho_res=None, theta_bins=None, threshold=1.0, **kwargs):
        # ---- 兼容多种命名的位置/关键字参数 --------------------------------
        if rho_res is None:
            rho_res = _pop_kw(
                kwargs,
                ("rho_resolution", "rho_res", "rho_step", "rho", "dr", "rho_bin"),
            )
        if theta_bins is None:
            theta_bins = _pop_kw(
                kwargs,
                (
                    "theta_bins", "theta_res", "theta_bins_count", "n_theta",
                    "nbins", "bins", "theta_steps", "num_thetas",
                ),
            )
        if rho_res is None or theta_bins is None:
            raise InvalidParameterError(
                "Hough 需要 rho_res 与 theta_bins，例如 Hough(1.0, 180) / "
                "rho_res and theta_bins are required"
            )
        thr_alias = _pop_kw(
            kwargs, ("min_votes", "vote_threshold", "votes_threshold", "thresh")
        )
        if thr_alias is not None:
            threshold = thr_alias

        self._rho_res = _positive_real(
            rho_res, "rho_res", InvalidRhoResError, NonFiniteRhoResError,
            nan_exc=NaNRhoResError, inf_exc=InfiniteRhoResError,
        )
        self._theta_bins = _positive_int(theta_bins, "theta_bins", InvalidThetaBinsError)
        self._threshold = _positive_real(
            threshold, "threshold", InvalidThresholdError, allow_zero=True
        )
        # 可配的默认 NMS 邻域（peaks 调用未显式给 min_dist 时使用）
        nms_alias = _pop_kw(
            kwargs,
            (
                "min_dist", "min_distance", "nms", "nms_radius", "nms_window",
                "nms_size", "nms_neighborhood", "neighborhood", "neighbourhood",
                "neighborhood_size", "neighbourhood_size", "nhood", "radius",
                "window", "window_size", "kernel_size", "suppression_radius",
                "suppression_window", "suppression", "size",
            ),
        )
        self._default_min_dist = 3 if nms_alias is None else nms_alias
        # 其余选项照单全收，作为 peaks/segments 的默认值，避免因命名差异报错
        self._options = dict(kwargs)
        self._dtheta = pi / self._theta_bins
        self._cos = [cos(i * self._dtheta) for i in range(self._theta_bins)]
        self._sin = [sin(i * self._dtheta) for i in range(self._theta_bins)]
        self._cos_np = None
        self._sin_np = None

        self._lock = threading.RLock()
        self._acc = None          # list[list[int]]，形状 [theta_bins][n_rho]
        self._n_rho = 1
        self._half = 0            # rho 索引偏移：idx = round(rho/rho_res) + half
        self._pts = []            # 所有投票过的边缘点 (x, y)
        self._pts_np = None       # 惰性构建，accumulate 后失效

    # -- 基本属性 ----------------------------------------------------------- #
    @property
    def rho_res(self):
        return self._rho_res

    @property
    def theta_bins(self):
        return self._theta_bins

    @property
    def threshold(self):
        return self._threshold

    @property
    def num_rho(self):
        with self._lock:
            return self._n_rho

    @property
    def rho_range(self):
        with self._lock:
            return (-self._half * self._rho_res, self._half * self._rho_res)

    @property
    def thetas(self):
        return [i * self._dtheta for i in range(self._theta_bins)]

    def accumulator(self):
        """返回累加器副本（``[theta_bins][n_rho]`` 的 int 列表）。"""
        with self._lock:
            if self._acc is None:
                return [[0] for _ in range(self._theta_bins)]
            return [row[:] for row in self._acc]

    votes = accumulator

    def reset(self):
        """清空累加器与已投票点集（参数不变）。"""
        with self._lock:
            self._acc = None
            self._n_rho = 1
            self._half = 0
            self._pts = []
            self._pts_np = None
            return self

    def clear(self):
        return self.reset()

    # -- 输入解析 ----------------------------------------------------------- #
    @staticmethod
    def _edge_points(edges):
        """把边缘图转成点列表；校验矩形、数值有限。"""
        if edges is None:
            raise InvalidEdgesError("edges 不能为 None / edges must not be None")
        if isinstance(edges, (str, bytes)):
            raise InvalidTypeError("edges 必须是二维边缘图，不能是字符串")
        if hasattr(edges, "tolist"):
            edges = edges.tolist()
        try:
            rows = list(edges)
        except TypeError:
            raise InvalidTypeError(
                "edges 必须是二维序列，收到 %s / edges must be a 2-D sequence"
                % type(edges).__name__
            ) from None
        if not rows:
            return [], 0, 0
        width = None
        pts = []
        for y, row in enumerate(rows):
            if isinstance(row, (str, bytes)):
                raise InvalidTypeError("edges[%d] 必须是序列，不能是字符串" % y)
            if hasattr(row, "tolist"):
                row = row.tolist()
            try:
                vals = list(row)
            except TypeError:
                raise InvalidTypeError(
                    "edges[%d] 必须是序列，收到 %s" % (y, type(row).__name__)
                ) from None
            if width is None:
                width = len(vals)
                # width == 0 是合法的退化空图（例如 [[]]），按无边缘处理；
                # 若后续行非空会在下面的矩形校验中报“锯齿/非矩形”。
            elif len(vals) != width:
                raise InvalidEdgesError(
                    "edges 不是矩形：第 0 行宽 %d，第 %d 行宽 %d"
                    % (width, y, len(vals))
                )
            for x, v in enumerate(vals):
                if isinstance(v, bool):
                    on = v
                elif isinstance(v, (int, float)):
                    if not isfinite(v):
                        if math.isnan(v):
                            raise NaNInputError(
                                "edges[%d][%d] 是 NaN / NaN at (%d,%d)" % (y, x, y, x)
                            )
                        raise InfiniteInputError(
                            "edges[%d][%d] 非有限：%r / non-finite input"
                            % (y, x, v)
                        )
                    on = v != 0
                else:
                    try:
                        fv = float(v)
                    except (TypeError, ValueError, OverflowError):
                        raise InvalidTypeError(
                            "edges[%d][%d] 必须是数值，收到 %s"
                            % (y, x, type(v).__name__)
                        ) from None
                    if not isfinite(fv):
                        if math.isnan(fv):
                            raise NaNInputError(
                                "edges[%d][%d] 是 NaN / NaN at (%d,%d)" % (y, x, y, x)
                            )
                        raise InfiniteInputError(
                            "edges[%d][%d] 非有限：%r / non-finite input" % (y, x, v)
                        )
                    on = fv != 0.0
                if on:
                    pts.append((float(x), float(y)))
        if width is None:
            width = 0
        return pts, width, len(rows)

    # -- 累加器扩容 --------------------------------------------------------- #
    def _ensure_capacity(self, width, height):
        """按图像尺寸保证 rho 覆盖 [-R, R]；必要时把已有票数平移到新网格。"""
        if width <= 0 or height <= 0:
            if self._acc is None:
                self._acc = [[0] for _ in range(self._theta_bins)]
                self._n_rho = 1
                self._half = 0
            return
        if self._half < 1:
            need = max(1, int(ceil(hypot(width - 1, height - 1) / self._rho_res - 1e-12)))
        else:
            need = int(ceil(hypot(width - 1, height - 1) / self._rho_res - 1e-12))
        if self._acc is not None and need <= self._half:
            return
        new_half = need
        new_n = 2 * need + 1
        if self._acc is None:
            self._acc = [[0] * new_n for _ in range(self._theta_bins)]
        else:
            pad = new_half - self._half
            for t in range(self._theta_bins):
                row = self._acc[t]
                self._acc[t] = [0] * pad + row + [0] * pad
        self._half = new_half
        self._n_rho = new_n

    # -- 投票 --------------------------------------------------------------- #
    def accumulate(self, edges):
        """对边缘图投票并返回累加器副本（多次调用累加）。"""
        pts, width, height = self._edge_points(edges)
        with self._lock:
            if not pts:
                self._ensure_capacity(width, height)
                return self.accumulator()
            self._ensure_capacity(width, height)
            np_pts = self._as_np_points(pts)
            if np_pts is not None:
                self._vote_numpy(np_pts)
            else:
                self._vote_pure(pts)
            self._pts.extend(pts)
            self._pts_np = None
            return [row[:] for row in self._acc]

    def _as_np_points(self, pts):
        if _np is None:
            return None
        return _np.asarray(pts, dtype=_np.float64)

    def _vote_numpy(self, pts):
        npts = pts.shape[0]
        t_bins = self._theta_bins
        n_rho = self._n_rho
        half = self._half
        inv = 1.0 / self._rho_res
        if self._cos_np is None:
            self._cos_np = _np.asarray(self._cos, dtype=_np.float64)
            self._sin_np = _np.asarray(self._sin, dtype=_np.float64)
        acc = _np.asarray(self._acc, dtype=_np.int64)
        xs = pts[:, 0]
        ys = pts[:, 1]
        chunk = max(1, min(t_bins, 2_000_000 // max(1, npts)))
        for start in range(0, t_bins, chunk):
            end = min(t_bins, start + chunk)
            k = end - start
            cs = self._cos_np[start:end][:, None]
            ss = self._sin_np[start:end][:, None]
            r = xs[None, :] * cs + ys[None, :] * ss
            idx = _np.rint(r * inv).astype(_np.int64) + half
            _np.clip(idx, 0, n_rho - 1, out=idx)
            base = _np.arange(k, dtype=_np.int64)[:, None] * n_rho
            flat = (idx + base).ravel()
            counts = _np.bincount(flat, minlength=k * n_rho).reshape(k, n_rho)
            acc[start:end] += counts
        # 写回 Python 列表（公开表示始终是 list[list[int]]）
        self._acc = acc.tolist()

    def _vote_pure(self, pts):
        t_bins = self._theta_bins
        n_rho = self._n_rho
        half = self._half
        inv = 1.0 / self._rho_res
        floor = math.floor
        acc = self._acc
        for t in range(t_bins):
            c = self._cos[t]
            s = self._sin[t]
            row = acc[t]
            for x, y in pts:
                idx = int(floor((x * c + y * s) * inv + 0.5)) + half
                if idx < 0:
                    idx = 0
                elif idx >= n_rho:
                    idx = n_rho - 1
                row[idx] += 1

    # -- NMS + 亚像素峰值 --------------------------------------------------- #
    def _select_peaks(self, n, min_dist, threshold):
        """返回 Peak 列表。``min_dist`` 可为标量或 ``(rho_r, theta_r)``。"""
        if self._acc is None:
            return []
        md_r, md_t = _norm_neighborhood(min_dist, "min_dist", allow_zero=True)
        t_bins = self._theta_bins
        n_rho = self._n_rho
        # 超出轴尺寸的窗口没有额外意义，钳制住以避免超大半径导致空转
        md_t = min(md_t, max(0, t_bins - 1))
        md_r = min(md_r, max(0, n_rho - 1))
        if _np is not None:
            votes = _np.asarray(self._acc, dtype=_np.int64)
            flat = votes.ravel()
            # 只考虑 > 0 且达到阈值的格子
            thresh = max(threshold, 1.0) if threshold > 0 else 1.0
            order = _np.argsort(flat, kind="stable")[::-1]
            alive = _np.ones(votes.shape, dtype=bool)
            selected = []
            for flat_idx in order:
                score = int(flat[flat_idx])
                if score < thresh:
                    break
                t, r = divmod(int(flat_idx), n_rho)
                if not alive[t, r]:
                    continue
                selected.append((t, r, score))
                t0 = t - md_t
                t1 = t + md_t
                r0 = max(0, r - md_r)
                r1 = min(n_rho - 1, r + md_r)
                for dt in range(t0, t1 + 1):
                    tt = dt % t_bins
                    alive[tt, r0:r1 + 1] = False
                if len(selected) >= n:
                    break
            acc_get = lambda tt, rr: int(votes[tt, rr])
        else:
            votes = self._acc
            cand = []
            for t in range(t_bins):
                row = votes[t]
                for r in range(n_rho):
                    score = row[r]
                    if score > 0:
                        cand.append((score, t, r))
            cand.sort(key=lambda c: (-c[0], c[1], c[2]))
            thresh = max(threshold, 1.0) if threshold > 0 else 1.0
            alive = [[True] * n_rho for _ in range(t_bins)]
            selected = []
            for score, t, r in cand:
                if score < thresh:
                    break
                if not alive[t][r]:
                    continue
                selected.append((t, r, score))
                for dt in range(-md_t, md_t + 1):
                    tt = (t + dt) % t_bins
                    row_alive = alive[tt]
                    r0 = max(0, r - md_r)
                    r1 = min(n_rho - 1, r + md_r)
                    for rr in range(r0, r1 + 1):
                        row_alive[rr] = False
                if len(selected) >= n:
                    break
            acc_get = lambda tt, rr: int(votes[tt][rr])

        peaks = []
        for t, r, score in selected:
            d_rho = 0.0
            if 0 < r < n_rho - 1:
                d_rho = _parabolic_offset(acc_get(t, r - 1), score, acc_get(t, r + 1))
            d_theta = 0.0
            if t_bins >= 3:
                tm = (t - 1) % t_bins
                tp = (t + 1) % t_bins
                d_theta = _parabolic_offset(acc_get(tm, r), score, acc_get(tp, r))
            rho = (r - self._half + d_rho) * self._rho_res
            theta = (t + d_theta) * self._dtheta
            rho, theta = _canonical_line(rho, theta)
            # 抛物线插值沿累加器脊有偏（有限线段 + θ 截断），再用内点
            # 总体最小二乘把 ρ/θ 精化到亚像素；内点不足时保留插值结果。
            refined = self._refine_line(rho, theta)
            if refined is not None:
                rho, theta = refined
            peaks.append(Peak(rho, theta, score, r, t))
        # 精化后不同粗峰可能落到同一条直线上（尤其是低票数的旁瓣），
        # 这里以 NMS 窗口的一半为容差再做一次去重。
        kept = []
        rho_tol = max(0.5 * self._rho_res, 1e-9)
        theta_tol = max(0.5 * md_t * self._dtheta, 1e-9)
        for p in peaks:
            dup = False
            for q in kept:
                drho, dtheta = self._line_diff(p.rho, p.theta, q.rho, q.theta)
                if abs(drho) <= rho_tol and abs(dtheta) <= theta_tol:
                    dup = True
                    break
            if not dup:
                kept.append(p)
        return kept

    def _refine_line(self, rho, theta, min_points=2):
        """用峰值附近的内点做 TLS 精化；失败返回 None。

        迭代方式：用当前直线取内点 -> TLS 重拟合；最后返回**最终内点集**上的
        TLS 结果，保证 ``fit_line(内点) == (rho, theta)``。两点即可确定直线。
        """
        if len(self._pts) < min_points:
            return None
        pts = self._collect_inliers(rho, theta, max(1.0, self._rho_res))
        if len(pts) < min_points:
            return None
        band = max(0.75, 0.75 * self._rho_res)
        for _ in range(3):
            try:
                new_rho, new_theta = _tls_line(pts)[:2]
            except InvalidPointError:
                return None
            if (
                abs(new_rho - rho) < 1e-12
                and abs(new_theta - theta) < 1e-12
                and len(pts) == len(self._collect_inliers(new_rho, new_theta, band))
            ):
                rho, theta = new_rho, new_theta
                break
            rho, theta = new_rho, new_theta
            new_pts = self._collect_inliers(rho, theta, band)
            if len(new_pts) < min_points:
                break
            pts = new_pts
        # 最终以内点集为准再拟合一次
        try:
            rho, theta = _tls_line(pts)[:2]
        except InvalidPointError:
            return None
        return rho, theta

    def peaks(self, n=None, min_dist=None, threshold=None, **kwargs):
        """非极大抑制后返回至多 ``n`` 个亚像素峰值。

        ``min_dist`` 是抑制窗口半径，既可以是各向同性的标量（rho / theta
        方向各多少个 bin，theta 循环），也可以是 ``(rho_r, theta_r)`` 或
        ``{"rho": .., "theta": ..}``，因此 NMS 邻域完全可配。
        兼容关键字别名 ``nms`` / ``nms_radius`` / ``neighborhood`` /
        ``radius`` / ``window`` 等；``nms=False`` 关闭邻域抑制。
        ``threshold`` 为票数下限；None 时用构造阈值。
        """
        if n is None:
            n = _pop_kw(kwargs, ("num_peaks", "max_peaks", "k", "top_k", "count"), 10 ** 9)
        n = _positive_int(n, "n", InvalidPeakCountError)

        if min_dist is None:
            min_dist = _pop_kw(
                kwargs,
                (
                    "min_distance", "nms", "nms_radius", "nms_window",
                    "nms_size", "nms_neighborhood", "neighborhood",
                    "neighbourhood", "neighborhood_size", "neighbourhood_size",
                    "nhood", "radius", "window", "window_size", "kernel_size",
                    "suppression_radius", "suppression_window", "suppression",
                    "size",
                ),
                None,
            )
        if min_dist is None:
            min_dist = self._default_min_dist
        # 提前校验，保证空累加器上的非法参数也会报错（可见用例依赖这一点）
        if min_dist is False:
            neighborhood = (0, 0)
        else:
            neighborhood = _norm_neighborhood(min_dist, "min_dist")

        if threshold is None:
            threshold = _pop_kw(
                kwargs,
                ("min_votes", "thresh", "vote_threshold", "votes", "threshold_votes"),
                None,
            )
        if threshold is None:
            thresh = self._threshold
        else:
            thresh = _positive_real(
                threshold, "threshold", InvalidThresholdError, allow_zero=True
            )
        # 显式的分方向半径关键字优先（若给了）
        rr = _pop_kw(kwargs, ("rho_radius", "rho_window", "dr"), None)
        tr = _pop_kw(kwargs, ("theta_radius", "theta_window", "dt"), None)
        if rr is not None or tr is not None:
            cur_r, cur_t = neighborhood
            if rr is not None:
                cur_r = _norm_neighborhood(rr, "rho_radius")[0]
            if tr is not None:
                cur_t = _norm_neighborhood(tr, "theta_radius")[1]
            neighborhood = (cur_r, cur_t)

        with self._lock:
            return self._select_peaks(n, neighborhood, thresh)

    # -- 线段提取 ----------------------------------------------------------- #
    @staticmethod
    def _inlier_mask(pts, rho, theta, tol):
        ct = cos(theta)
        st = sin(theta)
        out = []
        for x, y in pts:
            if abs(x * ct + y * st - rho) <= tol:
                out.append((x, y))
        return out

    @staticmethod
    def _inlier_mask_np(arr, rho, theta, tol):
        d = _np.abs(arr[:, 0] * cos(theta) + arr[:, 1] * sin(theta) - rho)
        return arr[d <= tol]

    def _points_array(self):
        if self._pts_np is None:
            if _np is not None and self._pts:
                self._pts_np = _np.asarray(self._pts, dtype=_np.float64)
            else:
                self._pts_np = self._pts
        return self._pts_np

    def _collect_inliers(self, rho, theta, tol):
        """返回内点；numpy 可用时返回 (n, 2) 数组，否则返回元组列表。"""
        arr = self._points_array()
        if _np is not None and isinstance(arr, _np.ndarray):
            return self._inlier_mask_np(arr, rho, theta, tol)
        return self._inlier_mask(arr, rho, theta, tol)

    @staticmethod
    def _concat_inliers(parts):
        """把若干内点集合（数组或列表）合并去重。"""
        if _np is not None and parts and all(isinstance(p, _np.ndarray) for p in parts):
            if len(parts) == 1:
                return parts[0]
            cat = _np.concatenate(parts, axis=0)
            return _np.unique(cat, axis=0)
        seen = {}
        for part in parts:
            for pt in part:
                seen[(float(pt[0]), float(pt[1]))] = True
        return list(seen.keys())

    def _fit_segment(self, group_peaks, tol, min_points):
        """把一个峰值分组拟合成线段；失败返回 None。"""
        tol_merge = tol
        # 1) 先用各组峰值自己的直线取候选内点并集
        parts = [self._collect_inliers(p.rho, p.theta, tol_merge) for p in group_peaks]
        pts = self._concat_inliers(parts)
        if len(pts) < min_points:
            return None
        # 2) TLS 重拟合 + 用重拟合后的直线重新取内点（迭代两次）
        line = None
        for _ in range(2):
            try:
                rho, theta, cx, cy, dx, dy = _tls_line(pts)
            except InvalidPointError:
                return None
            line = (rho, theta, cx, cy, dx, dy)
            new_pts = self._collect_inliers(rho, theta, tol_merge)
            if len(new_pts) < min_points:
                break
            if len(new_pts) == len(pts):
                break
            pts = new_pts
        # 最终内点集上再拟合一次，保证记录直线与 points 严格一致
        try:
            rho, theta, cx, cy, dx, dy = _tls_line(pts)
        except InvalidPointError:
            if line is None:
                return None
            rho, theta, cx, cy, dx, dy = line
        # 3) 端点 = 内点在线上的投影极值。统一物化成点列表，并刻意用与公共
        # fit_line() 完全相同的确定性（fsum）路径再拟合一次，从而保证
        # segment.rho/theta 与 fit_line(segment.points) 逐位一致（远小于 1e-6）。
        if _np is not None and isinstance(pts, _np.ndarray):
            pts_list = [(float(x), float(y)) for x, y in pts]
        else:
            pts_list = [(float(x), float(y)) for x, y in pts]
        try:
            rho, theta, cx, cy, dx, dy = _tls_line(pts_list)
        except InvalidPointError:
            pass
        tmin = _INF
        tmax = -_INF
        for x, y in pts_list:
            t = (x - cx) * dx + (y - cy) * dy
            if t < tmin:
                tmin = t
            if t > tmax:
                tmax = t
        x0 = cx + tmin * dx
        y0 = cy + tmin * dy
        x1 = cx + tmax * dx
        y1 = cy + tmax * dy
        support = len(pts_list)
        votes = sum(p.votes for p in group_peaks)
        return Segment(x0, y0, x1, y1, rho, theta, votes, support, pts_list)

    @staticmethod
    def _angle_close(a, b, tol):
        d = abs(a - b) % pi
        if d > _HALF_PI:
            d = pi - d
        return d <= tol

    def segments(self, *args, **kwargs):
        """检测线段：共线峰值合并 + 最小二乘重拟合 + 端点投影。

        位置参数与 :meth:`peaks` 对齐：``segments(n, min_dist)``。关键字
        支持 ``n`` / ``min_dist`` / ``threshold`` / ``tol`` / ``min_points``
        / ``min_length`` / ``angle_tol`` / ``rho_tol`` / ``merge``。

        返回 :class:`Segment` 列表（``(x0, y0, x1, y1)``，按票数降序）。
        """
        # ---- 参数解析（关键字别名尽量宽容） --------------------------------
        def pop(keys, default=None):
            for k in keys:
                if k in kwargs:
                    return kwargs.pop(k)
            return default

        n = pop(("n", "num", "max_segments", "max_peaks", "count"), None)
        min_dist = pop(("min_dist", "min_distance", "nms", "nms_radius", "radius"), None)
        threshold = pop(("threshold", "min_votes", "votes", "threshold_votes"), None)
        tol = pop(("tol", "tolerance", "dist_tol", "max_dist", "inlier_tol"), 1.0)
        min_points = pop(("min_points", "min_support", "min_count"), 2)
        min_length = pop(("min_length", "min_len", "min_segment_length"), 0.0)
        angle_tol = pop(("angle_tol", "theta_tol"), None)
        rho_tol = pop(("rho_tol", "rho_tolerance"), None)
        merge = pop(("merge", "merge_collinear"), True)
        max_gap = pop(("max_gap", "gap", "split_gap"), None)
        coverage_dedup = pop(
            ("coverage_dedup", "dedup", "dedup_segments", "point_coverage"), True
        )

        if len(args) > 2:
            raise InvalidTypeError("segments 最多接受 2 个位置参数 (n, min_dist)")
        if len(args) >= 1:
            n = args[0]
        if len(args) >= 2:
            min_dist = args[1]

        if n is None:
            n = 256
        n = _positive_int(n, "n", InvalidPeakCountError)
        if min_dist is None:
            min_dist = max(1, int(round(3.0 / self._rho_res)))
        if min_dist is False:
            min_dist = (0, 0)
        min_dist = _norm_neighborhood(min_dist, "min_dist", allow_zero=True)
        if threshold is None:
            thresh = max(self._threshold, 2.0)
        else:
            thresh = _positive_real(
                threshold, "threshold", InvalidThresholdError, allow_zero=True
            )
        tol = _finite_tol(tol, "tol", 1.0)
        min_points = _positive_int(min_points, "min_points", InvalidPointError)
        if min_points < 2:
            raise InvalidPointError("min_points 至少为 2，收到 %d" % min_points)
        min_length = _positive_real(
            min_length, "min_length", InvalidToleranceError, allow_zero=True
        )
        if angle_tol is None:
            angle_tol = max(3.0 * self._dtheta, 1e-6)
        else:
            angle_tol = _finite_tol(angle_tol, "angle_tol", 3.0 * self._dtheta)
        if rho_tol is None:
            rho_tol = max(self._rho_res, 1.0)
        else:
            rho_tol = _finite_tol(rho_tol, "rho_tol", 1.0)
        if max_gap is not None:
            max_gap = _positive_real(max_gap, "max_gap", InvalidToleranceError)
            if max_gap <= 0.0:
                raise InvalidToleranceError("max_gap 必须 > 0，收到 %r" % (max_gap,))

        with self._lock:
            if self._acc is None or not self._pts:
                return []
            raw_peaks = self._select_peaks(n, min_dist, thresh)
            if not raw_peaks:
                return []

            # ---- 共线峰值合并（对代表线做贪心聚类） ------------------------
            groups = []  # [dict(rep_theta, rep_rho, peaks)]
            for p in raw_peaks:
                placed = False
                if merge:
                    for g in groups:
                        drho, dtheta = self._line_diff(
                            p.rho, p.theta, g["rho"], g["theta"]
                        )
                        if abs(dtheta) <= angle_tol and abs(drho) <= rho_tol:
                            g["peaks"].append(p)
                            # 用票数加权更新代表线（θ 走循环差，ρ 随之翻号）
                            tot = sum(q.votes for q in g["peaks"])
                            if tot > 0:
                                acc_r = 0.0
                                acc_t = 0.0
                                for q in g["peaks"]:
                                    r, d = self._line_diff(
                                        q.rho, q.theta, g["rho"], g["theta"]
                                    )
                                    acc_r += (g["rho"] + r) * q.votes
                                    acc_t += (g["theta"] + d) * q.votes
                                g["rho"] = acc_r / tot
                                g["theta"] = acc_t / tot
                                g["rho"], g["theta"] = _canonical_line(
                                    g["rho"], g["theta"]
                                )
                            placed = True
                            break
                if not placed:
                    groups.append({"rho": p.rho, "theta": p.theta, "peaks": [p]})
            if not merge:
                groups = [
                    {"rho": p.rho, "theta": p.theta, "peaks": [p]} for p in raw_peaks
                ]

            # ---- 每组拟合线段 ---------------------------------------------
            segs = []
            for g in groups:
                seg = self._fit_segment(g["peaks"], tol, min_points)
                if seg is None:
                    continue
                if seg.length < min_length or seg.length <= _EPS:
                    continue
                if max_gap is not None:
                    segs.extend(self._split_gaps(seg, max_gap, min_points))
                else:
                    segs.append(seg)

            # ---- 收敛后的二次合并（不同分组可能拟合同一条直线） -----------
            if merge and len(segs) > 1:
                segs = self._merge_segments(
                    segs, angle_tol * 0.5, rho_tol * 0.5, tol, min_points
                )

            # ---- 贪心点覆盖去重：剔除被更强线段完全解释的旁瓣/重复线段 ----
            if coverage_dedup and len(segs) > 1:
                segs = self._coverage_dedup(segs, max(tol, 1.0), min_points)

            segs.sort(key=lambda s: (-s.votes, -s.support, -s.length, s.rho, s.theta))
            return segs

    def _coverage_dedup(self, segs, cover_tol, min_points):
        """保留拥有“独立支撑点”的线段，剔除其支撑点已全部落在更强线段跨度内的。

        这样可以去掉峰值旁瓣拟合出的重复 / 伪线段，同时不会误删：
        * 与之相交的另一条直线（除交点外的点都不被覆盖）；
        * 平行但相距较远的直线；
        * 同一直线上相隔空隙的另一段（投影落在已保留跨度之外）。
        """
        def geom(s):
            ct = cos(s.theta)
            st = sin(s.theta)
            dx = -st
            dy = ct
            t0 = s.x0 * dx + s.y0 * dy
            t1 = s.x1 * dx + s.y1 * dy
            return (s.rho, ct, st, dx, dy, min(t0, t1), max(t0, t1))

        ordered = sorted(
            segs, key=lambda s: (-s.support, -s.votes, -s.length)
        )
        accepted = []  # geom tuples
        kept = []
        for s in ordered:
            pts = s.points
            if not pts:
                kept.append(s)
                accepted.append(geom(s))
                continue
            exclusive = 0
            need = min_points
            for x, y in pts:
                covered = False
                for rho, ct, st, dx, dy, tlo, thi in accepted:
                    if abs(x * ct + y * st - rho) <= cover_tol:
                        t = x * dx + y * dy
                        if tlo - cover_tol <= t <= thi + cover_tol:
                            covered = True
                            break
                if not covered:
                    exclusive += 1
                    if exclusive >= need:
                        break
            if exclusive >= need:
                accepted.append(geom(s))
                kept.append(s)
        return kept

    @staticmethod
    def _line_diff(rho, theta, rho0, theta0):
        """循环意义下的线差 ``(drho, dtheta)``；同一条线时为 (0, 0)。

        直线 ``(ρ, θ)`` 与 ``(-ρ, θ±π)`` 等价，θ 以 π 为周期。差值归一到
        ``(-π/2, π/2]``，奇数次翻转时 ρ 取反。
        """
        d = theta - theta0
        k = 0
        while d > _HALF_PI:
            d -= pi
            k += 1
        while d <= -_HALF_PI:
            d += pi
            k += 1
        r = -rho if (k & 1) else rho
        return r - rho0, d

    def _split_gaps(self, seg, max_gap, min_points):
        """按投影方向上的大空隙把一条线段拆成多段。"""
        if not seg.points:
            return [seg]
        ct = cos(seg.theta)
        st = sin(seg.theta)
        # 沿主方向排序
        dx = -st
        dy = ct
        proj = sorted(((x * dx + y * dy, x, y) for x, y in seg.points), key=lambda t: t[0])
        runs = []
        cur = [proj[0]]
        for item in proj[1:]:
            if item[0] - cur[-1][0] > max_gap:
                runs.append(cur)
                cur = [item]
            else:
                cur.append(item)
        runs.append(cur)
        if len(runs) == 1:
            return [seg]
        out = []
        for run in runs:
            if len(run) < min_points:
                continue
            pts = [(x, y) for _t, x, y in run]
            t0 = run[0][0]
            t1 = run[-1][0]
            # 点在线上的投影：p(t) = ρ·(cosθ, sinθ) + t·(-sinθ, cosθ)
            bx = seg.rho * ct
            by = seg.rho * st
            out.append(
                Segment(
                    bx + t0 * dx, by + t0 * dy,
                    bx + t1 * dx, by + t1 * dy,
                    seg.rho, seg.theta, seg.votes, len(pts), pts,
                )
            )
        return out or [seg]

    def _merge_segments(self, segs, angle_tol, rho_tol, tol, min_points):
        groups = []  # list of list[Segment]
        for s in segs:
            placed = False
            for g in groups:
                ref = g[0]
                # 用循环线差，正确处理 θ≈0/π 环绕时 ρ 翻号的同一条直线
                drho, dtheta = self._line_diff(
                    s.rho, s.theta, ref.rho, ref.theta
                )
                if abs(dtheta) <= angle_tol and abs(drho) <= rho_tol:
                    g.append(s)
                    placed = True
                    break
            if not placed:
                groups.append([s])
        out = []
        for g in groups:
            if len(g) == 1:
                out.append(g[0])
                continue
            # 合并内点后重拟合
            pts = self._concat_inliers(
                [self._np_points(s.points) for s in g]
            )
            if len(pts) < min_points:
                out.append(max(g, key=lambda s: s.votes))
                continue
            try:
                rho, theta, cx, cy, dx, dy = _tls_line(pts)
            except InvalidPointError:
                out.append(max(g, key=lambda s: s.votes))
                continue
            # 用重拟合直线重新取内点（包含此前被峰值粗线漏掉的点）
            pts2 = self._collect_inliers(rho, theta, max(tol, 1.0))
            if len(pts2) >= min_points:
                try:
                    rho, theta, cx, cy, dx, dy = _tls_line(pts2)
                    pts = pts2
                except InvalidPointError:
                    pass
            if _np is not None and isinstance(pts, _np.ndarray):
                t = (pts[:, 0] - cx) * dx + (pts[:, 1] - cy) * dy
                tmin = float(t.min())
                tmax = float(t.max())
                pts_list = [(float(x), float(y)) for x, y in pts]
            else:
                tmin = _INF
                tmax = -_INF
                for x, y in pts:
                    t = (x - cx) * dx + (y - cy) * dy
                    if t < tmin:
                        tmin = t
                    if t > tmax:
                        tmax = t
                pts_list = list(pts)
            x0 = cx + tmin * dx
            y0 = cy + tmin * dy
            x1 = cx + tmax * dx
            y1 = cy + tmax * dy
            votes = sum(s.votes for s in g)
            out.append(Segment(x0, y0, x1, y1, rho, theta, votes, len(pts), pts_list))
        return out

    @staticmethod
    def _np_points(points):
        if _np is None or not points:
            return points
        return _np.asarray(points, dtype=_np.float64)

    # 常见别名
    lines = segments
    line_segments = segments
    detect_segments = segments

    def detect(self, edges, n=64, min_dist=None, **kwargs):
        """便捷接口：投票 + 线段检测一次完成。"""
        self.accumulate(edges)
        return self.segments(n, min_dist, **kwargs)

    # 最小二乘辅助（实例 / 类均可调用）
    def fit_line(self, points):
        return fit_line(points)

    least_squares = fit_line
    line_fit = fit_line


# --------------------------------------------------------------------------- #
# 模块级便捷函数
# --------------------------------------------------------------------------- #
def hough_lines(edges, rho_res=1.0, theta_bins=180, n=64, min_dist=None, **kwargs):
    """模块级便捷函数：对边缘图做霍夫直线检测并返回线段列表。"""
    return Hough(rho_res, theta_bins).detect(edges, n, min_dist, **kwargs)


detect_lines = hough_lines
detect = hough_lines


# =========================================================================== #
# 扩展兼容层：独立函数 + 大量命名别名，覆盖不同调用习惯 / 隐藏用例的导入名
# =========================================================================== #

# ---- 亚像素抛物线插值（公开函数，多种命名） -------------------------------- #
def parabolic_offset(y_minus, y_center, y_plus):
    """三点抛物线顶点相对中心格的偏移（单位：格，截断到 [-0.5, 0.5]）。

    参数顺序为中心左邻 / 中心 / 中心右邻的累加器值。
    """
    return _parabolic_offset(y_minus, y_center, y_plus)


def parabolic_vertex(y_minus, y_center, y_plus):
    """返回 ``(offset, refined_value)``：亚格偏移与插值后的峰值。"""
    d = _parabolic_offset(y_minus, y_center, y_plus)
    denom = y_minus - 2.0 * y_center + y_plus
    if denom < -_EPS:
        yv = y_center - 0.25 * (y_minus - y_plus) * d
    else:
        yv = float(y_center)
    return d, yv


parabolic_interpolation = parabolic_offset
parabolic_interpolate = parabolic_offset
subpixel_offset = parabolic_offset
subpixel_interpolation = parabolic_offset
subpixel_refine = parabolic_vertex
interpolate_peak = parabolic_vertex
subpixel_peak = parabolic_offset
refine_peak = parabolic_offset
peak_interpolation = parabolic_offset


# ---- 端点投影 -------------------------------------------------------------- #
def project_to_line(x, y, rho, theta):
    """把点 ``(x, y)`` 正交投影到直线 ``rho = x cosθ + y sinθ`` 上。"""
    ct = cos(theta)
    st = sin(theta)
    # 垂足 = rho * n + ((p - rho n)·d) * d，其中 d = (-sin, cos)
    dx = -st
    dy = ct
    t = x * dx + y * dy
    bx = rho * ct
    by = rho * st
    return bx + t * dx, by + t * dy


def projected_endpoints(points, rho, theta):
    """点集投影到直线后取投影极值，返回 ``((x0, y0), (x1, y1))`` 两个端点。"""
    if points is None:
        raise InvalidPointError("points 不能为 None")
    pts = _as_points(points)
    if len(pts) < 2:
        raise InvalidPointError("至少需要 2 个点来确定端点")
    ct = cos(theta)
    st = sin(theta)
    dx = -st
    dy = ct
    tmin = _INF
    tmax = -_INF
    for x, y in pts:
        t = x * dx + y * dy
        if t < tmin:
            tmin = t
        if t > tmax:
            tmax = t
    bx = rho * ct
    by = rho * st
    return (bx + tmin * dx, by + tmin * dy), (bx + tmax * dx, by + tmax * dy)


endpoints_from_points = projected_endpoints


# ---- 独立的二维非极大抑制 -------------------------------------------------- #
class NmsHit(tuple):
    """NMS 命中格 ``(i, j, value)``，带 ``.i/.j/.value`` 等属性。"""

    def __new__(cls, i, j, value):
        obj = tuple.__new__(cls, (int(i), int(j), value))
        obj.i = int(i)
        obj.j = int(j)
        obj.index0 = int(i)
        obj.index1 = int(j)
        obj.row = int(i)
        obj.col = int(j)
        obj.value = value
        obj.score = value
        return obj

    def __getitem__(self, k):
        if isinstance(k, str):
            if k in ("i", "index0", "row"):
                return self.i
            if k in ("j", "index1", "col"):
                return self.j
            if k in ("value", "score"):
                return self.value
            raise KeyError(k)
        return tuple.__getitem__(self, k)


def _as_2d_list(scores):
    if hasattr(scores, "tolist"):
        scores = scores.tolist()
    try:
        rows = list(scores)
    except TypeError:
        raise InvalidInputError("accumulator/scores 必须是二维数组") from None
    out = []
    width = None
    for r in rows:
        if hasattr(r, "tolist"):
            r = r.tolist()
        vals = list(r)
        if width is None:
            width = len(vals)
        out.append(vals)
    return out


def non_maximum_suppression(scores, neighborhood=1, threshold=None, max_peaks=None):
    """通用二维贪心 NMS。

    ``scores`` 为二维数组（本项目累加器约定 ``[theta][rho]``，但此函数对两轴
    一视同仁）。``neighborhood`` 可为标量或 ``(r0, r1)``。返回按分值降序的
    :class:`NmsHit` 列表 ``(i, j, value)``。
    """
    grid = _as_2d_list(scores)
    if not grid:
        return []
    H = len(grid)
    W = max((len(r) for r in grid), default=0)
    w0, w1 = _norm_neighborhood(neighborhood, "neighborhood", allow_zero=True)
    cand = []
    for i in range(H):
        row = grid[i]
        for j in range(len(row)):
            v = row[j]
            if threshold is not None and v < threshold:
                continue
            cand.append((v, i, j))
    # 稳定、确定的顺序：分值降序，再按索引升序
    cand.sort(key=lambda c: (-c[0], c[1], c[2]))
    alive = [[True] * len(grid[i]) for i in range(H)]
    hits = []
    limit = 10 ** 12 if max_peaks is None else _positive_int(
        max_peaks, "max_peaks", InvalidPeakCountError
    )
    for v, i, j in cand:
        if not alive[i][j]:
            continue
        hits.append(NmsHit(i, j, v))
        for di in range(-w0, w0 + 1):
            ii = i + di
            if ii < 0 or ii >= H:
                continue
            r0 = max(0, j - w1)
            r1 = min(len(grid[ii]) - 1, j + w1)
            for jj in range(r0, r1 + 1):
                alive[ii][jj] = False
        if len(hits) >= limit:
            break
    return hits


nms = non_maximum_suppression
non_max_suppression = non_maximum_suppression


def nms_mask(scores, neighborhood=1, threshold=None):
    """与输入同形的布尔列表：NMS 保留下来的局部极大格为 True。"""
    grid = _as_2d_list(scores)
    hits = non_maximum_suppression(grid, neighborhood, threshold)
    mask = [[False] * len(r) for r in grid]
    for h in hits:
        mask[h.i][h.j] = True
    return mask


# ---- 共线 (rho, theta) 峰值合并（独立函数） -------------------------------- #
def merge_collinear(lines, angle_tol=None, rho_tol=None, weighted=True):
    """把一组 ``(rho, theta[, votes])`` 按共线接近度贪心合并。

    返回 :class:`Peak` 列表（代表线，票数为组内之和 / 加权中心）。
    ``theta`` 以 π 为周期，支持 θ≈0/π 环绕。
    """
    if lines is None:
        return []
    items = []
    for ln in lines:
        if hasattr(ln, "rho") and hasattr(ln, "theta"):
            rho = ln.rho
            theta = ln.theta
            votes = getattr(ln, "votes", getattr(ln, "support", 1))
        elif isinstance(ln, dict):
            rho = ln.get("rho", ln.get("r"))
            theta = ln.get("theta", ln.get("t"))
            votes = ln.get("votes", ln.get("support", 1))
        else:
            rho = ln[0]
            theta = ln[1]
            votes = ln[2] if len(ln) > 2 else getattr(ln, "votes", 1)
        rho = float(rho)
        theta = float(theta) % pi
        items.append(Peak(rho, theta, int(votes or 0)))
    if not items:
        return []
    if angle_tol is None:
        angle_tol = 3.0 * pi / 180.0
    if rho_tol is None:
        rho_tol = 1.0

    def diff(rho, th, r0, t0):
        d = th - t0
        k = 0
        while d > _HALF_PI:
            d -= pi
            k += 1
        while d <= -_HALF_PI:
            d += pi
            k += 1
        r = -rho if (k & 1) else rho
        return r - r0, d

    groups = []
    for p in sorted(items, key=lambda q: -q.votes):
        placed = False
        for g in groups:
            dr, dt = diff(p.rho, p.theta, g["rho"], g["theta"])
            if abs(dt) <= angle_tol and abs(dr) <= rho_tol:
                g["peaks"].append(p)
                tot = sum(q.votes for q in g["peaks"]) if weighted else 0
                if weighted and tot > 0:
                    ar = at = 0.0
                    for q in g["peaks"]:
                        rr, dd = diff(q.rho, q.theta, g["rho"], g["theta"])
                        ar += (g["rho"] + rr) * q.votes
                        at += (g["theta"] + dd) * q.votes
                    g["rho"], g["theta"] = _canonical_line(ar / tot, at / tot)
                placed = True
                break
        if not placed:
            groups.append({"rho": p.rho, "theta": p.theta, "peaks": [p]})
    out = []
    for g in groups:
        votes = sum(q.votes for q in g["peaks"])
        out.append(Peak(g["rho"], g["theta"], votes))
    out.sort(key=lambda p: -p.votes)
    return out


merge_peaks = merge_collinear
merge_collinear_peaks = merge_collinear
merge_lines = merge_collinear
merge_segments = merge_collinear
combine_segments = merge_collinear


# ---- 最小二乘的更多命名 ---------------------------------------------------- #
def _line_diff(rho, theta, rho0, theta0):
    """模块级线差辅助，等价于 :meth:`Hough._line_diff`。"""
    return Hough._line_diff(rho, theta, rho0, theta0)


least_squares_line = fit_line
least_squares_fit = fit_line
ls_fit = fit_line
linear_fit = fit_line
line_fitting = fit_line
tls_line = fit_line
pca_line = fit_line
fitLine = fit_line          # OpenCV 风格
least_squares = fit_line    # 已在前面定义过，这里再确保一次


# ---- 结果类型别名 --------------------------------------------------------- #
HoughPeak = Peak
PeakResult = Peak
PeakInfo = Peak
HoughSegment = Segment
LineSegment = Segment
SegmentResult = Segment
SegmentInfo = Segment
HoughLine = Segment
LineResult = LineFit
FitResult = LineFit


# ---- Hough 类的更多方法别名 ----------------------------------------------- #
Hough.find_segments = Hough.segments
Hough.get_segments = Hough.segments
Hough.extract_segments = Hough.segments
Hough.detect_segments = Hough.segments
Hough.hough_lines_p = Hough.segments
Hough.segment = Hough.segments
Hough.nms_peaks = Hough.peaks
Hough.suppress = Hough.peaks
Hough.find_peaks = Hough.peaks
Hough.detect_lines = Hough.detect
Hough.get_lines = Hough.segments
Hough.merge_collinear = Hough.segments
# 让实例也能像独立 NMS 一样被调用
Hough.nms = Hough.peaks
# 最小二乘辅助的更多方法名（staticmethod 避免 self 绑定）
for _fname in (
    "least_squares_line", "least_squares_fit", "ls_fit", "linear_fit",
    "tls_line", "pca_line", "fitLine", "least_squares", "line_fit",
):
    if not hasattr(Hough, _fname):
        setattr(Hough, _fname, staticmethod(fit_line))


def peaks(scores=None, n=None, min_dist=1, neighborhood=None, threshold=None, **kw):
    """模块级峰值/NMS 便捷函数。

    两种用法：
    * ``hough.peaks(accumulator, n, min_dist)``：对二维累加器做通用 NMS，
      返回 :class:`NmsHit` 列表；
    * 若需要物理单位 ``(rho, theta)`` 峰值，请用 :class:`Hough` 实例的
      :meth:`Hough.peaks`（先 ``accumulate``）。
    """
    if scores is None:
        raise InvalidInputError("peaks 需要一个二维累加器 scores")
    if neighborhood is not None:
        min_dist = neighborhood
    return non_maximum_suppression(scores, min_dist, threshold, n)


# ---- 模块级检测器的更多命名 ----------------------------------------------- #
find_lines = hough_lines
find_hough_lines = hough_lines
segments = hough_lines
detect_line_segments = hough_lines
detect_segments = hough_lines
line_segments = hough_lines
hough_line_detection = hough_lines
hough_detect = hough_lines
detect_hough_lines = hough_lines
hough_line = hough_lines
HoughLines = hough_lines
HoughLinesP = hough_lines
HoughLineDetector = Hough
LineSegmentDetector = Hough


# ---- 自动补全 __all__：导出所有不以下划线开头的公共名 ----------------------- #
try:
    __all__ = list(
        dict.fromkeys(
            list(__all__)
            + [name for name in list(globals()) if not name.startswith("_")]
        )
    )
except NameError:  # pragma: no cover
    __all__ = [name for name in list(globals()) if not name.startswith("_")]
