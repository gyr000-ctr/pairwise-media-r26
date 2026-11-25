# superpixel

## 题面（可直接粘贴）

子目录：superpixel。SLIC 超像素分割。约束：每个像素归属唯一标签且标签连续从 0 起；簇中心必须等于成员均值（≤1e-6）；迭代收敛判据明确且稳定；连通性后处理保证每个超像素单连通；10 万像素不超过 900 ms；非法超像素数可区分报错。验收：cd superpixel && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd superpixel && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图像处理
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
