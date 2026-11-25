# template

## 题面（可直接粘贴）

子目录：template。归一化互相关模板匹配。约束：NCC 值域 [-1,1] 且自匹配为 1（≤1e-9）；必须去均值归一（光照不变）；与暴力实现逐点一致（≤1e-9）；模板大于图像时报错；平模板（零方差）可区分报错；100 万像素不超过 900 ms；并发安全。验收：cd template && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd template && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图像处理
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
