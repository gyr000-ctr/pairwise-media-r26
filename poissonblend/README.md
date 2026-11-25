# poissonblend

## 题面（可直接粘贴）

子目录：poissonblend。泊松融合。约束：梯度场保持一致，融合后边界梯度误差 ≤1e-3；求解泊松方程（离散 Laplacian）与直接解一致（≤1e-6）；源图平移不改变结果（平移不变性）；掩码外像素不变；100 万像素不超过 900 ms；非法掩码可区分报错。验收：cd poissonblend && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd poissonblend && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图像处理
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
