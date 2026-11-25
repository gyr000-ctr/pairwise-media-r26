# inpaint

## 题面（可直接粘贴）

子目录：inpaint。扩散图像修补。约束：掩码内像素由边界调和插值（与 Jacobi 迭代到收敛解差 ≤1e-4）；掩码外像素逐像素不变（≤1e-12）；全掩码与非连通掩码可区分报错；迭代收敛判据明确；100 万像素不超过 900 ms；并发安全。验收：cd inpaint && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd inpaint && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图像处理
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
