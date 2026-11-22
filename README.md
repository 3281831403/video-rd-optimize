# video-rd-optimize

## 题面

编码器接入 RD 优化后，码率暴涨、本该 skip 的块还在编码：`lambda_for_qp` 把指数步长写成 `(qp−12)/6`，同一 QP 下 λ 偏小 2^k 倍，码率失控；`rd_cost` 把代价写成 `D + R/λ`，λ 位置反了，低码率反而被重罚；`choose_mode` 直接取 bits 最小的候选而不是 J 最小，退化为纯码率优先；`ssd` 用绝对差之和冒充平方误差和，失真与 λ 量纲不匹配，模式选择整体偏移；`mode_bits` 计算 skip 时漏掉 skip flag，skip 的代价被系统性低估而总被选中；`refine_lambda` 用两端点斜率倒数且不处理符号，λ 估计方向反了；`choose_mode` 在 J 相等时取最后一个候选，结果随输入顺序漂移；`validate_candidates` 不检查 D 与 bits 的单调关系，被高码率却更高失真的异常候选混入；`rd_batch` 全程复用第一块的 λ，第二块用错 QP 对应的 λ；`mode_bits` 让 intra 比特随块大小线性增长，大块成本被高估。难点是量纲：λ 与 QP 的指数关系、J = D + λR 的位置、skip 的隐含比特。rd/optimize.py 是核心，tests/test_optimize.py 15个红。要修的是：…

## 运行方式

在仓库根目录执行：

```powershell
python -m pytest tests -q
```

初始状态必须恰好是 `15 failed`，且不能出现 collection error、import error 或 syntax error；失败原因只能是 `rd/optimize.py` 中的函数尚未实现（`NotImplementedError`）。实现完成后，同一命令应显示 `15 passed`。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 音视频处理 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## tests 不可修改

`tests/test_optimize.py` 是评分依据，不得修改、删除、跳过或放宽断言与容差；只允许修改 `rd/optimize.py`（以及新增不改变公开契约的内部实现细节）。所有测试都只使用标准库与 pytest，离线运行，不使用随机数。
