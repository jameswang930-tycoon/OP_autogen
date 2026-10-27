# 算子评估记录模板

每次评估由同一硬件、软件版本、shape、dtype 和计时方法下的验证结果填写。
本文件只定义记录格式；运行结果写入 outputs/<op>/final_output/。

| 算子 | shape | dtype | 正确性 | 基线耗时 | 候选耗时 | 加速比 | 计时方法 | 运行位置 |
|---|---|---|---|---|---|---|---|---|
| <operator> | <shape> | <dtype> | <pass/fail> | <baseline> | <candidate> | <ratio> | <method> | <run-dir> |

加速比 = 同口径基线耗时 / 候选耗时。
正确性未通过的候选不参与性能比较；不同计时方法的结果分别记录。
