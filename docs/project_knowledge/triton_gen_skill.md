# triton-gen Skill

triton-gen skill（`.claude/commands/triton-gen.md`）支持 5 种输入类型：

| 输入类型 | 检测规则 | 处理路径 |
|----------|---------|---------|
| 自然语言 | 纯文本描述算子 | Step 2a: 直接分析语义 |
| PyTorch 模型 | `.pt`/`.pth`、`nn.Module`、`torch.nn` | Step 2b: 提取算子语义 + 形状 |
| ONNX 模型 | `.onnx`、`onnxruntime`、`onnx.` | Step 2c: 解析计算图 |
| 基线 Triton kernel | `@triton.jit`、`import triton`、`tl.program_id` | Step 2d: 转为 emulator 兼容形式 |
| 固定 Shape | 模型名或 `[B,C,H,W]` 形状标注 | Step 2e: 查询 shapes_registry |

多种输入类型可共存，显式 shape 优先。

## 关键文件

- `.claude/commands/triton-gen.md` — skill 定义（~100 行）
- `models/shapes_registry.py` — 固定 shape 注册表（resnet18/34/50、bert-base、gpt2）
