# E1 上的 F1 / F2 分频融合实施方案

本文记录已经落地的 F1/F2 设计、实现边界和验证要求。运行代码、两个实验配置、严格 E1 初始化、训练辅助损失及统一入口均已接入；F1/F2 已各完成正式 100 epoch 训练及两种选优权重的推理评估，结果、耗时和预测诊断见 [README.md](README.md)。第 13 节保留完整编码约束，便于后续审查或复现本次修改。

## 1. 本轮固定的设计

F1 / F2 都继承 E1，只把 `fuse2` 完整替换为 `DualBandFusion`。L1 原 CSP 和 `fuse1`、L3/L4 的拼接卷积、decoder 均沿用 E1。

| 部分 | F1：差分 K/V | F2：预测残差 K/V |
|---|---|---|
| L2 输入 | 两个单源编码器的 64 通道特征 | 相同 |
| 分频 | 两个独立 LearnableFDM，核 3，组 8；动态深度卷积替代显式 unfold | 相同 |
| 低频 Q | 当前模态低频 | 相同 |
| 低频 K/V | 另一模态低频减当前低频 | 另一模态低频减单源预测值 |
| 交互 | 双向 VideoFusion 风格通道交叉注意力，4 头 | 相同 |
| 低频输出 | 两个源低频和两个注意力输出拼接，1×1 投影 | 相同 |
| 高频输出 | 两个源高频拼接，1×1 投影＋3×3 深度卷积 | 相同 |
| 频带合并 | 相加＋1×1 输出投影 | 相同 |
| 预测器 | 无 | 两个独立双尺度轻量预测器 |
| 预测损失 | 无 | 双向 MSE，只更新预测器 |
| CMA、额外 FFN、AFM、RFGM | 不添加 | 不添加 |

新模块不使用 alpha / beta / eta 等小残差缩放，不保留原 `fuse2` 并行路径，不用 `new_feature - old_feature` 构造校正量。最后直接返回新融合特征。

只保留三类有明确数学用途的减法：FDM 的 `high = input - low`、F1 的模态差分、F2 的预测残差。E1 已有 CSP 的内部机制不因本轮修改而改变。

## 2. 文件清单和复用边界

新增一个模块文件：

```text
fusion/models/dual_band_fusion.py
    LearnableFDM              保留 Fusion 核生成与参数，等价替换滤波应用后端
    ChannelCrossAttention     从 VideoFusion 精简迁移，删除内部输出投影
    SplitScalePredictor       新写：完整线性映射＋双尺度瓶颈
    DualBandFusion            新写：mode 切换 difference / prediction
```

新增两个配置文件：

```text
configs/module_screening/f1_e1_dualband_diff_l2_c0.yaml
configs/module_screening/f2_e1_dualband_pred_l2_c0.yaml
```

修改现有文件：

| 文件 | 改动 |
|---|---|
| `fusion/models/module_screening.py` | 增加继承 E1 的 F1/F2 类，替换 fuse2 |
| `fusion/models/__init__.py` | 增加两种模型注册名 |
| `train.py` | 支持严格加载 E1 时仅排除被替换的 fuse2；健康检查接入预测损失 |
| `fusion/engine.py` | 训练加入预测损失、分开梯度裁剪；验证仍只用 C0 |
| `run_module_screening.py` | 如需统一运行，增加 F1/F2 选择、独立输出目录及 E1 初始化来源 |

以下文件和保留产物不需要修改：P0/E1/E3 原配置、固定 split、P0/E1 权重和融合图、现有 SHA256 清单。

### 可以复制的内容

- `../Fusion/model/_fdafm_full.py` 的 `LearnableFDM`，当前约第 281 行：保留 GAP、filter_gen、GroupNorm、ReflectionPad、softmax、初始化及 state_dict 键；只把 unfold 后的加权聚合替换成动态深度卷积。不是逐行原样复制，但数学分频定义不变。
- `../VideoFusion-main/basicsr/archs/Transformer_util.py` 的 `Cross_Attention`：保留 Q/KV 的 1×1 和深度卷积、Q/K 空间归一化、温度参数及通道注意力，删除内部输出投影。
- 用 PyTorch reshape 替代 einops；删掉未在 forward 使用的 `self.norm`。直接返回合并多头的消息，由紧随其后的 low_fuse 完成通道混合。

删除两路 attention 输出投影的依据：它们与 low_fuse 的对应 1×1 线性映射之间没有非线性，可以合并为一个矩阵乘积 `W_low,U @ W_attn,out`。在本方案无 bias 的设定下不减少该段可表示的线性映射，但改变了参数化方式。C=64 时省去 8192 个权重和每个 L2 空间位置 8192 MAC。保留整个 DualBandFusion 最后的输出投影。

不复制整个 `_fdafm_full.py`，也不导入 VideoFusion 的 BasicSR、视频时序、配准等依赖。不复制 `Cross_TransformerBlock` 的 FFN 或外层双输入 LayerNorm。本轮 attention 直接接收当前低频和原始残差，因此 V 保留残差幅度；它是精简后的通道 attention 核心，不是完整 VideoFusion block。

复制时保留来源说明。官方参考：

- https://github.com/Linfeng-Tang/VideoFusion/blob/main/basicsr/archs/Transformer_util.py
- https://github.com/Linfeng-Tang/VideoFusion/blob/main/basicsr/archs/videofusion_Trans_arch.py

## 3. 前向结构：直接输出新特征

记 A 为可见光 L2，B 为红外 L2。

```text
A / B：两个单源编码器输出
        ↓
FDM_V / FDM_I
        ↓
A_L、A_H / B_L、B_H
        ├─────────────────────────────────────┐
        ↓                                     ↓
低频：                                      高频：
构造 E_I、E_V                              Concat(A_H, B_H)
U_A = CA(A_L, E_I)                         → 1×1：128→64
U_B = CA(B_L, E_V)                         → GELU
Concat(A_L, B_L, U_A, U_B)                 → DWConv3×3：64
→ 1×1：256→64                             → GELU
→ GELU                                   → F_H
→ F_L                                      │
        └────────────────┬────────────────────┘
                         ↓
                       F_L + F_H
                         ↓
                      1×1：64→64
                         ↓
                      新的 fuse2 输出
                         ↓
                  原 merge2 / dec2
```

为什么低频投影要读取 A_L / B_L：如果只用 `U_A = CA(A_L, E_I)`、`U_B = CA(B_L, E_V)` 作为全部低频输出，则 E 为零时，V 和 U 都为零，网络会丢失两模态共同拥有的低频结构。拼接源内容和交互消息后直接投影，可以保留这些内容，同时让注意力输出完整参与融合，无需小系数或原融合模块旁路。

两模态独立编码器已先计算完各层特征，E1 只在之后对 L1 执行 CSP。因此这里的 L2 输入尚未读取另一模态，适合定义单源预测。预测器不能改为读取 CSP 后的 L1 特征或 decoder 上下文。

### FDM：消除显式 unfold 展开，保留原分频定义

原 FDM 为每个样本、每个通道组生成一个 3×3 核，同组通道共享，核在空间位置上共享。因此可以将 batch 和通道折叠，用一次动态 depthwise conv2d 完成相同的局部加权和：

```text
kernel：[B,G,9]，沿 9 个核元素 softmax
    → 每组按通道数 C/G 重复 → weight：[B*C,1,3,3]
reflect_pad(x)：[B,C,H+2,W+2]
    → reshape：[1,B*C,H+2,W+2]
conv2d(input, weight, groups=B*C)
    → reshape：[B,C,H,W]，得到 low
high = x - low
```

conv2d 使用互相关，核的行优先顺序与原 unfold 一致，不翻转核。重复核的反向梯度自动在组内通道间求和，与原广播加权相同。不得 detach 动态核、包装为新 Parameter，或在 forward 中创建新的卷积模块。

原 patches 有 `9*B*C*H*W` 个元素，逐元素乘法还可能产生同尺寸临时张量。新实现只显式生成 `9*B*C` 个核元素、填充输入和输出。例如 B=2、C=64、H=W=128、FP32 时，一个 patches 张量约 72 MiB；复制后的核仅 4.5 KiB。算术量和参数数量不变，不是削弱滤波器的近似优化。

这项修改消除的是 Python/PyTorch 前向中显式物化的巨大 patches 张量；卷积后端仍可能使用内部 workspace，不能承诺零内部展开、整网显存固定降低九倍或必然加速。保持当前 FP32 协议，不为优化另开 AMP/TF32；浮点累加顺序变化允许合理容差，不能要求逐位一致。将来若增加 autocast，还需单独核对 dtype 提升和数值行为。

普通 NCHW 输入的折叠通常不复制；非连续或 channels-last 输入用 reshape 保证正确，可能发生一次与输入同阶的布局复制。不要无条件 contiguous、不要用每通道 Python 循环，也不新增 unfold/conv 后端开关或复杂 fallback。旧 unfold 实现仅作为临时验收参考。

API 依据：[PyTorch conv2d](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.conv2d.html)、[PyTorch Unfold](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Unfold.html)。

## 4. 建议的完整模块参考代码

下面代码块对应已新增的 `fusion/models/dual_band_fusion.py`，用于说明完整前向结构；实际运行以模块文件为准。LearnableFDM 保留原参数、初始化和数学算子，以动态深度卷积替换显式 unfold。attention 保留 VideoFusion 核心计算，删除内部输出投影，不附加 FFN、CMA 或小残差系数。

```python
from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class LearnableFDM(nn.Module):
    """
    Learnable low-pass decomposition adapted from FDIN FDM.

        X_l(p) = sum_{q in N(p)} K_x(q, p) * X(q)
        X_h(p) = X(p) - X_l(p)

    Kernel generation follows Fusion/model/_fdafm_full.py.
    Sample-folded depthwise conv replaces the explicit unfold aggregation.
    """
    def __init__(self, channels, kernel_size=3, group=8):
        super().__init__()
        if channels % group != 0:
            group = math.gcd(channels, group)
            group = max(group, 1)

        self.channels = channels
        self.kernel_size = kernel_size
        self.group = group
        self.kernel_area = kernel_size * kernel_size

        self.filter_gen = nn.Conv2d(
            channels,
            group * self.kernel_area,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.norm = nn.GroupNorm(
            num_groups=group,
            num_channels=group * self.kernel_area,
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.pad = nn.ReflectionPad2d(kernel_size // 2)

        nn.init.zeros_(self.filter_gen.weight)

    def forward(self, x):
        b, c, h, w = x.shape

        kernel = self.pool(x)
        kernel = self.filter_gen(kernel)
        kernel = self.norm(kernel)
        kernel = kernel.reshape(b, self.group, self.kernel_area)
        kernel = torch.softmax(kernel, dim=-1)
        weight = kernel.repeat_interleave(c // self.group, dim=1)
        weight = weight.reshape(
            b * c, 1, self.kernel_size, self.kernel_size
        )
        padded = self.pad(x)
        folded = padded.reshape(1, b * c, padded.shape[-2], padded.shape[-1])
        low = F.conv2d(folded, weight, groups=b * c).reshape(b, c, h, w)
        return low, x - low


class ChannelCrossAttention(nn.Module):
    """VideoFusion attention core; low_fuse absorbs its output projection."""

    def __init__(self, channels: int, heads: int = 4) -> None:
        super().__init__()
        if channels % heads:
            raise ValueError("channels must be divisible by heads")
        self.channels = channels
        self.heads = heads
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.q = nn.Conv2d(channels, channels, 1, bias=False)
        self.kv = nn.Conv2d(channels, 2 * channels, 1, bias=False)
        self.q_dwconv = nn.Conv2d(
            channels, channels, 3, padding=1, groups=channels, bias=False
        )
        self.kv_dwconv = nn.Conv2d(
            2 * channels, 2 * channels, 3,
            padding=1, groups=2 * channels, bias=False,
        )

    def forward(self, current: Tensor, message: Tensor) -> Tensor:
        b, c, h, w = current.shape
        q = self.q_dwconv(self.q(current))
        k, v = self.kv_dwconv(self.kv(message)).chunk(2, dim=1)
        shape = (b, self.heads, c // self.heads, h * w)
        q = F.normalize(q.reshape(shape), dim=-1)
        k = F.normalize(k.reshape(shape), dim=-1)
        v = v.reshape(shape)
        attention = ((q @ k.transpose(-2, -1)) * self.temperature).softmax(dim=-1)
        return (attention @ v).reshape(b, c, h, w)


class SplitScalePredictor(nn.Module):
    """Full linear channel mapping + narrow local/dilated context branch."""

    def __init__(self, channels: int, hidden: int = 32) -> None:
        super().__init__()
        if hidden <= 0 or hidden % 2:
            raise ValueError("hidden must be a positive even integer")
        self.channels = channels
        self.hidden = hidden

        # Equivalent to two separate 1x1 convolutions W_lin and W_in.
        # One combined convolution reduces one GPU convolution invocation.
        self.in_proj = nn.Conv2d(channels, channels + hidden, 1, bias=False)
        half = hidden // 2
        self.local = nn.Conv2d(half, half, 3, padding=1, groups=half, bias=False)
        self.wide = nn.Conv2d(
            half, half, 3, padding=2, dilation=2, groups=half, bias=False
        )
        self.out_proj = nn.Conv2d(hidden, channels, 1, bias=False)

        nn.init.dirac_(self.in_proj.weight[:channels])
        nn.init.zeros_(self.out_proj.weight)

    def forward(self, x: Tensor) -> Tensor:
        linear, z = self.in_proj(x).split([self.channels, self.hidden], dim=1)
        z1, z2 = z.chunk(2, dim=1)
        context = torch.cat([
            F.gelu(self.local(z1)),
            F.gelu(self.wide(z2)),
        ], dim=1)
        return linear + self.out_proj(context)


class DualBandFusion(nn.Module):
    """Direct L2 fusion output; modes differ only in low-frequency message."""

    def __init__(
        self,
        channels: int,
        mode: str = "difference",
        heads: int = 4,
        predictor_hidden: int = 32,
    ) -> None:
        super().__init__()
        if mode not in {"difference", "prediction"}:
            raise ValueError(f"Unknown low-frequency message mode: {mode}")
        self.mode = mode
        self.fdm_vis = LearnableFDM(channels, kernel_size=3, group=8)
        self.fdm_ir = LearnableFDM(channels, kernel_size=3, group=8)
        self.to_vis = ChannelCrossAttention(channels, heads=heads)
        self.to_ir = ChannelCrossAttention(channels, heads=heads)
        self.low_fuse = nn.Sequential(
            nn.Conv2d(4 * channels, channels, 1, bias=False),
            nn.GELU(),
        )
        self.high_fuse = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.GELU(),
        )
        self.out = nn.Conv2d(channels, channels, 1, bias=False)
        self._prediction_loss: Tensor | None = None

        # Construct every shared module before optional predictors.
        # Isolate predictor RNG consumption so F1/F2 retain the same data RNG.
        if mode == "prediction":
            with torch.random.fork_rng(devices=[]):
                self.pred_v_to_i = SplitScalePredictor(channels, predictor_hidden)
                self.pred_i_to_v = SplitScalePredictor(channels, predictor_hidden)

    def forward(self, vis: Tensor, ir: Tensor) -> Tensor:
        self._prediction_loss = None
        vis_low, vis_high = self.fdm_vis(vis)
        ir_low, ir_high = self.fdm_ir(ir)

        if self.mode == "prediction":
            pred_ir = self.pred_v_to_i(vis_low.detach())
            pred_vis = self.pred_i_to_v(ir_low.detach())
            if self.training and torch.is_grad_enabled():
                self._prediction_loss = (
                    F.mse_loss(pred_ir, ir_low.detach())
                    + F.mse_loss(pred_vis, vis_low.detach())
                )
            message_ir = ir_low - pred_ir.detach()
            message_vis = vis_low - pred_vis.detach()
        else:
            # Match F2's fusion-gradient routes; forward values are B-A / A-B.
            message_ir = ir_low - vis_low.detach()
            message_vis = vis_low - ir_low.detach()

        context_vis = self.to_vis(vis_low, message_ir)
        context_ir = self.to_ir(ir_low, message_vis)
        low = self.low_fuse(torch.cat([
            vis_low, ir_low, context_vis, context_ir,
        ], dim=1))
        high = self.high_fuse(torch.cat([vis_high, ir_high], dim=1))
        return self.out(low + high)

    def pop_prediction_loss(self) -> Tensor | None:
        loss = self._prediction_loss
        self._prediction_loss = None
        return loss
```

最终输出投影、attention 的 Q/KV 投影和 low/high 融合投影均正常随机初始化，不零初始化，不使用 sigmoid 限制模块强度。attention 内部已经没有输出投影。预测器非线性末层零初始化只用于定义预测起点，不会关闭整个分频模块。

### 为什么这个预测器是轻量改写

原来的 `W_lin:64→64` 和 `W_in:64→32` 都是读取同一输入的 1×1 卷积。可合并成 `64→96`，前 64 个通道是线性映射，后 32 个通道进入双尺度分支。表达能力和参数数量不变，只减少一次卷积调用；实际运行时间仍须测量。

16+16 的两条深度卷积一共只有 `32×3×3=288` 个权重，空洞卷积不会增加核参数。单方向总权重为：

```text
64×(64+32) + 16×9 + 16×9 + 32×64 = 8480
两个方向 = 16960
```

主实验固定 hidden=32。后续若需要进一步压缩，可只把 hidden 改为 16，仍保留完整 W_lin，此时单方向 6288、双方向 12576 个权重，减少约 26%。这是一项后续压缩消融，第一轮不要一边改预测机制一边改宽度。不建议先把 W_lin 改成通道分组映射，因为它会限制跨模态通道重组。

初始化时 `P(X)=X`，所以 F2 的初始预测残差等于 F1 的差分。线性映射和非线性输出投影都能训练；非线性前级首步梯度为零是末层零初始化的正常现象，末层更新后即可获得梯度。

F1 的被减项也使用 detach，这是为匹配 F2 的融合梯度路径，不改变 `B_L-A_L` 的前向数值。否则 F1 会通过差分的被减项额外更新源分支，而 F2 的预测被减项已停止梯度，两组差异就不只在 K/V 内容。该处理是本实验显式设定，不能称为完整 VideoFusion 差分 block 的逐行复刻。两组的源低频仍通过各自 Q 和 low_fuse 正常获得融合梯度。

## 5. 在现有 E1 类上挂载

在 `module_screening.py` 导入新模块，增加以下类；现有 P0/E1/E3 类保持原样。

```python
from .dual_band_fusion import DualBandFusion


class PlainE1DualBandDiffNet(PlainConditionalCSPL1ScreeningNet):
    kv_mode = "difference"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        channels = self.fuse2.op[0].out_channels
        self.fuse2 = DualBandFusion(channels, mode=self.kv_mode)

    def pop_prediction_loss(self):
        return self.fuse2.pop_prediction_loss()


class PlainE1DualBandPredNet(PlainE1DualBandDiffNet):
    kv_mode = "prediction"
```

注册：

```python
MODEL_REGISTRY.update({
    "plain_e1_dualband_diff_l2": PlainE1DualBandDiffNet,
    "plain_e1_dualband_pred_l2": PlainE1DualBandPredNet,
})
```

`plain_fusion.py` 的 forward 已经调用 `self.fuse2(a2, b2)`，不需要改它。两个模型仍然对原 L1 执行 CSP；不要额外给 FDM/attention 读取 CSP 的统计、门控或 decoder 信息。

## 6. E1 初始化：仅允许替换 fuse2

当前 `initialize_e1_base` 要求 E1 全部键都存在，并不适合替换原 fuse2。对这两个注册名增加一个明确的替换分支，其他模型仍遵守原加载规则。

允许从源 checkpoint 排除的键只应是原 `fuse2.op.*`，现有 E1 对应：

```text
fuse2.op.0.weight
fuse2.op.0.bias
fuse2.op.2.weight
fuse2.op.2.bias
```

加载顺序：

1. 校验来源确实是 E1；主干通道、编码深度、heads、expansion 相同。
2. 用实际 E1 类的 state_dict 校验来源的完整键集合，防止缺少主干/CSP 参数的 checkpoint 被接受。如需临时实例化源 E1，同样在 CPU fork_rng 中执行，不污染数据随机状态。
3. 排除上面的四个旧 fuse2 键，逐个校验并复制其余键和形状。
4. 目标模型相对源主干新增的所有键必须位于 `fuse2.` 内。
5. 将复制完的目标完整 state_dict 用 `strict=True` 加载。
6. 记录源 checkpoint 路径、SHA256、epoch、加载键数、排除的四个键及新增键。

参考关键逻辑如下；本段是现有初始化函数内部的替换分支，需要沿用其来源/config/hash 检查：

```python
allowed_models = {"plain_e1_dualband_diff_l2", "plain_e1_dualband_pred_l2"}
if cfg["model"] not in allowed_models:
    raise RuntimeError("This replacement rule is only for F1/F2")

expected_old = {
    "fuse2.op.0.weight", "fuse2.op.0.bias",
    "fuse2.op.2.weight", "fuse2.op.2.bias",
}
dropped = {key for key in source if key.startswith("fuse2.op.")}
if dropped != expected_old:
    raise RuntimeError("Unexpected source fuse2 keys")

target = model.state_dict()
shared = set(source) - dropped
if not shared.issubset(target):
    raise RuntimeError("Target does not preserve the complete E1 base")
new_keys = set(target) - shared
if not new_keys or any(not key.startswith("fuse2.") for key in new_keys):
    raise RuntimeError("Unexpected modifications outside fuse2")
if any(key in target for key in dropped):
    raise RuntimeError("Old fuse2 parameters should not remain as unused modules")

for key in shared:
    if target[key].shape != source[key].shape:
        raise RuntimeError(f"E1 base shape mismatch: {key}")
    target[key] = source[key].to(target[key])
model.load_state_dict(target, strict=True)
```

不要用全局 `strict=False` 静默忽略不匹配。新 fuse2 没有继承原 fuse2 权重，也不能声称新模型初始输出等于 E1。

### 保证 F1 / F2 公共部分完全一致

- 使用相同 seed，在 CPU 构建模型，再迁移到 GPU。
- 先构造所有公共层，再构造 F2 的预测器。上面的 fork_rng 只隔离预测器构造，不改变全局数据随机数状态。
- 两组从同一份 E1 best_metrics 加载公共参数；预测器的 W_lin 是恒等、W_out 是零。
- 对共享 state_dict 的每个 Tensor 执行完全相等检查，并记录公共新 fuse2 的初始化哈希。
- 在相同输入上验证 F1/F2 初始输出一致，允许设备浮点误差；差分和预测残差也应一致。
- 数据加载前固定相同随机状态，保持 shuffle/增强序列一致。不要因为 F2 多构造预测器而改变数据顺序。

## 7. 预测损失和训练器接入

预测器只读取单源低频：

```text
pred_ir  = P_vis_to_ir(vis_low.detach())
pred_vis = P_ir_to_vis(ir_low.detach())

L_pred = mean((pred_ir  - ir_low.detach())²)
       + mean((pred_vis - vis_low.detach())²)

message_ir  = ir_low  - pred_ir.detach()
message_vis = vis_low - pred_vis.detach()
```

含义：预测损失不更新 encoder、FDM 或 attention；融合损失不更新预测器。融合损失仍可以通过真实低频特征和 Q 更新 encoder/FDM，这是正常的任务学习路径。

主实验先固定 `L_train_total = L_C0 + L_pred`，MSE 使用 mean reduction，不增加额外的小预测权重或不确定性门控。由于两条梯度路径已分开，预测损失不会直接与主干的融合梯度竞争。预测器和主模型使用同一个优化器/学习率调度器即可。

训练循环在模型 forward 和 C0 loss 之后立即取走辅助损失：

```python
fused = model(vis, ir)
losses = criterion(fused, vis, ir)
fusion_loss = losses["loss"]
pop_aux = getattr(model, "pop_prediction_loss", None)
prediction_loss = pop_aux() if callable(pop_aux) else None
if prediction_loss is None:
    prediction_loss = fusion_loss.new_zeros(())

total_loss = fusion_loss + prediction_loss
if not torch.isfinite(total_loss):
    raise FloatingPointError("Non-finite fusion/prediction loss")
(total_loss / grad_accum_steps).backward()
```

日志分开记录 `loss`（原 C0）、`prediction_loss`、`total_loss`；不要把 `total_loss` 覆盖到验证选权重的 `loss` 上。辅助损失每次 forward 重置，并在读取后清空，避免跨 batch 保留计算图。

### 梯度裁剪要分开

当前训练器对 model.parameters 整体裁剪。如果 F2 的预测梯度也参与同一总范数裁剪，就会间接改变主干的裁剪系数，即使使用 detach 也不能说更新完全隔离。因此在循环外把参数分为两组：

```python
predictor_prefixes = ("fuse2.pred_v_to_i.", "fuse2.pred_i_to_v.")
fusion_parameters = []
predictor_parameters = []
for name, parameter in model.named_parameters():
    if not parameter.requires_grad:
        continue
    group = predictor_parameters if name.startswith(predictor_prefixes) else fusion_parameters
    group.append(parameter)
```

每次累积更新前仍检查所有梯度有限。主融合参数沿用公共协议 `grad_clip_norm=0.3`，预测器参数另外裁剪到 0.3；这是梯度稳定措施，不是前向小残差缩放。两组仍由同一个 optimizer.step 更新。

```python
if grad_clip_norm is not None and grad_clip_norm > 0:
    torch.nn.utils.clip_grad_norm_(fusion_parameters, grad_clip_norm)
    if predictor_parameters:
        torch.nn.utils.clip_grad_norm_(predictor_parameters, grad_clip_norm)
```

`check_training_health` 也应调用相同的辅助损失读取逻辑，检查预测器存在非零梯度后再清空；保留它原来的 RNG 恢复和输出 head 梯度检查。不要只检查融合输出而漏掉预测器的训练连接。

### 验证、诊断和保存

- `evaluate_loss` 继续使用 eval/no_grad，只算 C0。F2 在 eval 时仍进行单源预测以生成消息，但不建立预测损失计算图。
- `best_loss.pth` 按 C0 验证损失选择，不能按 C0+预测损失选择。
- `best_metrics.pth` 沿用现有 SD/QABF/SSIM/VIF 验证排名及并列规则。
- 现有输出梯度诊断仍测 C0；额外预测 MSE 只作为独立日志。
- F2 checkpoint 中必须包含预测器的模型参数和优化器状态；resume 不重新执行 E1 初始化。
- 保持当前指标评估、YCrCb 恢复、uint8 保存、原分辨率 padding/crop 与 361 对校验流程。

## 8. 配置与运行入口

两份 YAML 都完整复制 `e1_conditional_csp_l1_c0.yaml` 的公共协议，分别只把 model 改为：

```yaml
# f1_e1_dualband_diff_l2_c0.yaml
model: plain_e1_dualband_diff_l2
```

```yaml
# f2_e1_dualband_pred_l2_c0.yaml
model: plain_e1_dualband_pred_l2
```

不修改原 C0 loss_kwargs；F2 预测损失在模型/训练器中单独处理。hidden=32、heads=4、FDM kernel=3/group=8 作为两个类的固定实验定义。这样现有 comparable_config 仍可检查其余训练协议相同。

统一运行器如要扩展，应继续使用现有 `run_module_screening.py`，而非复制整份 pipeline。具体需要：

1. 新增可选 `--experiments F1 F2`；未指定时维持原 P0/E1/E3 行为，`--stage verify` 仍检查原保留资产。
2. 新实验输出到 `runs/e1_dualband_l2_c0_seed2026/{F1,F2}/`，评估和汇总放在同一个新根目录，避免覆盖原结果。
3. 对 F1/F2 使用 `--init-e1` 加载 E1 best_metrics，不能沿用 train_all 中写死的 INITIAL（P0 初始权重）。
4. 完整训练校验对 F1/F2 检查 `e1_base_init.json`，不能要求它们提供 P0 初始化的 `backbone_init.json`。
5. 完整记录源 E1 的 SHA256、被替换 fuse2 的键及新 fuse2 公共初始化哈希。
6. 100 epoch 完成后，对两类最佳权重推理和评估；显式读取保留的 P0/E1 结果作为参考，不能仍依赖本轮先遍历 P0 填充 baseline 字典。P0/E1/E3 无需重训。
7. 输出 validation_comparison.csv / comparison.csv 及 all/day/night，断点重跑从各自 weights/latest.pth 恢复；最终完成标志是 checkpoint.pt 且 epoch=100。
8. skip-existing 只跳过经完整校验的已完成阶段：核对配置、源 E1 哈希、split 名单与哈希、checkpoint 哈希、推理评估上下文及 361 对文件完整性。未完成的训练恢复执行，不静默跳过。

训练、推理和评估 F1/F2 的统一命令：

```powershell
Set-Location E:\code\fusion_clean
python run_module_screening.py --stage all --device cuda --experiments F1 F2 --skip-existing
```

### 训练预算的公平性

两组均从完整 100 epoch 的 E1 实验中、由验证集选出的同一个 best_metrics.pth 开始，再训练相同的 100 epoch。源权重选中 epoch 未必是 100。两组比较检验“受监督预测器＋预测残差消息”相对差分的收益，不能仅凭 F2 胜出排除新增参数的贡献。和历史 E1 主结果的比较还包含了额外训练收益。

如果要把相对 E1 的涨点归因于新融合结构，应另做 E1 同预算继续训练对照，使用原 E1 YAML、新输出目录和 `--init-weights E1/best_metrics.pth`。或者最终统一从同一原始初始化训练 E1/F1/F2。不要混淆“继承 E1 结构”和“从训练过的 E1 权重继续训练”。

## 9. 计算预算

以下为默认 C=64、heads=4、预测 hidden=32、卷积 bias=False 的参考。N 表示 L2 的 H2×W2，不是输入 batch size。不包含 norm/激活/softmax、反向传播或诊断开销。

| 项目 | 参考值 |
|---|---:|
| 原 ConcatFuse2 卷积权重（不含 bias） | 45,056 |
| 双向通道 attention 的主要卷积权重 | 28,032，已删除两路输出投影 |
| 双向预测器新增权重 | 16,960 |
| 两个 FDM filter_gen 权重 | 9,216，GAP 后每图执行 |
| F1 新 fuse2 总参数 | 66,792，含 FDM GN affine 和 attention temperature |
| F2 新 fuse2 总参数 | 83,752 |
| F1 局部前向 MAC | 约 62,528N + 9,216/图 |
| F2 局部前向 MAC | 约 79,488N + 9,216/图 |
| 原 ConcatFuse2 前向 MAC | 45,056N |

上述 MAC 包括双向 attention 的 QK/AV 矩阵乘法和两个 FDM 的 3×3 加权和，不包含 GAP、归一化、激活、softmax、反向或内存搬运。N 表示单张图像的 L2 空间大小，batch 总量需乘 batch size。

低频投影为 4C→C，保留源内容和直接 attention 输出；删除两路 attention 内部投影省去 8192 个权重和 8192N MAC。FDM 改成动态深度卷积不改变算术量和参数量。

F1/F2 的 fuse2 局部 MAC 分别比原 fuse2 约高 38.8%/76.4%，不是整网训练时间同比增加。不能把“通道 attention 比空间 attention 省”说成“新融合比原拼接卷积更快”。显式 unfold 已被替换，完整网络的实际显存、耗时和卷积 workspace 仍须测量；本机单 FDM 结果见第 12 节。

先仅替换 L2、仅一层双向 attention、仅两条小预测器；不增加 FFN/CMA/多层交互。记录普通 epoch 的训练时间和包含验证的端到端时间，分开报告。不要在每个 batch 保存全分辨率 attention/特征图或调用多次 GPU→CPU 同步。

## 10. 实施后的必要验证

1. FDM 迁移一致性：对相同输入、相同参数，比较原 unfold 与动态深度卷积的 low/high、输入梯度、filter_gen 和 GN affine 参数梯度；检查 low+high=input。包含 batch>1、随机非零 filter_gen/GN affine、非方形、非连续和 channels-last 输入；不能只测初始化的均匀核。
2. attention 一致性：和 VideoFusion 核心共享 Q/KV/DW/temperature，将官方 project_out 设为恒等映射后比较；新模块已无此层。不与带 FFN/LN 的完整 block 混比。
3. F1/F2 起点：共享参数逐个相等，预测 P(X)=X，两个新模型对随机/常量输入的初始输出一致。
4. 形状：64 通道 L2 输入输出保持 H2/W2；整网含非 8 倍数图像时 padding/crop 输出仍正确。
5. 共有内容：相等低频且消息为零时，context 为零，但 low_fuse 仍能读取 A_L/B_L，不存在结构上必然丢失共享低频的问题。
6. 梯度隔离：仅对预测损失 backward 时，encoder/FDM/attention 无预测损失梯度；仅 C0 backward 时预测器无梯度；合并训练时两个预测器获得有效梯度。
7. 注意预测器初始 W_out=0 时，非线性前级首步无梯度正常；检查 W_lin/W_out 首步有效、更新后前级可学习。
8. 初始化白名单：只允许丢弃四个旧 fuse2 参数；篡改任何 encoder/CSP/decoder 键应明确失败。
9. resume：模型和优化器中预测器状态可恢复，公共初始化不会覆盖 latest；固定数据 split 名单和哈希一致。
10. 原资产：实施结束仍运行 `python run_module_screening.py --stage verify --device cpu` 校验原 P0/E1 保留文件未变。
11. 成本：固定 batch、分辨率、dtype、TF32/AMP 设置和损失，分别测单 FDM 和完整 F1/F2 的前向＋反向耗时、峰值 allocated/reserved 显存。同形状预热、计时前后同步；不能只报推理内存或据理论张量大小推算整网节省。

## 11. 论文解释范围

FDM 是可学习局部平滑＋残差分解，不等同于理想频带滤波或 FFT 截断。直接差分是模态差异线索；预测残差是在目标模态表示坐标中、当前预测器未解释的部分。

预测残差仍包含预测器能力不足、噪声和配准误差。第一轮不增加不确定性/可靠性门控，也不因此宣称已区分所有有效残差与噪声。

本轮先比较 F1/F2。若 F2 有效，后续再加入无预测监督但相同预测器结构的容量对照，以及 hidden=16 的压缩实验。

容量对照必须单独允许 C0 融合损失更新预测器：保留预测器输入 detach，取消预测结果进入消息前的 detach，不使用预测 MSE。若仅关闭 MSE、仍保留主方案的全部 detach，预测器会一直保持恒等映射，该对照退化成 F1，不能检验新增容量。这个容量对照属于后续第三种配置，不改变本轮 F1/F2 的设定。

新融合最后直接输出特征，没有旧 fuse2 的小残差校正路径。

## 12. 实现与检查状态

运行代码已经落地，相关 Python 文件通过编译检查。LearnableFDM 的初始化语法树和 state_dict 键保持原实现；forward 已改为动态深度卷积，不能再称整个类的语法树相同。

CPU float64 检查包含随机非均匀核、GN affine、batch>1、非方形、非连续及 channels-last 输入，最大绝对差：low/high 约 1.11e-15，输入梯度约 8.67e-19，参数梯度约 4.16e-17。另已确认精简 attention 与官方恒等输出投影版本一致，F1/F2 公共初始化和 RNG 状态一致，初始预测 P(X)=X、融合输出与输入梯度一致，预测/融合梯度隔离及 eval 缓存清理正确。F1/F2 fuse2 参数量为 66792/83752，每个预测器 8480。

CUDA FP32、TF32 关闭的随机非均匀核检查同样通过，low/high 最大绝对差约 4.77e-7，输入梯度约 1.46e-11，参数梯度约 5.24e-10。本次核对的 28 个现有 Python/YAML 与主 README 的 SHA256 均未变化。

落地后执行了项目全部 17 个测试函数，包括 P0/E1/E3 回归、FDM 输出与梯度等价、F1/F2 初始输出一致、预测梯度隔离和严格 E1 fuse2 替换，全部通过。用真实保留 E1 best_metrics.pth 初始化时，F1/F2 均加载 573 个公共键，共享新 fuse2 初始化哈希一致；F2 单批训练步能够分别记录 C0、预测损失和总损失，并更新两个预测器。保留清单中的 1543 个 P0/E1 权重、记录及融合图哈希全部通过，统一入口的只读 verify 阶段通过。

### 单 FDM 的 CUDA 短测

环境：RTX 4090 D，PyTorch 2.5.1+cu121，FP32，TF32 关闭。B=2、C=64、G=8，前向＋反向，预热 3 次、测量 5 次；两实现使用相同参数和损失定义。下表是本进程峰值 allocated，包含该测试输入、梯度及相关分配，不是整卡显存或完整融合网络。

| L2 大小 | 实现 | 峰值 allocated / MiB | 前向＋反向 / ms |
|---|---|---:|---:|
| 128×128 | 原 unfold | 256.027 | 1.134 |
| 128×128 | 动态深度卷积 | 72.279 | 0.492 |
| 240×320 | 原 unfold | 1202.028 | 5.536 |
| 240×320 | 动态深度卷积 | 342.076 | 1.916 |

这次局部短测的峰值 allocated 降低约 72%；不据此承诺整网同比节省或正式训练加速比例。长时稳定性、整网加载、原分辨率整网显存/耗时、断点恢复和实际评估仍须在实施后验证。以上检查不能替代真实实验结果。

## 13. 可直接复制的完整编码提示词

以下提示词作为本次实现的完整约束和代码审查清单保留。它覆盖全部固定设计，并明确替换早期文案中的“保留 unfold”和“保留 attention 内部输出投影”。

```text
请实际修改 E:\code\fusion_clean，实现 E1 基础上的 F1/F2 分频融合实验。
先阅读适用的 AGENTS.md、README_F1_F2_IMPLEMENTATION.md，以及当前
模型、初始化、训练、推理、评估和 run_module_screening.py。复用原流程。

实现必须清晰简洁：只新增一个模块文件和两个实验 YAML，必要时小改现有
接入文件。不复制网络 forward、训练器、推理器或评估流水线，不增加新依赖、
通用插件系统、复杂配置层或大量辅助类。模块约200行为目标，清晰优先。
不重构无关代码，不改 P0/E1/E3 现有行为、配置、split、保留权重与融合图。

一、文件与固定结构

新增 fusion/models/dual_band_fusion.py，仅包含四个主要类：
LearnableFDM、ChannelCrossAttention、SplitScalePredictor、DualBandFusion。
新增 configs/module_screening/ 下：
f1_e1_dualband_diff_l2_c0.yaml
f2_e1_dualband_pred_l2_c0.yaml
必要的小改动仅涉及 module_screening.py、models/__init__.py、train.py、
fusion/engine.py、run_module_screening.py 及相关说明文档。

两组继承 E1，仅完整替换 self.fuse2；不要复制 PlainFusionNet.forward。
L1 原 CSP/fuse1、L3/L4、decoder、padding/crop 和输出处理保持原样。
固定 L2 C=64、attention heads=4、predictor hidden=32、FDM kernel=3/group=8。
不增加 RFGM、CMA、独立 FFN、AFM、空间门控或多层新模块。

二、LearnableFDM：保留数学定义，解决显式 unfold 内存展开

来源 E:\code\Fusion\model\_fdafm_full.py 的 LearnableFDM。
保留来源注释、原参数名、初始化、GAP、filter_gen、GroupNorm、softmax、
ReflectionPad 和 state_dict 键。两模态各一个独立实例，参数不共享。
只替换滤波应用后端，不保留运行时 unfold，不改成固定核或近似滤波。

forward 的核心计算必须为：
kernel = self.norm(self.filter_gen(self.pool(x)))
kernel = kernel.reshape(B, G, 9).softmax(dim=-1)
weight = kernel.repeat_interleave(C // G, dim=1).reshape(B*C, 1, 3, 3)
padded = self.pad(x)
folded = padded.reshape(1, B*C, padded.shape[-2], padded.shape[-1])
low = F.conv2d(folded, weight, groups=B*C).reshape(B, C, H, W)
return low, x-low

按实际 self.kernel_size/kernel_area 实现，不引入额外参数。
每个样本、每组的核在空间共享，按原通道组顺序复制。
conv2d 使用互相关，不翻转核；repeat_interleave 的梯度保持组内求和。
不能 detach 动态核、包装成新 Parameter 或在 forward 创建 nn.Conv2d。
使用 reshape，不无条件 contiguous，不写逐样本/逐通道卷积循环。
不新增后端选择开关、自定义 CUDA/Triton 或复杂 fallback。
旧 unfold 只作临时验收参考，不进入运行模块。
这消除显式9倍patches，不保证卷积库没有内部workspace或整网固定节省比例。
保持当前FP32协议，不为本轮优化新增AMP、TF32或其他训练精度变化。

三、ChannelCrossAttention：移除冗余输出投影

参考 E:\code\VideoFusion-main\basicsr\archs\Transformer_util.py
及 VideoFusion 官方 Cross_Attention，仅迁移核心计算并保留来源说明。
Q：当前模态低频 → 1×1(C,C) → DWConv3×3(C,C)。
KV：消息 → 1×1(C,2C) → DWConv3×3(2C,2C) → split(K,V)。
卷积均 bias=False、DW padding=1，固定4头。
reshape 为 [B,heads,C/heads,H*W]，Q/K 沿最后一维 L2 normalize。
每头 temperature 初始化为1；attention=softmax((Q@K^T)*temperature)。
V不归一化，直接返回 reshape(attention@V)，形状[B,C,H,W]。

删除未使用的 norm 和内部 project_out；由后面的 low_fuse 统一投影。
保留 DualBandFusion 最后的输出投影。
不复制外层 Transformer block、LayerNorm、FFN、CMA 或小残差缩放。
用原生PyTorch实现，不新增einops依赖。
双向 attention 参数独立，读取原始低频并行计算，不串行更新两模态。

四、SplitScalePredictor：双方向独立的单源预测

输入/输出64通道，hidden=32，全部卷积bias=False。
in_proj：1×1，64→96，一次计算后切分 M(64) 和 Z(32)。
M对应完整线性通道映射，Z再切分16+16。
分支1：DWConv3×3，dilation=1，padding=1 → GELU。
分支2：DWConv3×3，dilation=2，padding=2 → GELU。
拼接两分支 → out_proj：1×1，32→64。
输出 P(X)=M+out_proj(context)。
in_proj前64个输出初始化为恒等映射，out_proj权重初始化为零。
其余正常初始化，保证初始P(X)=X；每方向8480个权重。
不增加BN、归一化、门控、注意力或小残差系数。
预测器只读对应单源低频，不能读对方特征、融合特征、CSP双源交互结果
或decoder上下文。两方向预测器不共享参数。

五、DualBandFusion：一个实现、两个 mode

mode仅支持difference/prediction，共用全部分频、attention和融合代码。
先构造所有公共模块，再构造F2的两个预测器。
公共模块：两个FDM、两个CA、low_fuse、high_fuse、最终out。
low_fuse：Conv1×1(4C,C) → GELU。
high_fuse：Conv1×1(2C,C) → GELU → DWConv3×3(C,C,padding=1) → GELU。
out：Conv1×1(C,C)。新增卷积均bias=False。

A_L,A_H=fdm_vis(A)；B_L,B_H=fdm_ir(B)。
F1：E_I=B_L-A_L.detach()；E_V=A_L-B_L.detach()。
F2：pred_I=P_VI(A_L.detach())；pred_V=P_IV(B_L.detach())；
    E_I=B_L-pred_I.detach()；E_V=A_L-pred_V.detach()。
F1被减项detach用于匹配F2梯度路径，不能擅自删掉。

共同计算：
U_A=CA_V(A_L,E_I)；U_B=CA_I(B_L,E_V)。
F_L=low_fuse(cat(A_L,B_L,U_A,U_B))。
F_H=high_fuse(cat(A_H,B_H))。
return out(F_L+F_H)。

直接输出新融合特征，不保留旧fuse2并行路径，不计算A_L+alpha*U_A，
不使用new_feature-old_feature校正，不增加alpha/beta/eta小残差缩放。
attention、low/high和最终输出正常初始化，不零初始化整个交互路径。
仅预测器非线性末层零初始化以定义恒等预测起点；原CSP机制保持原样。

六、预测损失、缓存和训练接入

F2的L_pred=MSE(pred_I,B_L.detach())+MSE(pred_V,A_L.detach())，均mean。
F1总损失=原C0；F2总损失=C0+L_pred，预测损失系数固定1。
C0保持8*intensity+10*gradient+1*SSIM+12*color。
预测损失只训练预测器；融合损失不训练预测器。
不能删除预测输入、监督目标或预测值进入消息前的detach。

提供简单pop_prediction_loss()：返回并清空本次forward缓存。
每次forward先清旧缓存，仅training且grad_enabled时缓存L_pred。
eval仍运行预测器构造消息，但不缓存辅助损失计算图。
训练器在forward后立即消费一次缓存，不跨batch持有计算图。
日志分别记录C0、prediction_loss、total_loss；累积时对总损失除累积步数。

主网络和预测器分别按0.3裁剪梯度，不能一起做总范数裁剪。
复用一个optimizer/scheduler，沿用原梯度有限检查和更新流程。
健康检查消费预测损失并检查有效梯度，保留原RNG恢复和head梯度检查。
验证best_loss只使用C0；best_metrics使用原验证排名与并列规则。
checkpoint保存完整预测器和optimizer状态；resume时不重新执行E1初始化。

七、模型注册、严格初始化和相同起点

注册plain_e1_dualband_diff_l2 / plain_e1_dualband_pred_l2。
两个小型E1子类通过mode共用实现，替换self.fuse2即可。
两份YAML完整复制现有E1公共协议，只改变model名。

两组从同一个E1 best_metrics.pth加载，扩展现有initialize_e1_base。
源E1仅允许排除：
fuse2.op.0.weight、fuse2.op.0.bias、
fuse2.op.2.weight、fuse2.op.2.bias。
验证源checkpoint是完整、结构匹配的E1，其他键必须完整且形状一致。
目标新增键只允许位于fuse2.，不得残留旧fuse2参数。
组装完整目标state_dict后strict=True加载，不用笼统strict=False。
替换白名单只适用于F1/F2，原模型保留原加载规则。
记录来源路径、SHA256、源epoch、排除键、新增键和公共新模块初始化哈希。

在CPU构造后再移动GPU；F2预测器最后在fork_rng(devices=[])内构造。
如校验源完整性需临时构造E1，同样隔离RNG。
相同seed的公共参数逐个完全相等，数据shuffle/增强随机起点一致。
不得跨实验共享Parameter对象。初始P(X)=X，F1/F2初始消息、输出和
融合输入梯度应一致；不声称新模型初始输出等于旧E1。

八、统一训练、推理、评估和恢复

扩展现有run_module_screening.py支持--experiments F1 F2，
不指定时保留原默认行为及原--stage verify资产校验。
输出根目录runs/e1_dualband_l2_c0_seed2026/{F1,F2}/。
F1/F2使用--init-e1加载源E1，不能沿用P0 INITIAL入口；
完整性记录使用e1_base_init.json，而不是backbone_init.json。

实施后支持一个命令：
python run_module_screening.py --stage all --device cuda --experiments F1 F2 --skip-existing

仅训练F1/F2，各完整100epoch，然后对best_metrics.pth和best_loss.pth
分别自动推理评估。显式读取保留P0/E1结果作为参考，不重训P0/E1/E3，
也不能依赖本轮先运行P0来建立baseline字典。
必要时复用原评估器从保留融合图重建缺失参考统计，不改变评估协议。

数据固定：同一train_tensor、split文件名单与哈希；训练975、验证108；
MSRS测试361对，day179/night182。完整沿用E1预处理及数据增强。
训练固定seed2026、100epoch、batch2、累积4、lr8e-5、min_lr2e-7、
weight_decay1e-4、warmup8；其他设置均复制E1。
每10epoch原分辨率验证，排名权重SD0.1/QABF0.3/SSIM0.3/VIF0.3，
并列规则沿用原实现。推理归一化、padding/crop、YCrCb恢复、uint8保存、
输入输出配对校验和all/day/night评估全部复用现有流程。
输出validation_comparison.csv、comparison.csv和可追溯上下文。

中断从weights/latest.pth恢复；最终完成标志checkpoint.pt且epoch=100。
skip-existing只跳过验证完整的阶段；核对配置、源E1 hash、split名单/hash、
checkpoint hash、推理评估上下文及361对文件，未完成阶段继续执行。
不为显存改batch、分辨率、累积或精度，否则已改变公共协议。

九、验收与交付

先做小范围验收，不自动启动完整训练，不为此添加庞大测试框架。
FDM：随机非零filter_gen/GN affine、batch>1、非方形/非连续输入，
对比原unfold与新conv的low/high、输入梯度、全部核生成参数梯度；
验证low+high=x。CPU float64先验证，CUDA FP32记录合理容差和TF32设置。
不能只测初始化均匀核，不要求跨实现浮点逐位相同。
CA：官方project_out设恒等，复制Q/KV/DW/temperature后结果一致。
检查公共初始化/RNG、初始F1/F2输出和输入梯度、预测梯度隔离、
预测器首步与更新后的学习连接、缓存清理、整网奇数尺寸、严格加载失败
分支、checkpoint/resume及原P0/E1资产SHA256。

同batch/分辨率/dtype/损失，预热并同步计时，测单FDM及完整F1/F2的
前向+反向时间、峰值allocated/reserved显存；不要用局部结果代替整网。
参考参数量：每方向predictor8480，F1 fuse2=66792，F2 fuse2=83752。
不编造指标、训练耗时或加速比例，不在每batch保存大特征或频繁CPU同步。
交付修改文件、检查结果、实际命令及性能测量，保持实现简洁。

两组的源权重来自完整100epoch E1实验的验证选择，选中epoch未必为100，
再各训练100epoch。对历史E1的增益包含额外训练收益；结构归因需后续
E1同预算继续训练对照，本轮不自动增加第三个实验。
```
