import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import swin_v2_t, Swin_V2_T_Weights


# ============================================================
#  LoRA adapter
# ============================================================

class LoRALinear(nn.Module):
    """LoRA for nn.Linear. W_eff = W + (alpha/r) * BA.

    Uses @property weight/bias so torchvision's Swin attention
    (which calls F.linear(x, self.qkv.weight, self.qkv.bias))
    automatically gets the LoRA-adapted weight.
    """

    def __init__(self, linear: nn.Linear, r: int = 16, lora_alpha: int = 16):
        super().__init__()
        in_f, out_f = linear.in_features, linear.out_features

        self.linear = linear
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

        self.lora_A = nn.Parameter(torch.zeros(r, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r))
        self.scaling = lora_alpha / r

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    @property
    def weight(self):
        delta = (self.lora_B @ self.lora_A) * self.scaling
        return self.linear.weight + delta

    @property
    def bias(self):
        return self.linear.bias

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


# ============================================================
#  Lightweight Counting Head — no BN, no SE, no dilated convs
# ============================================================

class LightCountingHead(nn.Module):
    """Minimal head for crowd counting.

    Design principles (from user feedback):
    - No BN (unstable with batch=4)
    - No SE (counting needs spatial patterns, not channel re-weighting)
    - No dilated convs (texture/repetition patterns are local)
    - Simple merge: project stage2+stage3 to 128ch, add, refine.
    - Output: (B, 1, H/8, W/8) non-negative density map.
    """

    def __init__(self, in2: int = 192, in3: int = 384, mid: int = 128):
        super().__init__()
        self.proj2 = nn.Conv2d(in2, mid, 1)
        self.proj3 = nn.Conv2d(in3, mid, 1)

        self.refine = nn.Sequential(
            nn.Conv2d(mid, mid, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.ReLU(inplace=True),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, feats):
        f2 = self.proj2(feats['stage2'])   # H/8, 128
        f3 = self.proj3(feats['stage3'])   # H/16, 128
        f3 = F.interpolate(f3, size=f2.shape[2:], mode='bilinear',
                           align_corners=False)
        merged = f2 + f3
        return self.refine(merged)          # (B, 1, H/8, W/8)


# ============================================================
#  SwinCount — SwinV2-T + LoRA + LightCountingHead
# ============================================================

class SwinCount(nn.Module):
    """SwinV2-T backbone + LoRA + Lightweight Counting Head.

    Strategy (from user feedback):
    - Shallow layers (stage1-2): frozen, adapted via LoRA.
    - Deep layers (stage3-4): unfrozen (full SFT) + LoRA.
    - LoRA applied to qkv, proj, fc1, fc2 in every Swin block.
    - Light head: no BN, no SE, no dilated convs.
    """

    _TARGET_CHANNELS = {96: 'stage1', 192: 'stage2',
                        384: 'stage3', 768: 'stage4'}

    # Shallow indices (frozen): 0=PatchEmbed+Stage1, 1=Stage1extra,
    #                           2=PatchMerge, 3=Stage2
    _SHALLOW_END = 4   # features[0:4] frozen, features[4:8] unfrozen

    def __init__(self, pretrained: bool = True, lora_r: int = 16,
                 lora_alpha: int = 16):
        super().__init__()

        weights = Swin_V2_T_Weights.IMAGENET1K_V1 if pretrained else None
        swin = swin_v2_t(weights=weights)
        self.features = swin.features

        # 1. Apply LoRA to ALL Swin blocks (qkv, proj, fc1, fc2)
        self._apply_lora_all(lora_r, lora_alpha)

        # 2. Freeze shallow; unfreeze deep (indices 4-7)
        self._set_freeze_policy()

        # 3. Lightweight head
        self.head = LightCountingHead(in2=192, in3=384, mid=128)

    def _apply_lora_all(self, r, alpha):
        """Apply LoRA to qkv, proj, fc1, fc2 in every Swin block."""
        attn_count = 0
        mlp_count = 0
        for module in self.features.modules():
            if module.__class__.__name__ == 'ShiftedWindowAttentionV2':
                module.qkv = LoRALinear(module.qkv, r=r, lora_alpha=alpha)
                module.proj = LoRALinear(module.proj, r=r, lora_alpha=alpha)
                attn_count += 2
            elif module.__class__.__name__ == 'SwinTransformerBlockV2':
                # MLP: Sequential(Linear, GELU, Dropout, Linear, Dropout)
                names = list(module.mlp._modules.keys())
                for name in names:
                    m = module.mlp._modules[name]
                    if isinstance(m, nn.Linear):
                        module.mlp._modules[name] = LoRALinear(
                            m, r=r, lora_alpha=alpha)
                        mlp_count += 1
        print(f'LoRA: {attn_count} attention + {mlp_count} MLP linears')

    def _set_freeze_policy(self):
        """Shallow original weights frozen (LoRA only), deep unfrozen + LoRA."""
        # 1. Freeze ALL backbone params first
        for p in self.features.parameters():
            p.requires_grad = False

        # 2. Unfreeze deep original weights (indices 4-7)
        for i in range(self._SHALLOW_END, len(self.features)):
            for p in self.features[i].parameters():
                p.requires_grad = True

        # 3. Unfreeze ALL LoRA params (shallow + deep)
        for name, param in self.features.named_parameters():
            if 'lora' in name:
                param.requires_grad = True

        frozen = sum(1 for p in self.features.parameters() if not p.requires_grad)
        unfrozen = sum(1 for p in self.features.parameters() if p.requires_grad)
        lora_params = sum(1 for n, p in self.features.named_parameters()
                          if 'lora' in n and p.requires_grad)
        print(f'Freeze: frozen={frozen}, unfrozen={unfrozen} '
              f'(incl. {lora_params} LoRA params)')

    def _extract_features(self, x):
        feats = {}
        h = x
        for layer in self.features:
            h = layer(h)
            ch = h.shape[-1]
            if ch in self._TARGET_CHANNELS:
                stage_name = self._TARGET_CHANNELS[ch]
                feats[stage_name] = h.permute(0, 3, 1, 2).contiguous()
        return feats

    def forward(self, x):
        feats = self._extract_features(x)
        return self.head(feats)

    def parameter_stats(self):
        trainable = sum(p.numel() for p in self.parameters()
                        if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable / 1e6, total / 1e6
