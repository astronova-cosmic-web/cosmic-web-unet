# models

The compact U-Net (3-channel input: shear γ₁, γ₂ and mask; linear output).

```python
from cosmicweb.models import UNet

model = UNet()                        # 3 → 1 channels, base 32, depth 4, BatchNorm: 7.24 M parameters
kappa = model(x)                      # x: (B, 3, H, W) → (B, 1, H, W); H, W divisible by 16
```

| Setting | Default | Notes |
|---|---|---|
| `base_channels` | 32 | Channels at full resolution, doubled at each level. 16 gives 1.8 M parameters, 64 gives 29 M |
| `depth` | 4 | Number of 2× downsamplings. The bottleneck is 16 × 16 for 256² crops and 32 × 32 for full 512² maps |
| `norm` | `"batch"` | `"group"` is steadier for very small batches (for example full maps with batch size 2–4). `"none"` is also available |
| `padding_mode` | `"zeros"` | Also `"reflect"`, `"replicate"` or `"circular"`. Crops are not periodic, so avoid `"circular"` for them |

**Design**
- **Encoder:** (3×3 conv → norm → ReLU) × 2 per level, then 2×2 max pooling.
- **Decoder:** bilinear ×2 upsampling, a 1×1 conv, concatenation with the skip connection, then the same double conv.
- **Output:** a final 1×1 conv with **no activation**, because κ is signed and unbounded.
- **Why bilinear upsampling:** transposed convolutions tend to leave checkerboard artifacts, which would add false small-scale power, exactly what the power-spectrum loss and metrics measure.
- **Input sizes:** fully convolutional, so the same weights work on 256² crops and on full 512² maps.

Tests: `tests/test_unet.py`.
