# Third-party notices

`torch_unet.py` is a reduced PyTorch-only adaptation of MONAI 1.6.0:

- `monai/networks/nets/unet.py` (UNet)
- `monai/networks/blocks/convolutions.py` (Convolution and ResidualUnit)
- `monai/networks/blocks/acti_norm.py` (ADN)
- `monai/networks/layers/simplelayers.py` (SkipConnection)

Copyright (c) MONAI Consortium. Licensed under Apache License 2.0.
The full license is included in `licenses/MONAI-LICENSE.txt`.
Upstream: https://github.com/Project-MONAI/MONAI

Changes: only the 3-D network options used by the archived models are retained;
MONAI factories and utilities are replaced with standard torch.nn layers.
Module names, convolution settings, residual paths, normalization, activation
and skip ordering are retained for strict checkpoint compatibility.
This is a local adaptation, not an official MONAI distribution.
