import torch
import torch.nn as nn

from monai.networks.nets import BasicUNet
from synthgen.utils.scanutils import remove_parameterized_bias_field


class QualityNet(nn.Module):

    def __init__(
        self,
        features=(8, 16, 32, 64, 64, 8),
    ):

        super().__init__()

        self.unet = BasicUNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            features=features,
            norm="instance",
            upsample="pixelshuffle",
        )

    def forward(self, x, slice_mask=None):

        logits = self.unet(x)

        quality = torch.sigmoid(logits)

        if slice_mask is not None:
            # Assuming slice_mask is 1 for tissue, 0 for background
            # Force background pixels to be exactly 1.0
            quality = torch.where(
                slice_mask == 0, torch.tensor(1.0, device=x.device), quality
            )

        return quality


class BiasNet(nn.Module):
    """
    A lightweight ~97k parameter encoder for extracting global gradients.
    """

    def __init__(self, num_coeffs=6):
        super().__init__()

        # 4 sequential downsampling blocks
        self.features = nn.Sequential(
            # Input: (B, 2, H, W) -> Output: (B, 16, H/2, W/2)
            nn.Conv2d(2, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            # -> Output: (B, 32, H/4, W/4)
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            # -> Output: (B, 64, H/8, W/8)
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            # -> Output: (B, 128, H/16, W/16)
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            # Global Average Pooling flattens the remaining spatial dims to 1x1
            # -> Output: (B, 128, 1, 1)
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # Final regression to your 6 coefficients
        self.regressor = nn.Linear(128, num_coeffs)

    def forward(self, x_stack, x_mask, basis):
        x = torch.cat([x_stack, x_mask], dim=1)  # Concatenate along channel dimension
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten for linear layer
        coeffs = self.regressor(x)
        corrected_v, bias_field = remove_parameterized_bias_field(
            x_stack, coeffs, basis
        )
        return corrected_v, bias_field, coeffs


if __name__ == "__main__":

    model = QualityNet()

    x = torch.randn(1, 1, 128, 128)

    out = model(x)

    print(out.shape)

    # min max mean print
    print("quality min:", out.min().item())
    print("quality max:", out.max().item())
    print("quality mean:", out.mean().item())
    # print number of parameters
    print("number of parameters:", sum(p.numel() for p in model.parameters()))
