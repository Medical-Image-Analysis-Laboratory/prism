import torch
import torch.nn as nn

import torch
import torch.nn as nn
from monai.networks.blocks import ResidualUnit, UpSample, Convolution
from typing import Sequence, Union

from monai.networks.nets import BasicUNet


class DenoisingResUNet3D(nn.Module):
    """
    A customizable 3D Residual U-Net designed for denoising and artifact removal.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        strides: Sequence[Union[int, tuple[int, int, int]]],
        kernel_size: Union[int, tuple[int, int, int]] = 3,
        up_sample_mode: str = "pixelshuffle",
        norm: str | None = "instance",
        weights_path: str | None = None,
    ):
        """
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            channels: Sequence of channel counts for each level. Length must be len(strides) + 1.
            strides: Sequence of strides for downsampling. Length defines the depth.
            kernel_size: Convolutional kernel size.
            up_sample_mode: Upsampling strategy (e.g., "pixelshuffle", "deconv", "nontrainable").
            norm: Normalization for the conv blocks. Defaults to "instance"
                (InstanceNorm3d, affine=False): it normalizes per-sample over the
                spatial dims with NO running statistics, so it adapts to each
                subject's intensity scale instead of baking in a global scale the
                way BatchNorm would. The final residual projection stays unnormed.
            weights_path: Optional path to pre-trained weights for the regularizer.
        """
        super().__init__()

        self.norm = norm
        self.depth = len(strides)

        if len(channels) != self.depth + 1:
            raise ValueError(
                "The length of 'channels' must be one greater than the length of 'strides'."
            )

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.upsample = nn.ModuleList()

        # ---------------------------------------------------------------------
        # ENCODER
        # ---------------------------------------------------------------------
        # Level 0 (Produces the highest resolution skip connection; no downsampling)
        self.encoder.append(
            ResidualUnit(
                spatial_dims=3,
                in_channels=in_channels,
                out_channels=channels[0],
                strides=1,
                kernel_size=kernel_size,
                subunits=2,
                norm=norm,
                act="prelu",
            )
        )

        # Levels 1 to Depth-1 (Downsampling happens here)
        for i in range(self.depth - 1):
            self.encoder.append(
                ResidualUnit(
                    spatial_dims=3,
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    strides=strides[i],
                    kernel_size=kernel_size,
                    subunits=2,
                    norm=norm,
                    act="prelu",
                )
            )

        # ---------------------------------------------------------------------
        # BOTTLENECK
        # ---------------------------------------------------------------------
        self.bottleneck = ResidualUnit(
            spatial_dims=3,
            in_channels=channels[-2],
            out_channels=channels[-1],
            strides=strides[-1],
            kernel_size=kernel_size,
            subunits=2,
            norm=norm,
            act="prelu",
        )

        # ---------------------------------------------------------------------
        # DECODER
        # ---------------------------------------------------------------------
        # Iterate backwards to mirror the encoder
        for i in range(self.depth - 1, -1, -1):
            self.upsample.append(
                UpSample(
                    spatial_dims=3,
                    in_channels=channels[i + 1],
                    out_channels=channels[i],
                    scale_factor=strides[i],
                    mode=up_sample_mode,
                )
            )
            # Input to decoder is channels[i]*2 because of skip connection concatenation
            self.decoder.append(
                ResidualUnit(
                    spatial_dims=3,
                    in_channels=channels[i] * 2,
                    out_channels=channels[i],
                    strides=1,
                    kernel_size=kernel_size,
                    subunits=2,
                    norm=norm,
                    act="prelu",
                )
            )

        # Final projection to desired output channels
        self.final_conv = Convolution(
            spatial_dims=3,
            in_channels=channels[0],
            out_channels=out_channels,
            strides=1,
            kernel_size=1,
            norm=None,
            act=None,
        )
        self._initialize_weights()
        self._load_weights(weights_path)

    def _load_weights(self, weights_path: str | None):
        """Load the state dict from the provided weights path, if available."""
        if weights_path is not None:
            try:
                state_dict = torch.load(weights_path, map_location="cuda")
                self.load_state_dict(state_dict)
                print(f"Successfully loaded weights from {weights_path}")
            except Exception as e:
                print(f"Error loading weights from {weights_path}: {e}")
        else:
            print("No weights path provided; using randomly initialized weights.")

    def _initialize_weights(self):
        # The convs inside the pixelshuffle UpSample blocks are ICNR-initialized
        # by MONAI specifically to avoid checkerboard artifacts. Re-running a
        # generic Kaiming init over them overwrites that ICNR seeding and brings
        # the checkerboard back, so we skip every module under self.upsample.
        upsample_modules = set(self.upsample.modules())

        # 1. Standard initialization for all NON-upsample modules
        for m in self.modules():
            if m in upsample_modules:
                continue
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                # Kaiming is great for PReLU/ReLU
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm3d, nn.InstanceNorm3d)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # 2. THE CRITICAL STEP: Zero-out the final projection
        # This makes the global residual (x_input + out) ≈ x_input
        # We use a very small std dev instead of pure zero to help gradients start moving
        if hasattr(self.final_conv, "conv"):
            # If using MONAI's Convolution wrapper, access the underlying conv
            target_layer = self.final_conv.conv
        else:
            target_layer = self.final_conv

        nn.init.normal_(target_layer.weight, mean=0.0, std=1e-5)
        if target_layer.bias is not None:
            nn.init.constant_(target_layer.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # No input standardization: the input always comes straight from the CG
        # reconstruction (channel 0) plus the per-voxel reliability map (channel 1),
        # both already correctly bounded and scaled, so the net runs on the raw
        # volume and predicts the residual directly. The final projection is
        # zero-initialized, so the residual starts at ~0 (identity prior).
        skips = []

        # 1. Encoder Path
        for enc in self.encoder:
            x = enc(x)
            skips.append(x)

        # 2. Bottleneck
        x = self.bottleneck(x)

        # 3. Decoder Path (zips upsampling blocks, decoder blocks, and reversed skips)
        for up, dec, skip in zip(self.upsample, self.decoder, reversed(skips)):
            x = up(x)
            x = torch.cat([x, skip], dim=1)  # Inter-layer residual (Skip connection)
            x = dec(x)

        # 4. Final Projection: the raw residual (added to the input by the caller).
        out = self.final_conv(x)

        return out


class MoDLRegularizer3D(nn.Module):
    """
    3D Residual CNN Denoiser for MoDL/J-MoDL.

    This network acts as the proximal operator (regularizer) within the
    unrolled optimization loop. It predicts the noise/artifact component
    and subtracts it from the input (Residual Learning).
    """

    def __init__(self, in_channels=2, out_channels=1, hidden_channels=64, num_layers=5):
        super(MoDLRegularizer3D, self).__init__()

        layers = []

        # --- Layer 1: Input Convolution ---
        # 3x3x3 kernel, padding=1 to maintain spatial/depth dimensions
        layers.append(nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1))
        layers.append(nn.LeakyReLU(inplace=True))

        # --- Layers 2 to 4: Hidden Layers (Conv + BN + ReLU) ---
        # The MoDL paper typically uses 5 layers total, so we add (num_layers - 2) hidden blocks.
        for _ in range(num_layers - 2):
            layers.append(
                nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
            )
            # layers.append(nn.InstanceNorm3d(hidden_channels))
            layers.append(nn.LeakyReLU(inplace=True))

        # --- Layer 5: Output Convolution ---
        # Projects back to original channel dimension (e.g., Real/Imag)
        # No activation or BatchNorm here to allow full dynamic range.
        layers.append(
            nn.Conv3d(hidden_channels, out_channels, kernel_size=3, padding=1)
        )

        # Wrap layers in Sequential container
        self.net = nn.Sequential(*layers)

        # Initialize the weights automatically upon instantiation
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initializes the model's weights.
        Standard layers use Kaiming Normal for healthy gradient flow, while
        the final projection layer is initialized close to zero so the initial
        predictions (noise estimates) are minimal.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # Standard initialization for most layers (good for ReLU)
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.InstanceNorm3d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Target the very last convolutional layer specifically
        last_conv = self.net[-1]
        if isinstance(last_conv, nn.Conv3d):
            # Initialize with a very small standard deviation
            nn.init.normal_(last_conv.weight, mean=0.0, std=1e-3)
            if last_conv.bias is not None:
                nn.init.zeros_(last_conv.bias)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (Batch, Channels, Depth, Height, Width)
        Returns:
            out: Denoised tensor of same shape.
        """
        # The network estimates the noise/artifacts (residual)
        return self.net(x)


class SpatReg(nn.Module):

    def __init__(
        self,
        features=(8, 16, 32, 64, 64, 8),
    ):
        super().__init__()

        self.unet = BasicUNet(
            spatial_dims=3,
            in_channels=3,
            out_channels=1,
            features=features,
            norm="instance",
            upsample="pixelshuffle",
        )

    def forward(self, reg_vol, dc_vol):
        diff_map = torch.abs(dc_vol - reg_vol)
        # 1. Stack the volumes
        x = torch.stack([reg_vol, dc_vol, diff_map], dim=2)[0]

        # 2. Extract regional artifact features (Downsampled by 4x)
        regmap = self.unet(x)

        # 4. Constrain lambda to be positive and reasonably bounded (e.g., 0 to 10)
        lambda_map = torch.sigmoid(regmap)

        return lambda_map


# --- Example Usage ---
if __name__ == "__main__":
    # Define parameters
    batch_size = 1
    channels = 1  # Real and Imaginary parts
    depth = 128  # Number of slices
    height = 128
    width = 128

    # Instantiate the model
    denoiser = MoDLRegularizer3D(
        in_channels=channels, out_channels=1, hidden_channels=64, num_layers=5
    )

    # Create a random 3D input tensor (simulating a noisy complex MRI volume)
    input_volume = torch.randn(batch_size, channels, depth, height, width)

    # Forward pass
    output_volume = denoiser(input_volume)

    print(f"Input shape:  {input_volume.shape}")
    print(f"Output shape: {output_volume.shape}")

    # Verify parameter count (should be small, approx 200k-300k parameters)
    total_params = sum(p.numel() for p in denoiser.parameters())
    print(f"Total parameters: {total_params}")
