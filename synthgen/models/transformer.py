import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from synthgen.generator.transform import mat_update_resolution, point2mat
from synthgen.models.attention import TransformerEncoder, PositionalEncoding, ResNet
from synthgen.generator.slice_acquisition import slice_acquisition


class SVRtransformerV2(nn.Module):
    def __init__(
        self,
        n_res=50,
        n_layers=4,
        n_head=4 * 2,
        d_in=11,
        d_out=9,
        d_model=256,
        d_inner=512,
        dropout=0,
        n_channels=2,
    ):
        super().__init__()
        self.img_encoder = ResNet(
            n_res=n_res, d_model=d_model, pretrained=False, d_in=n_channels + 2
        )
        self.pos_emb = PositionalEncoding(d_model, d_in)
        self.encoder = TransformerEncoder(
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_model // n_head,
            d_v=d_model // n_head,
            d_model=d_model,
            d_inner=d_inner,
            dropout=dropout,
            activation_attn="softmax",
            activation_ff="gelu",
            prenorm=False,
        )
        self.fc = nn.Linear(d_model, d_out)
        self.fc_score = nn.Linear(d_model, 1)

    def pos_augment(self, slices, slices_est):
        n, _, h, w = slices.shape
        y = torch.linspace(-(h - 1) / 256, (h - 1) / 256, h, device=slices.device)
        x = torch.linspace(-(w - 1) / 256, (w - 1) / 256, w, device=slices.device)
        y, x = torch.meshgrid(y, x, indexing="ij")  # hxw
        if slices_est is not None:
            slices = torch.cat(
                [
                    slices,
                    slices_est,
                    y.view(1, 1, h, w).expand(n, -1, -1, -1),
                    x.view(1, 1, h, w).expand(n, -1, -1, -1),
                ],
                1,
            )
        else:
            slices = torch.cat(
                [
                    slices,
                    y.view(1, 1, h, w).expand(n, -1, -1, -1),
                    x.view(1, 1, h, w).expand(n, -1, -1, -1),
                ],
                1,
            )
        return slices

    def forward(self, theta, slices, pos, volume, params, attn_mask):
        y = volume
        if y is not None:
            with torch.no_grad():
                transforms = mat_update_resolution(point2mat(theta), 1, params["res_r"])
                y = slice_acquisition(
                    transforms,
                    y,
                    None,
                    None,
                    params["psf"],
                    params["slice_shape"],
                    params["res_s"] / params["res_r"],
                    False,
                    params["interp_psf"],
                )
        pos = torch.cat((theta, pos), -1)
        pe = self.pos_emb(pos)
        if isinstance(self.img_encoder, ResNet):
            slices = self.pos_augment(slices, y)
        else:  # ViT
            if y is not None:
                slices = torch.cat([slices, y], 1)
        x = self.img_encoder(slices)
        x, attn = self.encoder(x, pe, attn_mask)
        dtheta = self.fc(x)

        score = self.fc_score(x)
        score = F.softmax(score, dim=0) * score.shape[0]
        score = torch.clamp(score, max=3.0)

        return theta + dtheta, score, attn


class ResBlock3D(nn.Module):
    def __init__(self, channels, groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
        )

    def forward(self, x):
        return F.gelu(x + self.block(x))


class VolumeEncoder(nn.Module):
    def __init__(self, d_model, spatial_dim=8, patch_size=8):
        """
        Args:
            d_model: transformer embedding dimension
            spatial_dim: final spatial resolution per axis (default 8 → 512 tokens)
            patch_size: initial 3D patch size (default 8)
        """
        super().__init__()

        # --- Patch embedding (explicit tokenization) ---
        self.patch_embed = nn.Conv3d(
            in_channels=1,
            out_channels=64,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

        # --- Shallow residual encoder ---
        self.encoder = nn.Sequential(
            nn.GroupNorm(8, 64),
            nn.GELU(),
            ResBlock3D(64),
            ResBlock3D(64),
            # Downsample to reach spatial_dim
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            ResBlock3D(128),
        )

        # --- Projection to transformer dimension ---
        self.proj = nn.Conv3d(128, d_model, kernel_size=1, bias=False)

        # --- Positional embeddings ---
        self.pos_emb = nn.Parameter(torch.randn(1, spatial_dim**3, d_model))

    def forward(self, x):
        """
        x: (B, 1, 128, 128, 128)
        returns: (B, spatial_dim^3, d_model)
        """
        x = self.patch_embed(x)  # (B, 64, 16, 16, 16)
        x = self.encoder(x)  # (B, 128, 8, 8, 8)
        x = self.proj(x)  # (B, d_model, 8, 8, 8)

        B, C, H, W, D = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, 512, d_model)

        return x + self.pos_emb


class SVRtransformerV3(nn.Module):
    def __init__(
        self,
        n_res=50,
        n_layers=4 * 2,
        n_head=4,
        d_in=9 + 2,
        d_out=9,
        d_model=256,
        d_inner=512,
        dropout=0.0,
    ):
        super().__init__()
        # 2D Encoder for acquired slices
        self.img_encoder = ResNet(n_res=n_res, d_model=d_model, d_in=1)
        self.pos_emb_slice = PositionalEncoding(d_model, d_in)

        # 3D Encoder for current volume estimate
        self.vol_encoder = VolumeEncoder(d_model=d_model)

        # Special Tokens
        self.type_emb = nn.Embedding(2, d_model)  # 0: Slice, 1: Volume
        self.no_vol_token = nn.Parameter(torch.randn(1, d_model))  # Iter 0 fallback

        self.encoder = TransformerEncoder(
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_model // n_head,
            d_v=d_model // n_head,
            d_model=d_model,
            d_inner=d_inner,
            dropout=dropout,
            activation_attn="softmax",
            activation_ff="gelu",
            prenorm=False,
        )
        self.fc = nn.Linear(d_model, d_out)
        self.fc_score = nn.Linear(d_model, 1)

    def forward(self, theta, slices, pos, volume, params, attn_mask=None):
        # 1. Process Slices
        pe_slice = self.pos_emb_slice(torch.cat((theta, pos), -1))
        x_slice = self.img_encoder(slices) + pe_slice
        x_slice = x_slice + self.type_emb.weight[0]

        # 2. Process Volume
        if volume is None:
            # Iteration 0: Use "No Volume Available" token
            x_vol = self.no_vol_token + self.type_emb.weight[1]
        else:
            # Iteration N: Extract tokens from current volume estimate [5, 6]
            x_vol = self.vol_encoder(volume)[0]
            x_vol = x_vol + self.type_emb.weight[1]

        # 3. Concatenate and Attend
        # Sequence: [Slice_1, ..., Slice_N, Vol_1, ..., Vol_512]
        x_combined = torch.cat([x_slice, x_vol], dim=0)

        # Transformer attends to both modalities [8-10]
        # Note: We pass pos_enc as 0 because we already added it
        x_out, attn = self.encoder(x_combined, 0, attn_mask)

        # 4. Regress from Slice Tokens only
        x_slice_out = x_out[: x_slice.shape[0], :]

        dtheta = self.fc(x_slice_out)
        score = self.fc_score(x_slice_out)
        score = F.softmax(score, dim=0) * score.shape[0]
        score = torch.clamp(score, max=3.0)

        return theta + dtheta, score, attn[:, :, : x_slice.shape[0], : x_slice.shape[0]]


class ResNetEncoder(nn.Module):
    """ResNet backbone that exposes BOTH a pooled per-slice token (for the
    transformer) and the intermediate feature maps (for a dense decoder).

    Mirrors the construction of ``attention.ResNet`` but taps the stage
    activations so a U-Net style decoder can build a full-resolution map.
    """

    def __init__(self, n_res, d_model, d_in=1):
        super().__init__()
        resnet_fn = getattr(tvm, "resnet%d" % n_res)
        model = resnet_fn(
            pretrained=False,
            norm_layer=lambda x: nn.BatchNorm2d(x, track_running_stats=False),
        )
        model.conv1 = nn.Conv2d(
            d_in, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool
        self.fc = nn.Linear(model.fc.in_features, d_model)

    @staticmethod
    def _block_out_channels(layer):
        last = layer[-1]
        if hasattr(last, "conv3"):  # Bottleneck (resnet50/101/152)
            return last.conv3.out_channels
        return last.conv2.out_channels  # BasicBlock (resnet18/34)

    @property
    def feature_channels(self):
        """Channels of (f0, f1, f2, f3, f4) at strides (2, 4, 8, 16, 32)."""
        return (
            64,
            self._block_out_channels(self.layer1),
            self._block_out_channels(self.layer2),
            self._block_out_channels(self.layer3),
            self._block_out_channels(self.layer4),
        )

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        f0 = x  # (N, 64,  H/2,  W/2)
        x = self.maxpool(x)
        f1 = self.layer1(x)  # (N, c1, H/4,  W/4)
        f2 = self.layer2(f1)  # (N, c2, H/8,  W/8)
        f3 = self.layer3(f2)  # (N, c3, H/16, W/16)
        f4 = self.layer4(f3)  # (N, c4, H/32, W/32)
        pooled = self.fc(torch.flatten(self.avgpool(f4), 1))  # (N, d_model)
        return pooled, (f0, f1, f2, f3, f4)


class _UpBlock(nn.Module):
    """Upsample x2, fuse with a skip connection, refine with two conv blocks."""

    def __init__(self, in_ch, skip_ch, out_ch, groups=8):
        super().__init__()
        self.reduce = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.GELU(),
        )

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.reduce(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class QADecoder(nn.Module):
    """Dense per-pixel quality decoder.

    Reads the encoder skip features and is conditioned on each slice's
    post-transformer context vector (broadcast/added at the bottleneck) so the
    quality of a pixel can depend on the other slices in the stack.

    Returns logits of shape (N, 1, H, W) at the input slice resolution.
    """

    def __init__(self, feat_channels, ctx_dim, dec_channels=(256, 128, 64, 32)):
        super().__init__()
        c0, c1, c2, c3, c4 = feat_channels
        # Cross-slice conditioning: project the transformer context into the
        # bottleneck channel space and add it (FiLM-style additive bias).
        self.ctx_proj = nn.Linear(ctx_dim, c4)
        self.up3 = _UpBlock(c4, c3, dec_channels[0])  # H/32 -> H/16
        self.up2 = _UpBlock(dec_channels[0], c2, dec_channels[1])  # -> H/8
        self.up1 = _UpBlock(dec_channels[1], c1, dec_channels[2])  # -> H/4
        self.up0 = _UpBlock(dec_channels[2], c0, dec_channels[3])  # -> H/2
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # -> H
            nn.Conv2d(
                dec_channels[3], dec_channels[3], kernel_size=3, padding=1, bias=False
            ),
            nn.GroupNorm(8, dec_channels[3]),
            nn.GELU(),
            nn.Conv2d(dec_channels[3], 1, kernel_size=1),
        )

    def forward(self, feats, ctx):
        f0, f1, f2, f3, f4 = feats
        f4 = f4 + self.ctx_proj(ctx).unsqueeze(-1).unsqueeze(-1)
        x = self.up3(f4, f3)
        x = self.up2(x, f2)
        x = self.up1(x, f1)
        x = self.up0(x, f0)
        return self.final(x)


class SVRtransformerV4(nn.Module):
    """Multi-task slice-to-volume registration transformer.

    A single shared encoder + cross-slice transformer drives four heads:
        1. motion              -> theta + dtheta        (N, d_out)
        2. slice intensity     -> per-slice scale       (N, 1)
        3. slice bias field    -> polynomial coeffs      (N, n_bias_coeffs)   [aux]
        4. slice quality (QA)  -> dense pixel map        (N, 1, H, W)         [aux]

    The two aux heads (bias + QA) are gated by ``aux_heads`` so a cheap
    registration-only refiner can skip them.
    """

    def __init__(
        self,
        n_res=50,
        n_layers=4,
        n_head=4 * 2,
        d_in=11,
        d_out=9,
        d_model=256,
        d_inner=512,
        dropout=0,
        n_channels=2,
        n_bias_coeffs=6,
        aux_heads=True,
        max_int_log=0.6931,
        qa_floor=0.05,
        qa_init_bias=2.0,
    ):
        super().__init__()
        # Bound for the per-slice log intensity scale; exp(+-max_int_log) gives a
        # symmetric multiplicative range around 1.0 (default ~[0.5, 2.0]).
        self.max_int_log = max_int_log
        # Minimum dense-QA value: every in-brain pixel always contributes at least
        # `qa_floor` to the reconstruction / data-consistency, so QA can downweight
        # artifacts but cannot collapse a region to 0 and starve the recon of slice
        # information. Final QA = qa_floor + (1 - qa_floor) * sigmoid(logits).
        self.qa_floor = qa_floor
        self.qa_init_bias = qa_init_bias
        self.aux_heads = aux_heads
        self.img_encoder = ResNetEncoder(
            n_res=n_res, d_model=d_model, d_in=n_channels + 2
        )
        self.pos_emb = PositionalEncoding(d_model, d_in)
        self.encoder = TransformerEncoder(
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_model // n_head,
            d_v=d_model // n_head,
            d_model=d_model,
            d_inner=d_inner,
            dropout=dropout,
            activation_attn="softmax",
            activation_ff="gelu",
            prenorm=False,
        )
        self.fc = nn.Linear(d_model, d_out)
        self.fc_score = nn.Linear(d_model, 1)
        # Start the intensity head at the identity: zero weights+bias make the
        # output 0, so int_scale = exp(max_int_log * tanh(0)) = 1 for every slice.
        # The model then learns to move away from "no intensity shift" only where
        # the loss justifies it (the head is unsupervised), instead of corrupting
        # the reconstruction with a random per-slice scale at initialization.
        nn.init.zeros_(self.fc_score.weight)
        nn.init.zeros_(self.fc_score.bias)

        if aux_heads:
            self.fc_bias = nn.Linear(d_model, n_bias_coeffs)
            # The scanner forces the constant (0th) bias coefficient to zero to
            # avoid global intensity shifts; mirror that so predictions match GT
            # formatting (see apply_parameterized_bias_field).
            coeff_mask = torch.ones(n_bias_coeffs)
            coeff_mask[0] = 0.0
            self.register_buffer("bias_coeff_mask", coeff_mask.view(1, -1))
            # Start the bias head at the identity: zero coeffs => log_bias = 0 =>
            # bias_field = exp(0) = 1 (no bias) for every slice. The unsupervised
            # bias head then adds a field only where consistency demands it, rather
            # than applying a random bias to all slices from step 0.
            nn.init.zeros_(self.fc_bias.weight)
            nn.init.zeros_(self.fc_bias.bias)
            self.qa_decoder = QADecoder(
                self.img_encoder.feature_channels, ctx_dim=d_model
            )
            # Start QA high (trust all slices): bias the final QA logit positive so
            # the initial quality is ~qa_floor + (1-qa_floor)*sigmoid(qa_init_bias)
            # (e.g. ~0.89 for floor=0.1, bias=2.0). The model then learns to REJECT
            # artifacts, instead of starting at 0.5 and collapsing the interior to 0.
            nn.init.constant_(self.qa_decoder.final[-1].bias, self.qa_init_bias)

    def pos_augment(self, slices, slices_est):
        n, _, h, w = slices.shape
        y = torch.linspace(-(h - 1) / 256, (h - 1) / 256, h, device=slices.device)
        x = torch.linspace(-(w - 1) / 256, (w - 1) / 256, w, device=slices.device)
        y, x = torch.meshgrid(y, x, indexing="ij")  # hxw
        if slices_est is not None:
            slices = torch.cat(
                [
                    slices,
                    slices_est,
                    y.view(1, 1, h, w).expand(n, -1, -1, -1),
                    x.view(1, 1, h, w).expand(n, -1, -1, -1),
                ],
                1,
            )
        else:
            slices = torch.cat(
                [
                    slices,
                    y.view(1, 1, h, w).expand(n, -1, -1, -1),
                    x.view(1, 1, h, w).expand(n, -1, -1, -1),
                ],
                1,
            )
        return slices

    def forward(self, theta, slices, pos, volume, params, attn_mask, stack_mask=None):
        y = volume
        if y is not None:
            with torch.no_grad():
                transforms = mat_update_resolution(point2mat(theta), 1, params["res_r"])
                y = slice_acquisition(
                    transforms,
                    y,
                    None,
                    None,
                    params["psf"],
                    params["slice_shape"],
                    params["res_s"] / params["res_r"],
                    False,
                    params["interp_psf"],
                )
        pos = torch.cat((theta, pos), -1)
        pe = self.pos_emb(pos)
        slices_in = self.pos_augment(slices, y)
        pooled, feats = self.img_encoder(slices_in)
        ctx, attn = self.encoder(pooled, pe, attn_mask)

        dtheta = self.fc(ctx)
        # Per-slice intensity scale. Independent across slices (NO softmax, so the
        # head is free to model each slice's intensity shift instead of being forced
        # to a relative weight that sums to N). Parameterized as a bounded log-scale
        # centered at 1.0; the global intensity gauge is anchored by the
        # reconstruction (image) loss, not by a mean/sum constraint.
        int_scale = torch.exp(self.max_int_log * torch.tanh(self.fc_score(ctx)))
        out_theta = theta + dtheta

        if not self.aux_heads:
            return out_theta, int_scale, None, None, attn

        bias_coeffs = self.fc_bias(ctx) * self.bias_coeff_mask
        # Floor the QA so it lives in [qa_floor, 1]: artifacts can be downweighted
        # but no region can collapse to 0 and starve the reconstruction of slices.
        qa = self.qa_floor + (1.0 - self.qa_floor) * torch.sigmoid(
            self.qa_decoder(feats, ctx)
        )
        if stack_mask is not None:
            # Pixels outside the brain mask are trivially "good" -> force to 1.0
            qa = torch.where(stack_mask == 0, torch.ones_like(qa), qa)

        return out_theta, int_scale, bias_coeffs, qa, attn


class SVRtransformerV5(nn.Module):
    """Multi-task SVR transformer with per-task encoders and cross-task attention.

    Each task has its OWN lightweight ResNet encoder. The per-slice tokens from
    all active tasks are tagged with a learned task embedding plus a shared pose
    encoding and attend jointly (cross-task AND cross-slice in one pass), then
    each task is decoded by its own head:
        1. motion      -> theta + dtheta      (N, d_out)
        2. intensity   -> per-slice scale     (N, 1)
        3. bias        -> polynomial coeffs    (N, n_bias_coeffs)
        4. quality(QA) -> dense pixel map      (N, 1, H, W)

    A SINGLE model serves both iter=0 (no volume estimate) and later iterations:
    the volume re-slice ("sim") channel is zero-filled at iter 0 and a learned
    ``no_volume_emb`` is added to every token so the network knows the reference
    is absent. Every encoder therefore always sees both the real and the sim
    slice, so intensity/bias/quality can be judged relative to the reconstruction
    whenever one exists.

    ``active_tasks`` gates which heads run (and are supervised), so a training
    stage can optimise only a subset of tasks. All four encoders/heads are always
    BUILT (only the forward compute is gated) so checkpoints stay key-compatible
    across stages and a later stage can finetune from an earlier stage's weights.
    """

    TASK_ORDER = ("motion", "bias", "intensity", "quality")

    def __init__(
        self,
        n_res=18,
        n_layers=8,
        n_head=8,
        d_in=11,
        d_out=9,
        d_model=256,
        d_inner=512,
        dropout=0.0,
        n_channels_real=1,
        n_bias_coeffs=6,
        max_int_log=0.6931,
        qa_floor=0.05,
        qa_init_bias=2.0,
        active_tasks=("motion", "bias", "intensity", "quality"),
    ):
        super().__init__()
        # See SVRtransformerV4 for the rationale behind these bounds/floors.
        self.max_int_log = max_int_log
        self.qa_floor = qa_floor
        self.qa_init_bias = qa_init_bias
        self.active_tasks = tuple(active_tasks)
        self.n_channels_real = n_channels_real

        # Every encoder always sees: real slice(s) + sim re-slice + 2 coord
        # channels. The sim channel is zero-filled at iter 0 so the input channel
        # count is constant regardless of whether a volume estimate exists.
        in_ch = n_channels_real + 1 + 2

        # --- Per-task CNN encoders (separate weights) ---
        self.enc_motion = ResNet(n_res=n_res, d_model=d_model, d_in=in_ch)
        self.enc_bias = ResNet(n_res=n_res, d_model=d_model, d_in=in_ch)
        self.enc_intensity = ResNet(n_res=n_res, d_model=d_model, d_in=in_ch)
        # Quality needs the intermediate skip features for its dense decoder.
        self.enc_quality = ResNetEncoder(n_res=n_res, d_model=d_model, d_in=in_ch)

        # --- Token tagging ---
        self.pos_emb = PositionalEncoding(d_model, d_in)
        self.task_emb = nn.Embedding(len(self.TASK_ORDER), d_model)
        # Zero-init so it is a no-op at the start of training (clean identity for
        # the volume-present case); the model learns a non-zero flag only if the
        # no-volume case benefits from behaving differently.
        self.no_volume_emb = nn.Parameter(torch.zeros(1, d_model))

        # --- Cross-task / cross-slice communication ---
        self.encoder = TransformerEncoder(
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_model // n_head,
            d_v=d_model // n_head,
            d_model=d_model,
            d_inner=d_inner,
            dropout=dropout,
            activation_attn="softmax",
            activation_ff="gelu",
            prenorm=False,
        )

        # --- Task-specific heads ---
        self.fc = nn.Linear(d_model, d_out)

        # Per-slice reliability head (reads the motion token), SVoRT-style: a score
        # turned into a softmax-over-slices weight (x N so the mean is ~1) and
        # clamped, used to weight the CG reconstruction. Learned UNSUPERVISED purely
        # through the image loss ("down-weighting an unreliable slice lowers the
        # reconstruction error"). Zero-init -> uniform weights (= 1) at the start.
        self.fc_iqa = nn.Linear(d_model, 1)
        nn.init.zeros_(self.fc_iqa.weight)
        nn.init.zeros_(self.fc_iqa.bias)

        # Intensity head starts at the identity (int_scale = 1): zero weights+bias
        # so the unsupervised-at-init scale does not corrupt the reconstruction.
        self.fc_score = nn.Linear(d_model, 1)
        nn.init.zeros_(self.fc_score.weight)
        nn.init.zeros_(self.fc_score.bias)

        # Bias head starts at the identity (bias_field = 1); the scanner forces the
        # 0th (constant) coefficient to zero to avoid global shifts, so mirror that.
        self.fc_bias = nn.Linear(d_model, n_bias_coeffs)
        coeff_mask = torch.ones(n_bias_coeffs)
        coeff_mask[0] = 0.0
        self.register_buffer("bias_coeff_mask", coeff_mask.view(1, -1))
        nn.init.zeros_(self.fc_bias.weight)
        nn.init.zeros_(self.fc_bias.bias)

        # Dense QA decoder, conditioned on the post-attention quality token. Start
        # QA high (trust all slices) so the model learns to REJECT artifacts.
        self.qa_decoder = QADecoder(self.enc_quality.feature_channels, ctx_dim=d_model)
        nn.init.constant_(self.qa_decoder.final[-1].bias, self.qa_init_bias)

    def pos_augment(self, slices, slices_est):
        """Concatenate [real, sim, x_coord, y_coord]; sim is zero-filled if None.

        Unlike V4, the sim channel is ALWAYS present (zeros when no volume) so the
        encoder input channel count is constant across iterations.
        """
        n, _, h, w = slices.shape
        y = torch.linspace(-(h - 1) / 256, (h - 1) / 256, h, device=slices.device)
        x = torch.linspace(-(w - 1) / 256, (w - 1) / 256, w, device=slices.device)
        y, x = torch.meshgrid(y, x, indexing="ij")  # hxw
        if slices_est is None:
            slices_est = torch.zeros_like(slices)
        return torch.cat(
            [
                slices,
                slices_est,
                y.view(1, 1, h, w).expand(n, -1, -1, -1),
                x.view(1, 1, h, w).expand(n, -1, -1, -1),
            ],
            1,
        )

    def forward(self, theta, slices, pos, volume, params, attn_mask, stack_mask=None):
        no_volume = volume is None
        y = volume
        if y is not None:
            with torch.no_grad():
                transforms = mat_update_resolution(point2mat(theta), 1, params["res_r"])
                y = slice_acquisition(
                    transforms,
                    y,
                    None,
                    None,
                    params["psf"],
                    params["slice_shape"],
                    params["res_s"] / params["res_r"],
                    False,
                    params["interp_psf"],
                )
        slices_in = self.pos_augment(slices, y)
        n = slices.shape[0]

        # Shared per-slice tags: pose encoding + (optional) no-volume flag.
        pe = self.pos_emb(torch.cat((theta, pos), -1))
        nv = self.no_volume_emb if no_volume else 0.0
        active = self.active_tasks

        # --- Encode each active task and tag its tokens (motion always active) ---
        tokens = []  # ordered per-task token blocks, each (N, d_model)
        order = []  # parallel task names, to split the context back afterwards
        feats_q = None

        tokens.append(self.enc_motion(slices_in) + pe + self.task_emb.weight[0] + nv)
        order.append("motion")
        if "bias" in active:
            tokens.append(self.enc_bias(slices_in) + pe + self.task_emb.weight[1] + nv)
            order.append("bias")
        if "intensity" in active:
            tokens.append(
                self.enc_intensity(slices_in) + pe + self.task_emb.weight[2] + nv
            )
            order.append("intensity")
        if "quality" in active:
            pooled_q, feats_q = self.enc_quality(slices_in)
            tokens.append(pooled_q + pe + self.task_emb.weight[3] + nv)
            order.append("quality")

        # --- Joint cross-task / cross-slice attention ---
        # Pose encoding is already folded into the tokens, so pass a scalar 0 as
        # the additive pos_enc (same trick as SVRtransformerV3).
        ctx_all, attn = self.encoder(torch.cat(tokens, dim=0), 0, attn_mask)
        chunks = ctx_all.split(n, dim=0)
        ctx = {name: chunks[k] for k, name in enumerate(order)}

        # --- Task-specific heads (identity sentinels for inactive tasks) ---
        out_theta = theta + self.fc(ctx["motion"])

        if "intensity" in active:
            int_scale = torch.exp(
                self.max_int_log * torch.tanh(self.fc_score(ctx["intensity"]))
            )
        else:
            int_scale = torch.ones(n, 1, device=slices.device, dtype=slices.dtype)

        if "bias" in active:
            bias_coeffs = self.fc_bias(ctx["bias"]) * self.bias_coeff_mask
        else:
            bias_coeffs = torch.zeros(
                n, self.fc_bias.out_features, device=slices.device, dtype=slices.dtype
            )

        if "quality" in active:
            qa = self.qa_floor + (1.0 - self.qa_floor) * torch.sigmoid(
                self.qa_decoder(feats_q, ctx["quality"])
            )
            if stack_mask is not None:
                qa = torch.where(stack_mask == 0, torch.ones_like(qa), qa)
        else:
            qa = torch.ones_like(slices)

        # Per-slice reliability weight (SVoRT-style): softmax over slices x N so the
        # mean is ~1, then clamped so no single slice dominates. The softmax keeps
        # the total weight conserved (no slice collapses to 0), and it is learned
        # purely through the image loss (not detached in the CG).
        if "regconf" in active:
            score = self.fc_iqa(ctx["motion"])  # (N, 1)
            conf = torch.clamp(F.softmax(score, dim=0) * score.shape[0], max=3.0)
        else:
            conf = torch.ones(n, 1, device=slices.device, dtype=slices.dtype)

        return out_theta, int_scale, bias_coeffs, qa, conf, attn


if __name__ == "__main__":
    # Smoke test for SVRtransformerV5: shapes + identity-init + task gating.
    import math

    N, H, W = 5, 128, 128
    theta = torch.randn(N, 9)
    slices = torch.rand(N, 1, H, W)
    pos = torch.randn(N, 2)
    stack_mask = (torch.rand(N, 1, H, W) > 0.5).float()
    params = {
        "psf": None,
        "slice_shape": (H, W),
        "res_s": 1.0,
        "res_r": 1.0,
        "interp_psf": False,
    }
    attn_mask = None

    model = SVRtransformerV5(
        d_model=128,
        d_inner=256,
        active_tasks=("motion", "bias", "intensity", "quality"),
    ).eval()

    # --- iter 0: no volume (CPU-safe, no slice_acquisition needed) ---
    with torch.no_grad():
        out_theta, int_scale, bias_coeffs, qa, conf, attn = model(
            theta, slices, pos, None, params, attn_mask, stack_mask=stack_mask
        )
    assert out_theta.shape == (N, 9), out_theta.shape
    assert int_scale.shape == (N, 1), int_scale.shape
    assert bias_coeffs.shape == (N, 6), bias_coeffs.shape
    assert qa.shape == (N, 1, H, W), qa.shape
    assert conf.shape == (N, 1), conf.shape
    # regconf inactive here -> confidence is all ones (no down-weighting).
    assert torch.equal(conf, torch.ones_like(conf)), "regconf gate (off) failed"
    # Identity init: scale ~1, coeffs ~0, qa bounded in [floor, 1] and starts high
    # (~ sigmoid(qa_init_bias); not exact because the QA head's 1x1 conv has random
    # weights on top of the init bias, so per-pixel logits vary around it).
    assert torch.allclose(int_scale, torch.ones_like(int_scale), atol=1e-5)
    assert torch.allclose(bias_coeffs, torch.zeros_like(bias_coeffs), atol=1e-5)
    qa_expected = model.qa_floor + (1 - model.qa_floor) / (
        1 + math.exp(-model.qa_init_bias)
    )
    in_brain = qa[stack_mask > 0]
    assert in_brain.min() >= model.qa_floor - 1e-5 and in_brain.max() <= 1.0 + 1e-5
    assert abs(in_brain.mean().item() - qa_expected) < 0.1, (
        in_brain.mean().item(),
        qa_expected,
    )
    # Background (outside brain mask) is forced to exactly 1.
    assert torch.all(qa[stack_mask == 0] == 1.0)
    print(
        "iter0 (no volume) OK:",
        out_theta.shape,
        int_scale.shape,
        bias_coeffs.shape,
        qa.shape,
    )

    # --- task gating: quality inactive -> qa is all ones (QADecoder skipped) ---
    model_g = SVRtransformerV5(
        d_model=128, d_inner=256, active_tasks=("motion", "bias", "intensity")
    ).eval()
    with torch.no_grad():
        _, _, _, qa_g, _, _ = model_g(
            theta, slices, pos, None, params, attn_mask, stack_mask=stack_mask
        )
    assert torch.equal(qa_g, torch.ones_like(qa_g)), "quality gate failed"
    print("task gating (quality off) OK: qa all-ones =", bool(torch.all(qa_g == 1)))

    # --- regconf active -> softmax-over-slices weight, mean ~1, clamped <= 3 ---
    model_r = SVRtransformerV5(
        d_model=128,
        d_inner=256,
        active_tasks=("motion", "bias", "intensity", "quality", "regconf"),
    ).eval()
    with torch.no_grad():
        *_, conf_r, _ = model_r(
            theta, slices, pos, None, params, attn_mask, stack_mask=stack_mask
        )
    assert conf_r.shape == (N, 1)
    assert conf_r.min() > 0.0 and conf_r.max() <= 3.0 + 1e-5
    # zero-init head -> uniform weights = 1 at the start (mean is exactly 1).
    assert abs(conf_r.mean().item() - 1.0) < 1e-4
    print(
        "regconf active OK: conf mean=%.3f range=[%.3f, %.3f]"
        % (conf_r.mean(), conf_r.min(), conf_r.max())
    )

    # --- iter>0: volume present (requires CUDA slice_acquisition + a real PSF) ---
    if torch.cuda.is_available():
        from synthgen.utils.scanutils import get_PSF

        model_c = model.cuda()
        vol = torch.rand(1, 1, H, H, H, device="cuda")
        params_c = dict(params, psf=get_PSF(device=torch.device("cuda")))
        with torch.no_grad():
            ot, iscale, bc, q, cf, a = model_c(
                theta.cuda(),
                slices.cuda(),
                pos.cuda(),
                vol,
                params_c,
                None,
                stack_mask=stack_mask.cuda(),
            )
        assert ot.shape == (N, 9) and q.shape == (N, 1, H, W) and cf.shape == (N, 1)
        print("iter>0 (volume) OK:", ot.shape, iscale.shape, bc.shape, q.shape)
    else:
        print("CUDA unavailable: skipped the volume-present (slice_acquisition) path.")

    print("\nSVRtransformerV5 smoke test passed.")
