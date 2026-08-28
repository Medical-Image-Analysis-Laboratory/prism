from pathlib import Path

import torch
from synthgen.generator.transform import RigidTransform, mat2point, mat2euler
import nibabel as nib
import numpy as np


def save_volume(fname, data, res, affine=None):
    data = data[0, 0].detach().cpu().numpy().transpose(2, 1, 0)
    w, h, d = data.shape
    if affine is None:
        affine = np.eye(4)
        affine[:3, :3] *= res
        affine[:3, -1] = -np.array([(w - 1) / 2.0, (h - 1) / 2.0, (d - 1) / 2.0]) * res
    else:
        affine = affine[0]
    img = nib.Nifti1Image(data, affine)
    img.header.set_xyzt_units(2)
    img.header.set_qform(affine, code="aligned")
    img.header.set_sform(affine, code="scanner")
    nib.save(img, fname)


def save_stack(slices, transforms, res_s, s_thick, scale, dtype, fname):
    data = (slices * scale).squeeze(1).cpu().numpy().astype(dtype).transpose(2, 1, 0)
    w, h, d = data.shape
    affine = np.eye(4)
    R = transforms[0].matrix()[0, :, :-1].cpu().numpy()
    T = transforms[0].matrix()[0, :, -1].cpu().numpy()
    T[0] -= (w - 1) / 2 * res_s
    T[1] -= (h - 1) / 2 * res_s
    T = R @ T.reshape(3, 1)
    R = R @ np.diag([res_s, res_s, s_thick])

    affine[:3, :] = np.concatenate((R, T), -1)
    if np.any(np.isnan(affine)):
        print(f"Warning: NaN in affine for saving stack {fname}")
        affine = np.eye(4)
    img = nib.Nifti1Image(data, affine)
    img.header.set_xyzt_units(2)
    img.header.set_qform(affine, code="aligned")
    img.header.set_sform(affine, code="scanner")
    nib.save(img, fname)
    return fname


def save_motion_params(
    fname,
    transforms,
    positions,
    slice_shape,
    resolution_slice,
    slice_thickness=None,
    resolution_recon=None,
    write_csv=True,
):
    """Save per-slice rigid motion parameters in a single canonical format.

    The SAME representation is written by ``generate_stack_database`` (GT motion),
    ``inference`` (predicted motion) and ``eval`` (both), so two files can be
    loaded and compared directly. The ``points`` field is the anchor-point
    representation used by the model's ``point_loss``, so the point-loss MSE
    between e.g. two model versions (or GT vs prediction) is simply::

        gt = np.load("..._motion-gt.npz")
        pr = np.load("..._motion-pred.npz")
        point_mse = ((pr["points"] - gt["points"]) ** 2).mean()

    Args:
        fname: output path ending in ``.npz``. A sibling ``.csv`` (human-readable
            per-slice 6-DOF + anchor points) is also written when ``write_csv``.
        transforms: ``RigidTransform`` or per-slice matrix (N, 3, 4) / (N, 4, 4)
            tensor/array at resolution 1 -- the convention shared by the scanner's
            ``transforms_gt`` and the model's ``transforms_pred``/``transforms_gt``.
        positions: (N, 2) tensor/array; column 1 holds the stack id of each slice.
        slice_shape: (sx, sy) slice-grid size, as passed to ``mat2point``.
        resolution_slice: in-plane slice resolution ``rs``, as passed to ``mat2point``.
        slice_thickness, resolution_recon: stored as metadata (optional).
    """
    if isinstance(transforms, RigidTransform):
        mat = transforms.matrix(trans_first=True)
    else:
        mat = torch.as_tensor(transforms).float()
        if mat.shape[-1] == 6:  # axis-angle params -> matrix
            mat = RigidTransform(mat).matrix(trans_first=True)
    mat = mat[:, :3, :4].detach().float().cpu()  # (N, 3, 4), resolution 1

    sx, sy = int(slice_shape[0]), int(slice_shape[1])
    rs = float(resolution_slice)

    # anchor points (point_loss space) and human-readable 6-DOF (Euler, degrees).
    # Both are pure-torch / CPU-safe (the axis-angle op is CUDA-only).
    points = mat2point(mat, sx, sy, rs).detach().cpu()  # (N, 9)
    euler = mat2euler(mat).detach().cpu()  # (N, 6): tx, ty, tz, rx, ry, rz (deg)

    positions = torch.as_tensor(positions).detach().cpu()
    stack_ids = positions[:, 1].round().to(torch.int64).numpy()

    fname = Path(fname)
    fname.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        fname,
        matrix=mat.numpy().astype(np.float32),
        points=points.numpy().astype(np.float32),
        euler=euler.numpy().astype(np.float32),
        positions=positions.numpy().astype(np.float32),
        stack_ids=stack_ids.astype(np.int64),
        slice_shape=np.array([sx, sy], dtype=np.int64),
        resolution_slice=np.float32(rs),
        slice_thickness=np.float32(
            slice_thickness if slice_thickness is not None else np.nan
        ),
        resolution_recon=np.float32(
            resolution_recon if resolution_recon is not None else np.nan
        ),
        trans_first=np.bool_(True),
    )
    if write_csv:
        import pandas as pd

        e = euler.numpy()
        p = points.numpy()
        df = pd.DataFrame(
            {
                "stack_id": stack_ids,
                "slice_in_scan": np.arange(mat.shape[0]),
                "trans_x": e[:, 0],
                "trans_y": e[:, 1],
                "trans_z": e[:, 2],
                "rot_x_deg": e[:, 3],
                "rot_y_deg": e[:, 4],
                "rot_z_deg": e[:, 5],
                **{f"p{j}": p[:, j] for j in range(9)},
            }
        )
        df.to_csv(fname.with_suffix(".csv"), index=False)
    return fname


def load_stack(f, device):
    img = nib.load(f)
    slices = img.get_fdata().astype(np.float32).transpose(2, 1, 0)
    d, h, w = slices.shape
    slices = torch.tensor(slices, device=device).unsqueeze(1)
    res = img.header["pixdim"][1:4]
    affine = img.affine
    if np.any(np.isnan(affine)):
        affine = img.get_qform()

    R = affine[:3, :3]
    negative_det = np.linalg.det(R) < 0
    T = affine[:3, -1:]
    R = R @ np.linalg.inv(np.diag(res))

    T0 = np.array([(w - 1) / 2 * res[0], (h - 1) / 2 * res[1], 0])
    T = np.linalg.inv(R) @ T + T0.reshape(3, 1)

    tz = (
        torch.arange(0, torch.tensor(d), device=device, dtype=torch.float32) * res[2]
        + T[2].item()
    )
    tx = torch.ones_like(tz) * T[0].item()
    ty = torch.ones_like(tz) * T[1].item()
    t = torch.stack((tx, ty, tz), -1).view(-1, 3, 1)
    R = torch.tensor(R, device=device).unsqueeze(0).expand(d, -1, -1).clone()

    if negative_det:
        slices = torch.flip(slices, (-1,))
        # mask = torch.flip(mask, (-1,))
        t[:, 0, -1] *= -1
        R[:, :, 0] *= -1

    transforms = RigidTransform(
        torch.cat((R, t), -1).to(device).to(torch.float32), trans_first=True
    )

    # Reset the origin
    transform_ax = transforms.axisangle().clone()
    transform_ax[:, :-1] = 0  # Zero out rotations and X/Y translations
    # Adjust Z translation relative to the stack mean [11]
    transform_ax[:, -1] -= transform_ax[:, -1].mean()

    # filter slices with no brain
    # valid_idxs = torch.where((slices > 0).abs().sum(dim=(1, 2, 3)) > 10)[0]
    # slices = slices[valid_idxs]
    # transform_ax = transform_ax[valid_idxs]

    final_rigid_transform = RigidTransform(transform_ax)

    res_s = (res[0] + res[1]) / 2
    s_thick = res[2]

    return slices, final_rigid_transform, res_s, s_thick
