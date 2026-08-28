import torch
import torch.nn as nn
import torch.nn.functional as F
from synthgen.generator.transform import axisangle2mat
from synthgen.generator.slice_acquisition import (
    slice_acquisition,
    slice_acquisition_adjoint,
)


def dot(x, y):
    return torch.dot(x.flatten(), y.flatten())


def CG(A, b, x0, n_iter):
    if x0 is None:
        x = 0
        r = b
    else:
        x = x0
        r = b - A(x)
    p = r
    dot_r_r = dot(r, r)
    i = 0
    while True:
        if dot_r_r < 1e-10:
            return x
        Ap = A(p)
        alpha = dot_r_r / dot(p, Ap)
        x = x + alpha * p  # alpha ~ 0.1 - 1
        i += 1
        if i == n_iter:
            if torch.any(torch.isnan(x)):
                raise ValueError("NaN detected in CG solution at final iteration.")
            return x
        r = r - alpha * Ap
        dot_r_r_new = dot(r, r)
        p = r + (dot_r_r_new / dot_r_r) * p
        dot_r_r = dot_r_r_new


def PSFreconstruction(
    transforms, slices, slices_mask, vol_mask, params
) -> torch.Tensor:
    return slice_acquisition_adjoint(
        transforms,
        params["psf"],
        slices,
        slices_mask,
        vol_mask,
        params["volume_shape"],
        params["res_s"] / params["res_r"],
        params["interp_psf"],
        True,
    )


class SRR(nn.Module):
    def __init__(self, n_iter=10, use_CG=False, alpha=0.5, beta=0.02, delta=0.1):
        super().__init__()
        self.n_iter = n_iter
        self.alpha = alpha
        self.beta = beta * delta * delta
        self.delta = delta
        self.use_CG = use_CG

    def forward(
        self,
        theta,
        slices,
        volume,
        params,
        mu=0,
        slice_qa=None,
        z=None,
        vol_mask=None,
        slices_mask=None,
    ):
        if len(theta.shape) == 2:
            transforms = axisangle2mat(theta)
        else:
            transforms = theta
        p = slice_qa if slice_qa is not None else None

        A = lambda x: self.A(transforms, x, vol_mask, slices_mask, params)
        At = lambda x: self.At(transforms, x, slices_mask, vol_mask, params)
        AtA = lambda x: self.AtA(transforms, x, vol_mask, slices_mask, p, params, mu, z)

        x = volume
        y = slices

        b = At(y * p if p is not None else y)
        if z is not None:
            b = b + mu * z
        x = CG(AtA, b, volume, self.n_iter)

        return F.relu(x, False)

    def A(self, transforms, x, vol_mask, slices_mask, params):
        return slice_acquisition(
            transforms,
            x,
            vol_mask,
            slices_mask,
            params["psf"],
            params["slice_shape"],
            params["res_s"] / params["res_r"],
            False,
            params["interp_psf"],
        )

    def At(self, transforms, x, slices_mask, vol_mask, params):
        return slice_acquisition_adjoint(
            transforms,
            params["psf"],
            x,
            slices_mask,
            vol_mask,
            params["volume_shape"],
            params["res_s"] / params["res_r"],
            params["interp_psf"],
            False,
        )

    def AtA(self, transforms, x, vol_mask, slices_mask, p, params, mu, z):
        slices = self.A(transforms, x, vol_mask, slices_mask, params)
        if p is not None:
            slices = slices * p
        vol = self.At(transforms, slices, slices_mask, vol_mask, params)
        if z is not None:
            vol = vol + mu * x
        return vol
