from pathlib import Path

from torch.autograd import Function
import torch

from synthgen.generator.cuda_ext import load_extension

_HERE = Path(__file__).resolve().parent
transform_convert_cuda = load_extension(
    "transform_convert_cuda",
    [
        _HERE / "transform_convert_cuda.cpp",
        _HERE / "transform_convert_cuda_kernel.cu",
    ],
)


class Axisangle2MatFunction(Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, axisangle):
        outputs = transform_convert_cuda.axisangle2mat_forward(axisangle)
        mat = outputs[0]
        ctx.save_for_backward(axisangle)
        return mat

    @staticmethod
    def backward(ctx, grad_mat):
        axisangle = ctx.saved_variables[0]
        outputs = transform_convert_cuda.axisangle2mat_backward(grad_mat, axisangle)
        grad_axisangle = outputs[0]
        return grad_axisangle


class Mat2AxisangleFunction(Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, mat):
        outputs = transform_convert_cuda.mat2axisangle_forward(mat)
        axisangle = outputs[0]
        ctx.save_for_backward(mat)
        return axisangle

    @staticmethod
    def backward(ctx, grad_axisangle):
        mat = ctx.saved_variables[0]
        outputs = transform_convert_cuda.mat2axisangle_backward(mat, grad_axisangle)
        grad_mat = outputs[0]
        return grad_mat


def axisangle2mat(axisangle):
    return Axisangle2MatFunction.apply(axisangle)


def mat2axisangle(mat):
    return Mat2AxisangleFunction.apply(mat)
