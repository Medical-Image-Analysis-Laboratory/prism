from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

src = "synthgen/generator"

setup(
    name="synthgen",
    version="1.0.0",
    packages=find_packages(),
    description=(
        "PRISM: learned priors for robust slice-to-volume registration in "
        "ill-posed fetal MRI, together with the synthetic fetal-MRI stack "
        "generator it is trained on. Builds on SVoRT (Xu et al. 2022), "
        "Brain-ID (Liu et al. 2024) and FetalSynthSeg (Zalevskyi et al. 2024), "
        "implemented in PyTorch, TorchIO and MONAI."
    ),
    python_requires=">=3.10",
    ext_modules=[
        CUDAExtension(
            "slice_acq_cuda",
            [
                f"{src}/slice_acquisition/slice_acq_cuda.cpp",
                f"{src}/slice_acquisition/slice_acq_cuda_kernel.cu",
            ],
        ),
        CUDAExtension(
            "transform_convert_cuda",
            [
                f"{src}/transform/transform_convert_cuda.cpp",
                f"{src}/transform/transform_convert_cuda_kernel.cu",
            ],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
