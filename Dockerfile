# PRISM -- Learning Priors for Robust Slice-to-Volume Registration in Fetal MRI
#
# A CUDA *devel* base image is required, not a runtime one: the forward model
# (slice acquisition and its adjoint) is a CUDA extension compiled with nvcc at
# install time.
#
# Build (from the repository root):
#   docker build -t prism .
#
# Run inference on a BIDS folder of brain-masked stacks:
#   docker run --gpus all --rm \
#       -v /data/mybids:/data:ro \
#       -v /weights:/weights:ro \
#       -v /output:/out \
#       prism \
#       scripts/predict_bids.py -i /data -o /out -w /weights/prism_weights.pth
#
# Interactive shell:
#   docker run --gpus all --rm -it --entrypoint bash prism
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Compile the extensions for the common data-center / workstation GPUs.
    # Trim or extend this list to speed up the build for your own hardware
    # (7.0 V100, 7.5 T4/RTX20xx, 8.0 A100, 8.6 A40/RTX30xx, 8.9 L40/RTX40xx).
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9+PTX" \
    # Where the JIT fallback of synthgen/generator/cuda_ext.py caches builds.
    SYNTHGEN_EXT_CACHE=/tmp/synthgen_ext \
    MPLBACKEND=Agg

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip \
        git build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python3

WORKDIR /opt/prism

# PyTorch first, from the CUDA 11.8 index, so the extensions build against it.
RUN python -m pip install --upgrade pip setuptools==69.5.1 wheel \
    && python -m pip install \
        torch==2.3.0 torchvision==0.18.0 \
        --index-url https://download.pytorch.org/whl/cu118

# Then the rest of the dependencies (cached unless requirements.txt changes).
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

# Finally the package itself, which compiles the two CUDA extensions.
COPY . .
# `import torch` first: it is what puts libtorch on the dynamic loader path,
# so the extensions cannot be imported before it.
RUN python -m pip install --no-build-isolation -e . \
    && python -c "import torch, slice_acq_cuda, transform_convert_cuda; \
print('CUDA extensions built for', torch.__version__)"

ENV PROJECT_ROOT=/opt/prism
ENTRYPOINT ["python"]
CMD ["scripts/predict_bids.py", "--help"]
