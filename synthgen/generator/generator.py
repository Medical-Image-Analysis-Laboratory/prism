import os
from typing import Iterable

import numpy as np
import torch

from torch import Tensor
from torchio.transforms import Compose
from torchio import ScalarImage, LabelMap, Subject

from synthgen.generator.intensity import ImageFromSeeds


class SynthGen:
    def __init__(
        self,
        shape: Iterable[int],
        resolution: Iterable[float],
        intensity_generator: ImageFromSeeds,
        image_transforms: Compose,
    ):
        """
        Initialize the model with the given parameters.

        Args:
            shape: Shape of the output image.
            resolution: Resolution of the output image.
            device: Device to use for computation.
            intensity_generator: Intensity generator.
            image_transforms: Image transforms to apply.

        """
        self.shape = shape
        self.resolution = resolution
        self.intensity_generator = intensity_generator
        self.image_transforms = image_transforms

    def sample(
        self,
        image: Tensor | None,
        segmentation: LabelMap,
        seeds: dict | None,
        output_seeds: bool = False,
    ) -> Subject:
        """
        Generate a synthetic image from the input data.
        Supports both random generation and from a fixed genparams dictionary.

        Args:
            image: Image to use as intensity prior if required.
            segmentation: Segmentation to use as spatial prior.
            seeds: Seeds to use for intensity generation.
        Returns:
                subject: TorchIO Subject containing the generated image and segmentation.
                synth_params: Dictionary with the generation parameters used for this sample.
        """

        # 1. Generate domain randomized intensity output
        if seeds is not None:
            loaded_seeds = self.intensity_generator.load_seeds(seeds=seeds)

            output = self.intensity_generator.sample_intensities(
                seeds=loaded_seeds,
            )

            # turn output to torchio.ScalarImage
            output = ScalarImage(
                tensor=output.unsqueeze(0),  # Add batch and channel dimensions
                affine=segmentation.affine,
            )
        # or use the provided image as intensity prior
        else:
            if image is None:
                raise ValueError(
                    "If no seeds are passed, an image must be loaded to be used as intensity prior!"
                )
            # normalize the image from 0 to 255 to
            # match the intensity generator
            output = image
        subject_dict = {
            "image": output,
            "label": segmentation,
        }

        if output_seeds and seeds is not None:
            subject_dict["seeds"] = LabelMap(
                tensor=loaded_seeds.unsqueeze(0), affine=segmentation.affine
            )
        # combine the output and segmentation into a TorchIO subject to
        # ensure applied transformations are consistent
        subject = Subject(subject_dict)

        # 2. Apply main torchio transforms
        subject = self.image_transforms(subject)

        # check if volume or label is nan after transforms
        if torch.isnan(subject["image"].data).any():
            raise ValueError("Generated image contains NaN values after transforms!")

        return subject


if __name__ == "__main__":

    from synthgen.data.datasets import SynthDataset, TestDataset
    import time
    import nibabel as nib
    from torchio.transforms import (
        RandomAffineElasticDeformation,
        RandomAnisotropy,
        Compose,
        RandomBiasField,
        RandomGamma,
        RandomNoise,
        RandomMotion,
        RandomGhosting,
        RandomSpike,
        RandomBlur,
        RandomFlip,
    )

    # intensity generation: ~500ms

    # blur: 500ms
    blurtransf = RandomBlur(
        std=(0.4, 0.8),
        p=1,
    )

    # randaffineelastic: 23000ms
    spatial_deform = RandomAffineElasticDeformation(
        p=0.9,
        affine_first=False,
        affine_kwargs={
            "scales": 0.1,
            "degrees": 20,
            "translation": 5,
            "isotropic": False,
        },
        elastic_kwargs={
            "num_control_points": 7,
            "max_displacement": 8,
        },
    )

    # randflip: 50ms
    randflip = RandomFlip(
        axes=("L", "R"),
        flip_probability=0.5,
    )

    # 100ms
    resampletransf = RandomAnisotropy(
        axes=(0, 1, 2),
        downsampling=(1.5, 5),
        scalars_only=True,
        p=1.0,
    )

    # randgamma: 50ms
    gammatransf = RandomGamma(
        log_gamma=(-0.5, 0.5),
        p=0.8,
    )

    # biasfield: 50ms
    biasfield = RandomBiasField(coefficients=0.8, order=1, p=1)

    # MR artifacts: ~200ms
    randmotion = RandomMotion(
        degrees=10,
        translation=5,
        num_transforms=1,
        p=0.2,
    )

    # 300ms
    randghosting = RandomGhosting(
        num_ghosts=(1, 10),
        intensity=(0.1, 0.5),
        p=0.2,
    )

    # 500ms
    randspike = RandomSpike(
        num_spikes=(1, 3),
        intensity=0.3,
        p=0.2,
    )

    # 100ms
    noistransf = RandomNoise(
        mean=1.0,
        std=(0, 0.25),
        p=0.2,
    )

    scannerkwargs = {
        "resolution_slice_fac": 1.25,
        "resolution_slice_max": 1.5,
        "slice_thickness": [2.5, 3.5],
        "gap": [2.5, 3.5],
        "resolution_recon": 0.5,  # what's the SR used for stack sim
        "min_num_stack": 3,
        "max_num_stack": 3,
        "max_num_slices": 450,
        "noise_sigma": [0, 0.03],
        "TR": [1, 2],
        "prob_void": 0.03,
        "prob_motion": 0.00,
        "slice_size": 256,
        "restrict_transform": False,
        "txy": 3,
    }

    generator = SynthGen(
        shape=(256, 256, 256),
        resolution=(0.5, 0.5, 0.5),
        intensity_generator=ImageFromSeeds(
            min_subclusters=1,
            max_subclusters=6,
            seed_labels=list(range(0, 90)),
            generation_classes=list(range(0, 90)),
            meta_labels=list(range(1, 5)),
            empty_background_prob=1,  # 50% chance to have empty background
            skullstripprob=0.0,  # 90% chance to have skull
        ),
        # image_transforms=None,
        image_transforms=Compose(
            [
                # smooth synth data
                # spatial
                blurtransf,
                spatial_deform,
                # randflip,
                resampletransf,
                # intensity
                gammatransf,
                biasfield,
                # # MR artifacts | AFTER MOTION AND SPATIAL DEFORMATIONS
                # randmotion,
                # randghosting,
                # randspike,
                # last one so it's not cancelled by other transforms
                noistransf,
                # svort_transform,
            ]
        ),
    )

    # Smoke test on the small demo BIDS folder shipped with the repo
    # (override with PRISM_DEMO_BIDS=/path/to/bids).
    demo_bids = os.environ.get("PRISM_DEMO_BIDS", "data")
    dataset = SynthDataset(
        bids_path=demo_bids,
        generator=generator,
        seed_path=None,
        sub_list=None,
        load_image=True,
        image_as_intensity=True,
    )

    # from torchio import SubjectsLoader
    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        dataset,
        multiprocessing_context="spawn",
        batch_size=2,
        shuffle=False,
        num_workers=4,
    )

    print(f"Dataset length: {len(dataset)}")
    starttime = time.time()
    # for i in range(1):
    for batch in train_loader:
        sample = batch
        # for

        print(f"Type of sample: {type(sample)}")
        print(f"Generated sample in {time.time() - starttime:.2f} seconds")

        image = sample["image"]  # .data
        seg = sample["label"]  # .data
        print(f"Type of image: {type(image)}")
        # print(image.keys())
        print(f"Image shape: {image['data'].shape} Seg shape: {seg['data'].shape}")
        print(f"Image dtype: {image['data'].dtype} Seg dtype: {seg['data'].dtype}")
        print(
            f"Image min/max: {image['data'].min().item():.2f}/{image['data'].max().item():.2f}"
        )
        print(f"Seg min/max: {seg['data'].min().item()}/{seg['data'].max().item()}")
        print(f"Image affine:\n{image['affine']} \nSeg affine:\n{seg['affine']}")
        print(f"Image device: {image['data'].device} Seg device: {seg['data'].device}")
        print(f'Shape of affine: {image["affine"].shape}')
        # save image
        nibimage = nib.Nifti1Image(
            image["data"].numpy().astype(np.float32)[0][0],
            affine=seg["affine"][0],
        )
        nib.save(nibimage, "sample_image.nii.gz")
        nibseg = nib.Nifti1Image(
            seg["data"].numpy().astype(np.int16)[0][0],
            affine=seg["affine"][0],
        )
        nib.save(nibseg, "sample_label.nii.gz")
