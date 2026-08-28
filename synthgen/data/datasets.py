from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any

import torch
from torchio.transforms import RescaleIntensity
from torchio.transforms import Compose
from torchio import Subject

from synthgen.generator.generator import SynthGen
from synthgen.utils.image_reading import TorchIOReader
from synthgen.utils.scanutils import (
    get_2d_polynomial_basis,
    validateafterscandims,
    validatebeforescandims,
    percentile_normalize,
)
from synthgen.generator.scanner import get_PSF, Scanner
from synthgen.utils.io import load_stack, save_stack
from synthgen.generator.transform import RigidTransform

from torchio.data import ScalarImage
from torch.utils.data import Dataset
import numpy as np


class BIDSDataset(Dataset):
    """Abstract class defining a dataset for loading BIDS datsa."""

    def __init__(
        self,
        bids_path: str,
        sub_list: list[str] | None,
        img_suffix: str = "T2w",
        seg_suffix: str = "dseg",
        load_segmentations: bool = True,
    ):
        """
        Args:
            bids_path: Path to the bids folder with the data.
            sub_list: List of the subjects to use. If None, all subjects are used.


        """
        super().__init__()

        self.img_suffix = img_suffix
        self.seg_suffix = seg_suffix
        self.load_segmentations = load_segmentations
        self.bids_path = Path(bids_path)
        self.subjects = sorted(self.find_subjects(sub_list))
        self.sub_ses = [
            (x, y) for x in self.subjects for y in self._get_ses(self.bids_path, x)
        ]

        self.loader = TorchIOReader()

        self.img_paths, img_subject_sessions = self._load_bids_path(
            self.bids_path, self.img_suffix
        )
        if load_segmentations:
            # load the segmentation paths and ensure they are in the same order as images
            self.segm_paths, segm_subject_sessions = self._load_bids_path(
                self.bids_path, self.seg_suffix
            )

            if len(self.img_paths) != 0 and len(self.segm_paths) != 0:
                # ensure that image and segmentation paths are in the same order and have only overlapping subjects
                overlapping_subjects = set(img_subject_sessions) & set(
                    segm_subject_sessions
                )
                if len(overlapping_subjects) == 0:
                    raise ValueError(
                        f"No overlapping subjects found between image and segmentation paths."
                    )

                # realign img_paths and segm_paths to the filtered self.sub_ses
                img_map = {
                    sess: path
                    for sess, path in zip(img_subject_sessions, self.img_paths)
                }
                seg_map = {
                    sess: path
                    for sess, path in zip(segm_subject_sessions, self.segm_paths)
                }
                self.img_paths = [img_map[key] for key in overlapping_subjects]
                self.segm_paths = [seg_map[key] for key in overlapping_subjects]
                self.sub_ses = list(overlapping_subjects)
                self.subjects = list([x[0] for x in self.sub_ses])

        else:
            self.segm_paths = None
            self.sub_ses = img_subject_sessions
            self.subjects = list([x[0] for x in img_subject_sessions])

    def find_subjects(self, sub_list):
        """Subjects of the BIDS folder, restricted to ``sub_list`` if given."""
        subj_found = [x.name for x in Path(self.bids_path).glob("sub-*")]
        if sub_list is None:
            return subj_found
        return list(set(subj_found) & set(sub_list))

    def _sub_ses_string(self, sub, ses):
        return f"{sub}_{ses}" if ses is not None else sub

    def _sub_ses_idx(self, idx):
        sub, ses = self.sub_ses[idx]
        return self._sub_ses_string(sub, ses)

    def _get_ses(self, bids_path, sub):
        """Get the session names for the subject."""
        sub_path = bids_path / sub
        ses_dir = [x for x in sub_path.iterdir() if x.is_dir()]
        ses = []
        for s in ses_dir:
            if "anat" in s.name:
                ses.append(None)
            else:
                ses.append(s.name)

        return sorted(ses, key=lambda x: x or "")

    def _get_pattern(self, sub, ses, suffix, extension=".nii.gz"):
        """Get the pattern for the file name."""
        if ses is None:
            return f"{sub}/anat/{sub}*_{suffix}{extension}"
        else:
            return f"{sub}/{ses}/anat/{sub}_{ses}*_{suffix}{extension}"

    def _load_bids_path(self, path, suffix):
        """
        "Check that for a given path, all subjects have a file with the provided suffix
        """
        files_paths = []
        subject_sessions = []
        for sub, ses in self.sub_ses:

            pattern = self._get_pattern(sub, ses, suffix)
            files = list(path.glob(pattern))
            if len(files) == 0:
                print(
                    f"No files found for requested subject {sub} in {path} "
                    f"({pattern} returned nothing)"
                )
                continue
            elif len(files) > 1:
                raise ValueError(
                    f"Multiple files found for requested subject {sub} in {path} "
                    f"({pattern} returned {files}). "
                    "Please ensure that there is only one file per subject/session."
                )
            files_paths.append(files[0])
            subject_sessions.append((sub, ses))

        return files_paths, subject_sessions

    def __len__(self):
        return len(self.sub_ses)

    def __getitem__(self, idx):
        raise NotImplementedError(
            "This method should be implemented in the child class."
        )


class TestDataset(BIDSDataset):
    """Dataset class for loading images offline.
    Used to load test/validation data.

    Use the `transforms` argument to pass additional processing steps
    (scaling, resampling, cropping, etc.).
    """

    def __init__(
        self,
        bids_path: str,
        sub_list: list[str] | None,
        transforms: Compose | None = None,
        img_suffix: str = "T2w",
        seg_suffix: str = "dseg",
        load_segmentations: bool = True,
    ):
        """
        Args:
            bids_path: Path to the bids folder with the data.
            sub_list: List of the subjects to use. If None, all subjects are used.
            transforms: Compose object with the transformations to apply.
                Default is None, no transformations are applied.

        !!! Note
            We highle recommend using the `transforms` arguments with at
            least the re-oriented transform to RAS and the intensity scaling
            to `[0, 1]` to ensure the data consistency.

            A torchio ``Compose`` of ToOrientation(RAS) -> Resample ->
            CropOrPad -> RescaleIntensity is a reasonable default.
        """
        super().__init__(
            bids_path,
            sub_list,
            img_suffix,
            seg_suffix,
            load_segmentations,
        )
        self.transforms = transforms

    def _load_data(self, idx):
        # load the image and segmentation
        image = self.loader(self.img_paths[idx], "intensity")
        segm = (
            self.loader(self.segm_paths[idx], "label")
            if self.load_segmentations
            else None
        )
        if len(image.shape) == 3:
            # add channel dimension
            image = image.unsqueeze(0)
            segm = segm.unsqueeze(0) if segm is not None else None
        elif len(image.shape) != 4:
            raise ValueError(f"Expected 3D or 4D image, got {len(image.shape)}D image.")

        # transform name into a single string otherwise collate fails
        name = self.sub_ses[idx]
        name = self._sub_ses_string(name[0], ses=name[1])

        subjdict = {"image": image, "name": name}
        if segm is not None:
            subjdict["label"] = segm

        subject = Subject(subjdict)

        return subject

    def __getitem__(self, idx) -> dict:
        """
        Returns:
            Dictionary with the `image` , `label` and the `name`
                keys. `image` and `label` are  `torch.float32`
                [`monai.data.meta_tensor.MetaTensor`](https://docs.monai.io/en/stable/data.html#metatensor)
                instances  with dimensions `(1, H, W, D)` and `name` is a string
                of a format `sub_ses` where `sub` is the subject name
                and `ses` is the session name.


        """
        subject = self._load_data(idx)

        if self.transforms:
            subject = self.transforms(subject)

        subject["label"].set_data(
            subject["label"].data.long()
        )  # ensure label is long tensor
        return subject

    def reverse_transform(self, subject: Subject) -> Subject:
        """Reverse the transformations applied to the data.

        Args:
            subject: Subject instance with the `image` and `label` attributes,
                like the one returned by the `__getitem__` method.

        Returns:
            Subject instance with the `image` and `label` attributes where
                the transformations are reversed.
        """
        if self.transforms:
            subject = self.transforms.inverse(subject)
        return subject


class SynthDataset(BIDSDataset):
    """Dataset class for generating/augmenting on-the-fly images."""

    def __init__(
        self,
        bids_path: str,
        generator: SynthGen | None,
        seed_path: str | None,
        sub_list: list[str] | None,
        load_image: bool = False,
        image_as_intensity: bool = False,
        img_suffix: str = "T2w",
        seg_suffix: str = "dseg",
        scanner: Scanner | None = None,
    ):
        """

        Args:
            bids_path: Path to the bids-formatted folder with the data.
            seed_path: Path to the folder with the seeds to use for
                intensity sampling. See `scripts/seed_generation.py`
                for details on the data formatting. If seed_path is None,
                the intensity  sampling step is skipped and the output image
                intensities will be based on the input image.
            generator: a class object defining a generator to use.
            sub_list: List of the subjects to use. If None, all subjects are used.
            load_image: If **True**, the image is loaded and passed to the generator,
                where it can be used as the intensity prior instead of a random
                intensity sampling or spatially deformed with the same transformation
                field as segmentation and the syntehtic image. Default is **False**.
            image_as_intensity: If **True**, the image is used as the intensity prior,
                instead of sampling the intensities from the seeds. Default is **False**.
        """
        super().__init__(bids_path, sub_list, img_suffix, seg_suffix)
        self.seed_path = Path(seed_path) if isinstance(seed_path, str) else None
        self.load_image = load_image
        self.generator = generator
        self.image_as_intensity = image_as_intensity
        # parse seeds paths
        if isinstance(self.seed_path, Path):
            if not self.seed_path.exists():
                raise FileNotFoundError(
                    f"Provided seed path {self.seed_path} does not exist."
                )
            else:
                self._load_seed_path()
        self.scanner = scanner

    def _load_seed_path(self):
        """Load the seeds for the subjects."""
        self.seed_paths = {
            self._sub_ses_string(sub, ses): defaultdict(dict)
            for (sub, ses) in self.sub_ses
        }
        avail_seeds = [
            int(x.name.replace("subclasses_", ""))
            for x in self.seed_path.glob("subclasses_*")
        ]
        min_seeds_available = min(avail_seeds)
        max_seeds_available = max(avail_seeds)
        for n_sub in range(
            min_seeds_available,
            max_seeds_available + 1,
        ):
            seed_path = self.seed_path / f"subclasses_{n_sub}"
            if not seed_path.exists():
                raise FileNotFoundError(
                    f"Provided seed path {seed_path} does not exist."
                )
            # load the seeds for the subjects for each meta label 1-4
            for i in self.generator.intensity_generator.meta_labels:
                files, __ = self._load_bids_path(seed_path, f"mlabel_{i}")
                for (sub, ses), file in zip(self.sub_ses, files):
                    sub_ses_str = self._sub_ses_string(sub, ses)
                    self.seed_paths[sub_ses_str][n_sub][i] = file

    def sample(
        self,
        idx,
    ) -> tuple[dict, dict]:
        """
        Retrieve a single item from the dataset at the specified index.

        Args:
            idx (int): The index of the item to retrieve.
        Returns:
            Dictionaries with the generated data and the generation parameters.
                First dictionary contains the `image`, `label` and the `name` keys.
                The second dictionary contains the parameters used for the generation.

        !!! Note
            The `image` is scaled to `[0, 1]` and oriented with the `label` to **RAS**
        """
        image = (
            self.loader(self.img_paths[idx], type="intensity")
            if self.load_image
            else None
        )
        segm = self.loader(self.segm_paths[idx], type="label")

        # transform name into a single string otherwise collate fails
        name = self.sub_ses[idx]
        name = self._sub_ses_string(sub=name[0], ses=name[1])

        # initialize seeds as dictionary
        # with paths to the seeds volumes
        # or None if image is to be used as intensity prior
        if self.seed_path is not None:
            seeds = self.seed_paths[name]
        if self.image_as_intensity:
            seeds = None

        subject = self.generator.sample(
            image=image,
            segmentation=segm,
            seeds=seeds,
        )

        if torch.rand(1).item() < 0.2:
            # mask image with the segmentation
            subject["image"].data = (
                subject["image"].data * (subject["label"].data > 0).float()
            )

        # remove background intensities
        subject["image"].data = (
            subject["image"].data * (subject["image"].data > 0.05).float()
        )

        if self.scanner is not None:
            # torchio.subject with keys 'image' and 'label' both with shape (1, H, W, D) and dict_keys(['data', 'affine', 'path', 'stem', 'type'])
            subject = self.unwrap_subject(subject)
            validatebeforescandims(subject)
            # move all tensors to gpu for all keys if they are tensor
            for key in subject.keys():
                if isinstance(subject[key], torch.Tensor):
                    subject[key] = (
                        subject[key].to("cuda:0", non_blocking=True).contiguous()
                    )

            subject = self.scanner.scan(subject)
            validateafterscandims(subject)
            # move all tensors to cpu for all keys if they are tensor
            for key in subject.keys():
                if isinstance(subject[key], torch.Tensor):
                    subject[key] = subject[key].cpu()

        subject["name"] = name
        return subject

    def unwrap_subject(self, subject: Subject):
        """
        Unwrap a TorchIO Subject into its components.

        Args:
            subject: TorchIO Subject to unwrap.

        Returns:
            image: Generated image tensor.
            segmentation: Segmentation tensor.
        """
        image = subject["image"].data
        segmentation = subject["label"].data
        image_affine = subject["image"]["affine"]
        threshold = subject.get("threshold", torch.tensor(0.0))
        vq = subject.get("scale", torch.tensor(1.0))
        resolution = np.sqrt(np.sum(image_affine[:3, :3] ** 2, axis=0))
        resolution = torch.tensor(resolution, dtype=torch.float32)[0]
        mask = subject["label"].data > 0
        # threshold = torch.quantile(image[mask], 0.01)
        volume = image.unsqueeze(0)

        # bias + gamma added as torchio transform

        subj_dict = {
            "volume": volume,  # add batch dimension
            "label": segmentation.unsqueeze(0),  # add batch dimension
            "resolution": float(resolution),
            "mask": mask.unsqueeze(0).float(),  # add batch dimension,
            "threshold": threshold,
            "scale": vq,
        }

        return subj_dict

    def __getitem__(self, idx) -> dict:
        """
        Retrieve a single item from the dataset at the specified index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            Dictionary with the `image`, `label` and the `name` keys.
                `image` and `label` are `torch.float32`
                [`monai.data.meta_tensor.MetaTensor`](https://docs.monai.io/en/stable/data.html#metatensor)
                and `name` is a string of a format `sub_ses` where `sub` is the subject name
                and `ses` is the session name.

        !!!Note
            The `image` is scaled to `[0, 1]` and oriented to **RAS** with the `label`.
        """
        data_out = self.sample(idx)
        return data_out


class InferenceStackDataset(BIDSDataset):
    """
    Reads multi-stack NIfTI files and prepares the input dictionary
    required by SRRLightningModule.predict_step, synthesizing necessary
    parameters (resolutions, PSFs, volume shape).
    """

    def __init__(
        self,
        bids_path: str,
        sub_list: list[str] | None,
        img_suffix: str = "T2w",
        seg_suffix: str | None = "dseg",
        target_resolution_recon: float = 1.0,
        slice_size: int = 128,
        device=None,
        reconstructed_volume_shape: tuple[int] = (128, 128, 128),
        drop_empty_slices: bool = False,
    ):
        """
        Initializes the reader with target reconstruction parameters.

        Args:
            target_resolution_recon: The desired isotropic resolution (mm)
                                     for the reconstructed volume (e.g., 0.8 mm [9]).
            slice_size: The expected in-plane pixel size (H x W) for input slices (e.g., 128 [10]).
            device: The target PyTorch device (e.g., 'cuda:0' or 'cpu').
            drop_empty_slices: If ``True``, near-empty slices (<=100 brain voxels)
                are removed from the model input. Default ``False`` to mirror
                training (``SynthDataset`` / ``Scanner`` never drop them) and to
                keep the predicted ``positions`` one-to-one with the ground-truth
                motion; empty slices are instead skipped when metrics are computed
                (see ``synthgen/inference.py`` / ``scripts/predict_bids.py``).
        """
        if seg_suffix is None:
            load_segmentations = False
        else:
            load_segmentations = True
        super().__init__(
            bids_path, sub_list, img_suffix, seg_suffix, load_segmentations
        )

        self.device = device if device else torch.device("cpu")
        self.resolution_recon = target_resolution_recon
        self.slice_size = slice_size
        self.drop_empty_slices = drop_empty_slices

        # Approximate target volume shape based on common FeTA sizes (~256^3 at 0.8mm)
        # In a real setup, this might be dynamically derived from the maximal extent of slices.
        self.approx_volume_shape = reconstructed_volume_shape

        # Mock default initial transform matrix (identity)
        self.initial_transform_mat = torch.eye(3, 4).to(self.device).float()

    def _load_bids_path(self, path, suffix):
        """
        "Check that for a given path, all subjects have a file with the provided suffix
        """
        # files_paths = []
        # subject_sessions = []
        subsesfiles = {}
        for sub, ses in self.sub_ses:

            pattern = self._get_pattern(sub, ses, suffix)
            files = list(path.glob(pattern))
            if len(files) == 0:
                print(
                    f"No files found for requested subject {sub} in {path} "
                    f"({pattern} returned nothing)"
                )
                continue

            # files_paths.append(files[0])
            # subject_sessions.append((sub, ses))
            subsesfiles[(sub, ses)] = files
        self.subsesfiles = subsesfiles
        files_paths = list(subsesfiles.values())
        self.subject_sessions = list(subsesfiles.keys())

        return files_paths, self.subject_sessions

    def __getitem__(self, idx):
        """
        Loads the subject at the given index, reading the NIfTI stack(s)
        and preparing the input dictionary for SVoRT inference.

        Args:
            idx: Index of the subject to load.
        Returns:
            Dictionary suitable for SRRLightningModule.predict_step.
        """
        stack_data = self.prepare_inference_data(
            [str(p) for p in self.subsesfiles[self.subject_sessions[idx]]]
        )
        stack_data["name"] = self._sub_ses_string(
            self.subject_sessions[idx][0], ses=self.subject_sessions[idx][1]
        )
        return stack_data

    @staticmethod
    def _stack_id_from_path(path: str, fallback: int) -> int:
        """Recover the scanner's stack id from a BIDS ``stackid-K`` entity.

        Simulated stacks are written as ``..._stackid-K_run-J_...`` by
        ``generate_stack_database`` (``K`` is the scanner's original, randomly
        assigned stack id). Using it here keeps the predicted ``positions[:, 1]``
        identical to the ground-truth motion, so GT and prediction can be joined
        on ``(stack_id, slice_index)`` with no matching. Real data without the
        entity falls back to a sequential id.
        """
        for tok in Path(path).name.split("_"):
            if tok.startswith("stackid-"):
                try:
                    return int(tok.split("-", 1)[1])
                except ValueError:
                    break
        return fallback

    def prepare_inference_data(self, stack_paths: List[str]) -> Dict[str, Any]:
        """
        Processes list of NIfTI stack paths into the SVoRT input dictionary.

        Args:
            stack_paths: List of absolute file paths to input NIfTI stacks.

        Returns:
            Dictionary suitable for SRRLightningModule.predict_step.
        """
        if not stack_paths:
            raise ValueError("Input stack_paths cannot be empty.")
        # Deterministic stack order (ascending scanner stack id when available),
        # so repeated runs and the saved motion are reproducible.
        stack_paths = sorted(
            stack_paths, key=lambda p: self._stack_id_from_path(p, 1 << 30)
        )
        subjname = Path(stack_paths[0]).stem.split("_")[:2]
        subjname = "_".join(subjname)
        all_stacks_data = []
        all_transforms_rigid = []
        all_positions_slice_idx = []

        # Parameters derived from the first stack (assumed to be consistent for slices)
        res_s_avg = None
        s_thick_avg = None

        s_thick_all = []
        res_s_all = []

        # TODO: ADD VALIDATION TO ENSURE CONSISTENCY ACROSS STACKS OF STACK RESOLUTIONS AND VOXEL SIZES
        # 1. Load, Aggregate, and Parameterize Stacks
        for stack_idx, path in enumerate(stack_paths):

            # Load stack data and initial transforms from NIfTI header [1, 2]
            slices, transforms_rigid, res_s, s_thick = load_stack(path, device="cuda")
            # padd all slices to expected size if needed [10]
            nslices, nchannels, h, w = slices.shape
            pad_h = max(0, self.slice_size - h)
            pad_w = max(0, self.slice_size - w)
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            slices = torch.nn.functional.pad(
                slices,
                (pad_left, pad_right, pad_top, pad_bottom),
                mode="constant",
                value=0,
            )
            # Ensure the first stack sets the resolutions for consistency
            if res_s_avg is None:
                res_s_avg = res_s
                s_thick_avg = s_thick

            N_slices = slices.shape
            # print(N_slices)
            # Aggregate slice image data (N_total x 1 x H x W)
            all_stacks_data.append(slices)

            # Aggregate transformations (RigidTransform object)
            all_transforms_rigid.append(transforms_rigid)

            # Create slice position indices (required for PE) [11, 12], matching
            # the convention the model was TRAINED with (see Scanner.scan):
            #   Index 0: slice index within stack, CENTERED as ``arange(N) - N//2``
            #   Index 1: the scanner's original stack id (from the ``stackid-K``
            #            filename entity; sequential fallback for real data)
            # Both mirror generate_stack_database's ground-truth motion, so the
            # predicted per-slice poses are directly comparable to the GT.
            n_stack = N_slices[0]
            slice_indices = (
                torch.arange(n_stack, device=self.device, dtype=torch.float32)
                - n_stack // 2
            )
            stack_id = torch.full_like(
                slice_indices, float(self._stack_id_from_path(path, stack_idx + 1))
            )

            # Combine slice index and stack index (N_slices, 2)
            positions = torch.stack((slice_indices, stack_id), dim=-1)
            all_positions_slice_idx.append(positions)
            s_thick_all.append(s_thick)
            res_s_all.append(res_s)

        # Check that for all stacks the slice resolutions and thicknesses are consistent
        res_s_all = torch.tensor(res_s_all)
        s_thick_all = torch.tensor(s_thick_all)

        if not torch.allclose(res_s_all, torch.tensor(res_s_avg), atol=1e-1):
            print(
                f"Warning: Inconsistent in-plane resolutions across stacks for {subjname}: {res_s_all.tolist()}"
            )
        if not torch.allclose(s_thick_all, torch.tensor(s_thick_avg), atol=1e-1):
            print(
                f"Warning: Inconsistent slice thicknesses across stacks for {subjname}: {s_thick_all.tolist()}"
            )
        # 2. Final Aggregation and Tensor Concatenation

        # Concatenate all slices, positions, and transforms
        final_stacks = torch.cat(all_stacks_data, 0)
        final_positions = torch.cat(all_positions_slice_idx, 0)
        final_transforms_rigid = RigidTransform.cat(all_transforms_rigid)

        # 3. PSF and Parameter Calculation (Mimicking Scanner.scan())

        # PSF for Reconstruction (relies on res_s, res_r, s_thick) [14]
        psf_rec = get_PSF(
            res_ratio=(
                res_s_avg / self.resolution_recon,
                res_s_avg / self.resolution_recon,
                s_thick_avg / self.resolution_recon,
            ),
            device=self.device,
        )

        # PSF for Acquisition (required for completeness, although unused in pure inference)
        # Assuming original volume resolution (res) is 1.0 based on typical SVoRT data [15]
        psf_acq = get_PSF(
            res_ratio=(res_s_avg / 1.0, res_s_avg / 1.0, s_thick_avg / 1.0),
            device=self.device,
        )

        # The initial transform fed to the model is usually the identity/initial estimate [13]
        # We use the combined initial transforms derived from the NIfTI headers [2].
        final_transforms_mat = final_transforms_rigid.matrix()

        # By default keep every slice: training never drops empties, and keeping
        # them makes `positions` line up one-to-one with the GT motion. Empty
        # slices are skipped later, at metric-computation time.
        if self.drop_empty_slices:
            final_stacks, final_positions, final_transforms_mat = (
                self.remove_empty_slices(
                    final_stacks, final_positions, final_transforms_mat
                )
            )

        # Robust per-session 99th-percentile scaling (x / hi) over the brain pixels,
        # matching the simulated-training normalization (Scanner.scan) so the model
        # sees the same input scale at train and inference. multiplicative=True forces
        # lo=0 to keep this a pure scale (see percentile_normalize / Scanner.scan). The
        # mask is taken from the loaded (brain-masked) stacks before scaling so it
        # stays consistent afterwards.
        stacks_mask = final_stacks > 0
        # final_stacks = percentile_normalize(
        #     final_stacks, stacks_mask, multiplicative=False
        # )
        final_stacks = (final_stacks - final_stacks.min()) / (
            final_stacks.max() - final_stacks.min() + 1e-8
        )
        H, W = final_stacks.shape[2], final_stacks.shape[3]
        bias_basis = get_2d_polynomial_basis(H, W, order=2, device=final_stacks.device)

        # 4. Construct the Final Input Dictionary
        inference_data_dict = {
            # Data tensors
            "stacks": final_stacks,
            "stacks_masks": stacks_mask.float(),  # brain mask (pre-normalization)
            "bias_basis": bias_basis,
            "positions": final_positions,
            "transforms": final_transforms_mat,  # Input initial estimates T0
            # Resolution and Geometry parameters (for internal model use)
            "resolution": self.resolution_recon,
            "resolution_slice": res_s_avg,
            "resolution_recon": self.resolution_recon,
            "slice_thickness": float(s_thick_avg),
            "slice_shape": (
                self.slice_size,
                self.slice_size,
            ),  # Assumed common size [10]
            "volume_shape": torch.Size(
                self.approx_volume_shape
            ),  # Target volume dimensions
            # PSF Tensors (required by params dictionary in model forward pass [16])
            "psf_rec": psf_rec,
            "psf_acq": psf_acq,
            "gap": float(s_thick_avg),
            "threshold": torch.tensor(0.0),  # Dummy threshold, unused in pure inference
        }

        return inference_data_dict

    def remove_empty_slices(self, stacks, positions, transforms):
        """
        Utility function to remove empty slices from the input stacks and corresponding metadata.

        Args:
            stacks: Tensor of shape (N_total, 1, H, W) containing all slices.
            positions: Tensor of shape (N_total, 2) containing slice and stack indices.
            transforms: RigidTransform object containing transformations for each slice.
        """
        # Identify non-empty slices based on the sum of voxel intensities
        non_empty_mask = (stacks > 0).sum(dim=[1, 2, 3]) > 100

        # Filter out empty slices
        filtered_stacks = stacks[non_empty_mask.to(stacks.device)]
        filtered_positions = positions[non_empty_mask.to(positions.device)]
        filtered_transforms = transforms[non_empty_mask.to("cpu")]

        return filtered_stacks, filtered_positions, filtered_transforms


def main():
    """Smoke test: load the first subject of a BIDS folder of acquired stacks.

    Point ``PRISM_STACKS_BIDS`` at a BIDS folder of brain-masked 2D stacks.
    """
    import os

    infds = InferenceStackDataset(
        bids_path=os.environ.get("PRISM_STACKS_BIDS", "data/stacks"),
        sub_list=None,
        target_resolution_recon=1.0,
        slice_size=128,
        device="cpu",
        reconstructed_volume_shape=[128, 128, 128],
        img_suffix=os.environ.get("PRISM_IMG_SUFFIX", "T2w"),
        seg_suffix=None,
    )

    res = infds[0]
    for k, v in res.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: {v.shape} - {v.dtype}")
        else:
            print(f"{k}: {v}")
    from synthgen.utils.scanutils import validateafterscandims

    validateafterscandims(res)


if __name__ == "__main__":
    main()
