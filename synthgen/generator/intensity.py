import torch
from pathlib import Path
import numpy as np
from synthgen.utils.image_reading import TorchIOReader


class ImageFromSeeds:

    def __init__(
        self,
        min_subclusters: int,
        max_subclusters: int,
        seed_labels: list[int],
        generation_classes: list[int],
        meta_labels: list[int] = [1, 2, 3, 4],
        empty_background_prob: float = 0.2,
        skullstripprob: float = 0.1,
    ):
        """

        Args:
            min_subclusters (int): Minimum number of subclusters to use.
            max_subclusters (int): Maximum number of subclusters to use.
            seed_labels (Iterable[int]): Iterable with all possible labels
                that can occur in the loaded seeds. Should be a unique set of
                integers starting from [0, ...]. 0 is reserved for the background,
                that will not have any intensity generated.
            generation_classes (Iterable[int]): Classes to use for generation.
                Seeds with the same generation class will be generated with
                the GMM sampled from the same distribution. Should be the same length as seed_labels.
                (Ensures that all CSF/WM/GM regions, for example, have similar (not the same) GMMs).
            meta_labels (list[int], optional): Meta-labels numbers used.
                Defaults to [1, 2, 3, 4 (background)].
                Seeds are only used by the label-driven synthesis mode,
                which PRISM does not use (it samples intensities from real
                reference volumes).
            empty_background_prob (float, optional): Probability of generating
                an empty background. Defaults to 0.2.
            skullstripprob (float, optional): Probability of removing skull in the image (only if background is empty).
                Real probability is a product of empty_background_prob and skullstripprob. Defaults to 0.1.
        """
        self.min_subclusters = min_subclusters
        self.max_subclusters = max_subclusters
        self.empty_background_prob = empty_background_prob
        self.skullstripprob = skullstripprob
        try:
            assert len(set(seed_labels)) == len(seed_labels)
        except AssertionError:
            raise ValueError("Parameter seed_labels should have unique values.")
        try:
            assert len(seed_labels) == len(generation_classes)
        except AssertionError:
            raise ValueError(
                "Parameters seed_labels and generation_classes should have the same lengths."
            )
        self.seed_labels = seed_labels
        self.generation_classes = generation_classes
        self.meta_labels = meta_labels
        self.loader = TorchIOReader()

    def load_seeds(self, seeds: dict[int, dict[int, Path]]) -> torch.Tensor:
        """Generate an intensity image from seeds.
        If seed_mapping is provided, it is used to
        select the number of subclusters to use for
        each meta label. Otherwise, the number of subclusters
        is randomly selected from a uniform discrete distribution
        between `min_subclusters` and `max_subclusters` (both inclusive).

        Args:

            seeds: Dictionary with the mapping `subcluster_number: {meta_label: seed_path}`.
            mlabel2subclusters: Mapping to use when defining how many subclusters to
                use for each meta-label. Defaults to None.

        Returns:
            torch.Tensor: Intensity image with the same shape as the seeds.
                Tensor dimensions are **(H, W, D)**. Values inside the tensor
                correspond to the subclusters, and are grouped by meta-label.
                `1-19: CSF, 20-29: GM, 30-39: WM, 40-49: Extra-cerebral`.
        """
        # Randomly select the number of subclusters to use
        # for each meta-label in the format {mlabel: n_subclusters}
        mlabel2subclusters = {
            meta_label: np.random.randint(
                self.min_subclusters, self.max_subclusters + 1
            )
            for meta_label in self.meta_labels
        }
        # load the first seed as the one corresponding to mlabel 1
        seed = self.loader(seeds[mlabel2subclusters[1]][1], "label").data

        for mlabel in self.meta_labels[1:]:
            new_seed = self.loader(
                seeds[mlabel2subclusters[mlabel]][mlabel], "label"
            ).data
            seed += new_seed

        return seed.long().squeeze(0)

    def sample_intensities(self, seeds: torch.Tensor) -> torch.Tensor:
        """Sample the intensities from the seeds.

        Args:
            seeds (torch.Tensor): Tensor with the seeds.

        Returns:
            tuple[torch.Tensor, dict]: Intensity image with the same shape as the seeds.
                Tensor dimensions are **(H, W, D)**. Values inside the tensor
                correspond to the intensities sampled from the GMM defined by the parameters.
                Dictionary with the GMM parameters used for generation.
        """
        # save seeds
        nlabels = max(self.seed_labels) + 1
        nsamp = len(self.seed_labels)

        # # Sample GMMs means and stds
        mus = 25 + 200 * torch.rand(nlabels, dtype=torch.float)

        sigmas = 5 + 10 * torch.rand(
            nlabels,
            dtype=torch.float,
        )

        # if there are seed labels from the same generation class
        # set their mean to be the same with some random perturbation
        if self.generation_classes != self.seed_labels:
            mus[self.seed_labels] = torch.clamp(
                mus[self.generation_classes]
                + 30
                * torch.randn(
                    nsamp,
                    dtype=torch.float,
                ),
                0,
                225,
            )
        # perform the sampling
        intensity_image = mus[seeds] + sigmas[seeds] * torch.randn(
            seeds.shape,
            dtype=torch.float,
        )
        intensity_image[intensity_image < 0] = 0

        # randomly set background and skull to zero
        if self.empty_background_prob > np.random.rand():
            intensity_image[seeds == 0] = 0
            if self.skullstripprob > np.random.rand():
                maxmetalabel = max(self.meta_labels)
                intensity_image[seeds >= maxmetalabel * 10] = 0

        return intensity_image
