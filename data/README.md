# Demo data

Three high-resolution fetal brain reference volumes (+ tissue segmentations)
from a public spatio-temporal atlas, at gestational weeks 21, 30 and 38, so the
simulation pipeline can be exercised without downloading anything.
`participants.csv` assigns them to the `train` / `val` / `test` splits, in the
two-column format (`participant_id,splits`) every config expects.

The volumes are 256^3 at 0.5 mm; the simulator expects them on the grid the
generator is configured for (128^3 at 1 mm), so resample them first:

```bash
python scripts/resample_crop_pad.py \
    --bids-path data --out-dir /tmp/demo_1mm \
    --res 1.0 --target-size 128 128 128 \
    --image-patterns T2w --label-patterns dseg
cp data/participants.csv /tmp/demo_1mm/
```

Simulate a scan of stacks from them (`simrealbad` = the hardest split: 1-3
stacks, heavy artefacts), then register it with PRISM:

```bash
PRISM_TRAIN_DATA=/tmp/demo_1mm python -m synthgen.generate_stack_database \
    experiment=datagen/simrealbad output_dir=/tmp/demo_stacks \
    data.val_split=train data.img_suffix=T2w data.seg_suffix=dseg logger=csv

python scripts/predict_bids.py -i /tmp/demo_stacks -o /tmp/demo_pred \
    -w weights/prism_weights.pth
```

The simulated dataset carries the ground-truth poses
(`*_motion.npz`), the reference volume and the segmentation under
`derivatives/`, so the predicted motion can be scored directly — that is what
`analysis/build_metrics.py` does for the test sets of the paper.

This is *not* the training set of the paper: PRISM was trained on 229 subjects
and tested on 162, drawn from several public datasets and atlases listed in the
supplementary material.
