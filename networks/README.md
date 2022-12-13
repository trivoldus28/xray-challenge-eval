
### Install dependencies

`pip install -r requirements.txt`

### Prepare training data

#### Download data

We'll assume that the dataset is already downloaded to "../xpress-challenge.zarr"

#### Add labels mask

`python scripts/add_labels_mask.py ../xpress-challenge.zarr`

This adds `volumes/training_gt_labels_mask` to the Zarr volume, denoting the valid region-of-interest (ROI) to train.

#### Add unlabeled mask

`python scripts/add_unlabeled_mask.py ../xpress-challenge.zarr --include_zero 0 --output_ds_label volumes/training_gt_unlabeled_mask_no_bg`

`python scripts/add_unlabeled_mask.py ../xpress-challenge.zarr --include_zero 1 --output_ds_label volumes/training_gt_unlabeled_mask`

This adds two `unlabeled_mask`s, denoting whether the empty space between neurons should be trained as background or not. For our reference setup we will not use the empty space for training.

#### Visualize masks with Neuroglancer

`../download_scripts/neuroglancer --file ../xpress-challenge.zarr --datasets volumes/training_raw volumes/training_gt_labels volumes/training_gt_labels_mask volumes/training_gt_unlabeled_mask volumes/training_gt_unlabeled_mask_no_bg`

#### Run training

We use `gunpowder` for sampling and augmenting training batches; a tutorial of `gunpowder` can be found at http://funkey.science/gunpowder/. Our reference training pipeline is in `reference_setup/`:

```
cd reference_setup/
CUDA_VISIBLE_DEVICES=0 bash train.sh
```
