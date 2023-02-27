Training the baseline model. The model is a U-net predicting affinity between pixels as implemented in [Funke et al](https://ieeexplore.ieee.org/abstract/document/8364622/) and refined for x-ray segmentation in [Kuan et al](https://www.nature.com/articles/s41593-020-0704-9). In addition to nearest neighbor affinities, we also add [long-range affinities prediction](https://arxiv.org/abs/1706.00120) as an auxiliary task to improve performance of the model.

### Install dependencies

`pip3 install -r requirements.txt`

Then install JAX:
`pip install --upgrade jax[cuda]==0.3.25 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html`

### Prepare training data

We'll assume that the dataset is already downloaded to "../../data/xpress-challenge.zarr"

#### Add labels mask

`python3 scripts/add_labels_mask.py ../../data/xpress-challenge.zarr`

This adds `volumes/training_gt_labels_mask` to the Zarr volume, denoting the valid region-of-interest (ROI) to train.

#### Add unlabeled mask

`python3 scripts/add_unlabeled_mask.py ../../data/xpress-challenge.zarr --include_zero 0 --output_ds_label volumes/training_gt_unlabeled_mask`

This adds `unlabeled_mask`, a binary mask that excludes the empty space between labeled neurons from training.

#### Visualize masks with Neuroglancer

`../../data/neuroglancer --file ../../data/xpress-challenge.zarr --datasets volumes/training_raw volumes/training_gt_labels volumes/training_gt_labels_mask volumes/training_gt_unlabeled_mask`

#### Run training

We use `gunpowder` for sampling and augmenting training batches; an in-depth tutorial of `gunpowder` can be found at http://funkey.science/gunpowder/. Our reference training pipeline is in `reference_setup/`:

```
cd reference_setup/
CUDA_VISIBLE_DEVICES=0 bash train.sh
```

This will train the model to 400k iterations using 24 threads and GPU device 0, saving a checkpoint for every 20k iterations. Make sure that you have a working NVIDIA GPU (run `nvidia-smi` to check), and a working JAX library (follow the installation guide at https://github.com/google/jax if `pip3 install -r requirements.txt` did not install it for you).
