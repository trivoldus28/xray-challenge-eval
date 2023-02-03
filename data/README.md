
Download links and data prep scripts for the XPRESS datasets.

## Downloading datasets

### Install dependencies for the below scripts

`pip3 install -r requirements.txt`

### Download image data

Training raw: [download](https://github.com/htem/xpress-challenge-files/releases/download/v1.0/xpress-training-raw.h5)

Training voxel labels: [download](https://github.com/htem/xpress-challenge-files/releases/download/v1.0/xpress-training-voxel-labels.h5)

Validation raw: [download](https://github.com/htem/xpress-challenge-files/releases/download/v1.0/xpress-validation-raw.h5)

Test raw: [download](https://github.com/htem/xpress-challenge-files/releases/download/v1.0/xpress-test-raw.h5)

### Download ground-truth skeletons

The skeletons are stored as [NetworkX](https://networkx.org/) graphs in [.npz](https://numpy.org/doc/stable/reference/generated/numpy.savez.html) files. Please see the ../eval/eval.py script to see how to properly read these skeleton files.

Training skeletons: [download](https://github.com/htem/xpress-challenge-files/releases/download/v1.0/XPRESS_training_skels.npz)

Validation skeletons: [download](https://github.com/htem/xpress-challenge-files/releases/download/v1.0/XPRESS_validation_skels.npz)

### (Optional) Convert h5 files to Zarr

For high performance training in the baseline model, we will need to convert .h5 files to chunked .zarr files.

```
python3 convert_to_zarr_h5.py xpress-training-raw.h5 volumes/raw xpress-challenge.zarr volumes/training_raw
python3 convert_to_zarr_h5.py xpress-training-voxel-labels.h5 volumes/labels xpress-challenge.zarr volumes/training_gt_labels
python3 convert_to_zarr_h5.py xpress-validation-raw.h5 volumes/raw xpress-challenge.zarr volumes/validation_raw
python3 convert_to_zarr_h5.py xpress-test-raw.h5 volumes/raw xpress-challenge.zarr volumes/test_raw
```

### (Optional) Convert h5 files to image stack

If your workflow requires stack of 2D images, you'll need to extract them from the .h5 files. TODO: we may provide an example script later.

### Visualize downloaded data

We'll use the `neuroglancer` script adapted from https://github.com/funkelab/funlib.show.neuroglancer/blob/master/scripts/neuroglancer to visualize the Zarr dataset.

Training dataset: `./neuroglancer --file xpress-challenge.zarr --datasets volumes/training_raw volumes/training_gt_labels`

Validation dataset: `./neuroglancer --file xpress-challenge.zarr --datasets volumes/validation_raw`

Test dataset: `./neuroglancer --file xpress-challenge.zarr --datasets volumes/test_raw`
