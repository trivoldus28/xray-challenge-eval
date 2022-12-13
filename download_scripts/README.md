
### Install dependencies

`pip install -r requirements.txt`

### Download data

Run the provided scripts, e.g.,:

`python download_training_raw.py` and `python download_training_gt.py`

These will download the datasets to a new Zarr file "../xpress-challenge.zarr".

### Visualize downloaded data

We'll use the `neuroglancer` script adapted from https://github.com/funkelab/funlib.show.neuroglancer/blob/master/scripts/neuroglancer.

Training dataset: `./neuroglancer --file xpress-challenge.zarr --datasets volumes/training_raw volumes/training_gt_labels`

Validation dataset: `./neuroglancer --file xpress-challenge.zarr --datasets volumes/validation_raw`

Test dataset: `./neuroglancer --file xpress-challenge.zarr --datasets volumes/test_raw`
