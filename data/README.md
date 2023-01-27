
### Install dependencies

`pip install -r requirements.txt`

### Download data

Training raw: https://storage.googleapis.com/lee-pacureanu_data-exchange_us-storage/xray-challenge/xpress-training-raw.h5

Training voxel labels: https://storage.googleapis.com/lee-pacureanu_data-exchange_us-storage/xray-challenge/xpress-training-voxel-labels.h5

Validation raw: https://storage.googleapis.com/lee-pacureanu_data-exchange_us-storage/xray-challenge/xpress-validation-raw.h5

Test raw: https://storage.googleapis.com/lee-pacureanu_data-exchange_us-storage/xray-challenge/xpress-test-raw.h5

### (Optional) Convert h5 files to Zarr

For high performance training, we will need to convert .h5 files to chunked .zarr files.

```
python convert_to_zarr_h5.py xpress-training-raw.h5 volumes/raw xpress-challenge.zarr volumes/training_raw
python convert_to_zarr_h5.py xpress-training-voxel-labels.h5 volumes/labels xpress-challenge.zarr volumes/training_gt_labels
python convert_to_zarr_h5.py xpress-validation-raw.h5 volumes/raw xpress-challenge.zarr volumes/validation_raw
python convert_to_zarr_h5.py xpress-test-raw.h5 volumes/raw xpress-challenge.zarr volumes/test_raw
```

### Visualize downloaded data

We'll use the `neuroglancer` script adapted from https://github.com/funkelab/funlib.show.neuroglancer/blob/master/scripts/neuroglancer to visualize the Zarr dataset.

Training dataset: `./neuroglancer --file xpress-challenge.zarr --datasets volumes/training_raw volumes/training_gt_labels`

Validation dataset: `./neuroglancer --file xpress-challenge.zarr --datasets volumes/validation_raw`

Test dataset: `./neuroglancer --file xpress-challenge.zarr --datasets volumes/test_raw`


