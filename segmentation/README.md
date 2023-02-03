
Training and running our baseline for segmentation.

## Install dependencies

`pip install -r requirements.txt`

## Training

First, see [network/](network/) for directions on preparing the dataset for training. Finish training then come back here to use this model for segmentation.

TODO: provide a pre-trained model checkpoint.

## Run segmentation

We'll use `segway`, a package as developed in [Nguyen et al](https://www.nature.com/articles/s41586-022-05471-w) to run the segmentation pipeline. The pipeline utilizes [Daisy](https://github.com/funkelab/daisy/) to run tasks in small blocks in parallel, and chain tasks together in one single command.

### Predict affs

To run affs on the validation dataset, first run inference using GPU device 0:
```
CUDA_VISIBLE_DEVICES=0 python segway/tasks/segmentation/task_01_predict.py configs/segment_validation_reference.json
```

`segment_validation_reference.json` assumes the reference model is trained in `networks/`. To run the affs model on `train` and `test` sets, see the other configs in [configs/](configs/).

Note that for the `validation` volume, we segment only the inner 23um^3 while for the `test` volume we segment 33um^3 because these are the ROIs of the ground-truth skeletons.

TODO: provide affs from the pre-trained model.

### Run watershed and agglomeration

Then the rest of the pipeline:
```
python segway/tasks/segmentation/task_06a_extract_segments.py configs/segment_validation_reference.json
```

The output for this config will be in `outputs/validation/reference/400000/output.zarr`.

TODO: provide segmentations from the pre-trained model.

## Visualize

Here we use the `neuroglancer` in the `data/` folder to check the resulting affs and segmentation outputs.

`../data/neuroglancer --file ../xpress-challenge.zarr --datasets volumes/validation_raw --file outputs/validation/reference/400000/output.zarr --datasets volumes/affs volumes/segmentation_0.500`
