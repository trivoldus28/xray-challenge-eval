
Training and running our baseline for segmentation.

### Install dependencies

`pip install -r requirements.txt`

### Training

First, see [network/](network/) for directions on preparing the dataset for training. Finish training then come back here to use this model for segmentation.

TODO: provide a pre-trained model checkpoint.

### Run segmentation

We'll use `segway`, a package developed for the manuscript ["Structured cerebellar connectivity supports resilient pattern separation"](https://www.nature.com/articles/s41586-022-05471-w) to run the segmentation pipeline. The pipeline utilizes Daisy to run tasks in small blocks and chain tasks together in one single command. Please see Daisy's [README](https://github.com/funkelab/daisy/) for more details.

To run segmentation on the validation dataset, first run inference using GPU device 0:
```
CUDA_VISIBLE_DEVICES=0 python segway/tasks/segmentation/task_01_predict.py configs/segment_validation.json
```

TODO: provide affs from the pre-trained model.

Then the rest of the pipeline:
```
python segway/tasks/segmentation/task_06a_extract_segments.py configs/segment_validation.json
```

The output Zarr will be in `output/`.

TODO: provide segmentations from the pre-trained model.

### Visualize

`../download_scripts/neuroglancer --file ../xpress-challenge.zarr --datasets volumes/validation_raw --file outputs/validation/setup01_no_bg/400000/output.zarr --datasets volumes/affs volumes/segmentation_0.500`

