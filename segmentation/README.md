
Training and running our baseline for segmentation.

## Install dependencies

`pip install -r requirements.txt`

### SQLite3
You may get an error like below:
```
In file included from src/blob.c:1:0:
src/blob.h:4:10: fatal error: sqlite3.h: No such file or directory
#include "sqlite3.h"
        ^~~~~~~~~~~
compilation terminated.
error: command '/usr/bin/x86_64-linux-gnu-gcc' failed with exit code 1
[end of output]
note: This error originates from a subprocess, and is likely not a problem with pip.
ERROR: Failed building wheel for pysqlite3
```

To resolve this, in Ubuntu first install: `sudo apt-get install libsqlite3-dev`, then rerun `pip install -r requirements.txt`

## Training

First, see [network/](network/) for directions on preparing the dataset for training. Finish training then come back here to use this model for segmentation.

### Using our baseline pre-trained model

We're providing our pre-trained model which achieves a `validation` ERL+Rand XPRESS score of 0.75424 and `test` score of TBD. Download it [here](https://github.com/htem/xpress-challenge-files/releases/download/baseline/model_checkpoint_320000). To use this model, simply download it to `networks/reference_setup/` and continue the following segmentation steps.

## Run segmentation

We'll use `segway`, a package as developed in [Nguyen et al](https://www.nature.com/articles/s41586-022-05471-w) to run the segmentation pipeline. The pipeline utilizes [Daisy](https://github.com/funkelab/daisy/) to run tasks in small blocks in parallel, and chain tasks together in one single command.

### Predict affs

To run affs on the validation dataset, first run inference using GPU device 0:
```
CUDA_VISIBLE_DEVICES=0 python segway/tasks/segmentation/task_01_predict.py configs/segment_validation_reference.json
```

`segment_validation_reference.json` assumes the reference model is trained in `networks/` at least to 320,000 iterations. To run the affs model on `train` and `test` sets, see the other configs in [configs/](configs/).

Note that for the `validation` volume, we segment only the inner 23um^3 while for the `test` volume we segment 33um^3 because these are the ROIs of the ground-truth skeletons.

#### Using pre-computed affs of the baseline model

To help participants who may only want to work on optimizing watershed+agglomeration, we provide the following affs outputs from the baseline for:
- `training` volume: [download](https://github.com/htem/xpress-challenge-files/releases/download/baseline/baseline_affs_training.h5), [preview](https://neuroglancer-demo.appspot.com/#!%7B%22dimensions%22:%7B%22x%22:%5B3.3e-8%2C%22m%22%5D%2C%22y%22:%5B3.3e-8%2C%22m%22%5D%2C%22z%22:%5B3.3e-8%2C%22m%22%5D%7D%2C%22position%22:%5B614.39697265625%2C575.5819091796875%2C598.5%5D%2C%22crossSectionScale%22:1.6989323086185522%2C%22projectionOrientation%22:%5B-0.527566134929657%2C0.579612672328949%2C-0.5048351287841797%2C0.3617522418498993%5D%2C%22projectionScale%22:2048%2C%22layers%22:%5B%7B%22type%22:%22image%22%2C%22source%22:%22precomputed://gs://lee-pacureanu_data-exchange_us-storage/xray-challenge/training-raw%22%2C%22tab%22:%22source%22%2C%22name%22:%22training-raw%22%7D%2C%7B%22type%22:%22segmentation%22%2C%22source%22:%22precomputed://gs://lee-pacureanu_data-exchange_us-storage/xray-challenge/xpress-training-voxel-labels%22%2C%22tab%22:%22source%22%2C%22name%22:%22xpress-training-voxel-labels%22%2C%22visible%22:false%7D%2C%7B%22type%22:%22segmentation%22%2C%22source%22:%22precomputed://https://catmaid3.hms.harvard.edu/cb2o2/staged_alignment_v3/delete_me/ng_skeletons/cutout5_230123%22%2C%22tab%22:%22source%22%2C%22segments%22:%5B%221%22%5D%2C%22segmentQuery%22:%221%22%2C%22name%22:%22gt_skeletons%22%7D%2C%7B%22type%22:%22image%22%2C%22source%22:%22precomputed://gs://lee-pacureanu_data-exchange_us-storage/xray-challenge/precomputed/baseline_training_affs%22%2C%22tab%22:%22source%22%2C%22channelDimensions%22:%7B%22c%5E%22:%5B1%2C%22%22%5D%7D%2C%22name%22:%22baseline_affs%22%7D%5D%2C%22showSlices%22:false%2C%22selectedLayer%22:%7B%22layer%22:%22baseline_affs%22%7D%2C%22layout%22:%22xy%22%7D)
- `validation` volume: [download](https://github.com/htem/xpress-challenge-files/releases/download/baseline/baseline_affs_validation.h5), [preview](https://neuroglancer-demo.appspot.com/#!%7B%22dimensions%22:%7B%22x%22:%5B3.3e-8%2C%22m%22%5D%2C%22y%22:%5B3.3e-8%2C%22m%22%5D%2C%22z%22:%5B3.3e-8%2C%22m%22%5D%7D%2C%22position%22:%5B600.2473754882812%2C600.5645751953125%2C552.5%5D%2C%22crossSectionScale%22:0.8228346580560184%2C%22projectionOrientation%22:%5B-0.7071067690849304%2C0%2C0%2C0.7071067690849304%5D%2C%22projectionScale%22:1571.2377855505529%2C%22layers%22:%5B%7B%22type%22:%22image%22%2C%22source%22:%7B%22url%22:%22precomputed://gs://lee-pacureanu_data-exchange_us-storage/xray-challenge/validation-raw%22%2C%22subsources%22:%7B%22default%22:true%2C%22bounds%22:true%7D%2C%22enableDefaultSubsources%22:false%7D%2C%22tab%22:%22source%22%2C%22name%22:%22raw%22%7D%2C%7B%22type%22:%22segmentation%22%2C%22source%22:%22precomputed://https://catmaid3.hms.harvard.edu/cb2o2/staged_alignment_v3/delete_me/ng_skeletons/cutout4_230123%22%2C%22tab%22:%22source%22%2C%22segments%22:%5B%221%22%5D%2C%22segmentQuery%22:%221%22%2C%22name%22:%22skeletons_gt%22%7D%2C%7B%22type%22:%22image%22%2C%22source%22:%22precomputed://gs://lee-pacureanu_data-exchange_us-storage/xray-challenge/precomputed/baseline_validation_affs%22%2C%22tab%22:%22source%22%2C%22channelDimensions%22:%7B%22c%5E%22:%5B1%2C%22%22%5D%7D%2C%22name%22:%22baseline_affs%22%7D%5D%2C%22showSlices%22:false%2C%22selectedLayer%22:%7B%22layer%22:%22baseline_affs%22%7D%2C%22layout%22:%22xy%22%7D)
- `test` volume: download [1](https://github.com/htem/xpress-challenge-files/releases/download/baseline/baseline_affs_test.h5.z01),[2](https://github.com/htem/xpress-challenge-files/releases/download/baseline/baseline_affs_test.h5.z02),[3](https://github.com/htem/xpress-challenge-files/releases/download/baseline/baseline_affs_test.h5.zip), [preview](https://neuroglancer-demo.appspot.com/#!%7B%22dimensions%22:%7B%22x%22:%5B3.3e-8%2C%22m%22%5D%2C%22y%22:%5B3.3e-8%2C%22m%22%5D%2C%22z%22:%5B3.3e-8%2C%22m%22%5D%7D%2C%22position%22:%5B619.7257690429688%2C581.660888671875%2C618.5%5D%2C%22crossSectionScale%22:1.8221188003905098%2C%22projectionScale%22:2048%2C%22layers%22:%5B%7B%22type%22:%22image%22%2C%22source%22:%22precomputed://gs://lee-pacureanu_data-exchange_us-storage/xray-challenge/test-raw%22%2C%22tab%22:%22source%22%2C%22name%22:%22test-raw%22%7D%2C%7B%22type%22:%22image%22%2C%22source%22:%22precomputed://gs://lee-pacureanu_data-exchange_us-storage/xray-challenge/precomputed/baseline_test_affs%22%2C%22tab%22:%22source%22%2C%22channelDimensions%22:%7B%22c%5E%22:%5B1%2C%22%22%5D%7D%2C%22name%22:%22baseline_affs%22%7D%5D%2C%22selectedLayer%22:%7B%22layer%22:%22baseline_affs%22%7D%2C%22layout%22:%22xy%22%7D)

Please note that since the checkpoint used is optimized by the final segmentation score and not the intrinsic quality of the affs, these outputs are not necessarily optimal for other watershed+agglomeration parameters/methods.

To use these provided affs, download the .h5 file and use the `convert_to_zarr_h5.py` script to convert and place it in the output folder. For example:
```
python3 convert_to_zarr_h5.py baseline_affs_validation.h5 volumes/affs outputs/validation/setup03/320000/output.zarr volumes/affs
```

### Run watershed and agglomeration

Then the rest of the pipeline:
```
python segway/tasks/segmentation/task_06a_extract_segments.py configs/segment_validation_reference.json
```

The output for this config will be in `outputs/validation/reference/320000/output.zarr`.

#### Example segmentation from the baseline

Here we are providing the segmentation of the baseline model at 0.55 agglomeration threshold (the threshold that had the highest XPRESS score):
- `training` volume: [download](https://github.com/htem/xpress-challenge-files/releases/download/baseline/baseline_seg_training.h5), [preview](https://neuroglancer-demo.appspot.com/#!%7B%22dimensions%22:%7B%22x%22:%5B3.3e-8%2C%22m%22%5D%2C%22y%22:%5B3.3e-8%2C%22m%22%5D%2C%22z%22:%5B3.3e-8%2C%22m%22%5D%7D%2C%22position%22:%5B598.1517333984375%2C597.8560180664062%2C597.5%5D%2C%22crossSectionScale%22:0.9323938199059493%2C%22projectionOrientation%22:%5B-0.527566134929657%2C0.579612672328949%2C-0.5048351287841797%2C0.3617522418498993%5D%2C%22projectionScale%22:2048%2C%22layers%22:%5B%7B%22type%22:%22image%22%2C%22source%22:%22precomputed://gs://lee-pacureanu_data-exchange_us-storage/xray-challenge/training-raw%22%2C%22tab%22:%22source%22%2C%22name%22:%22training-raw%22%7D%2C%7B%22type%22:%22segmentation%22%2C%22source%22:%22precomputed://gs://lee-pacureanu_data-exchange_us-storage/xray-challenge/xpress-training-voxel-labels%22%2C%22tab%22:%22source%22%2C%22name%22:%22xpress-training-voxel-labels%22%2C%22visible%22:false%7D%2C%7B%22type%22:%22segmentation%22%2C%22source%22:%22precomputed://https://catmaid3.hms.harvard.edu/cb2o2/staged_alignment_v3/delete_me/ng_skeletons/cutout5_230123%22%2C%22tab%22:%22source%22%2C%22segments%22:%5B%221%22%5D%2C%22segmentQuery%22:%221%22%2C%22name%22:%22gt_skeletons%22%7D%2C%7B%22type%22:%22image%22%2C%22source%22:%22precomputed://gs://lee-pacureanu_data-exchange_us-storage/xray-challenge/precomputed/baseline_training_affs%22%2C%22tab%22:%22source%22%2C%22channelDimensions%22:%7B%22c%5E%22:%5B1%2C%22%22%5D%7D%2C%22name%22:%22baseline_affs%22%7D%2C%7B%22type%22:%22segmentation%22%2C%22source%22:%22precomputed://gs://lee-pacureanu_data-exchange_us-storage/xray-challenge/precomputed/baseline_training_seg%22%2C%22tab%22:%22source%22%2C%22name%22:%22baseline_seg%22%7D%5D%2C%22showSlices%22:false%2C%22selectedLayer%22:%7B%22layer%22:%22baseline_seg%22%7D%2C%22layout%22:%22xy%22%7D)
- `validation` volume: [download](https://github.com/htem/xpress-challenge-files/releases/download/baseline/baseline_seg_validation.h5), [preview](https://neuroglancer-demo.appspot.com/#!%7B%22dimensions%22:%7B%22x%22:%5B3.3e-8%2C%22m%22%5D%2C%22y%22:%5B3.3e-8%2C%22m%22%5D%2C%22z%22:%5B3.3e-8%2C%22m%22%5D%7D%2C%22position%22:%5B600.2473754882812%2C600.5645751953125%2C552.5%5D%2C%22crossSectionScale%22:0.8228346580560184%2C%22projectionOrientation%22:%5B-0.7071067690849304%2C0%2C0%2C0.7071067690849304%5D%2C%22projectionScale%22:1571.2377855505529%2C%22layers%22:%5B%7B%22type%22:%22image%22%2C%22source%22:%7B%22url%22:%22precomputed://gs://lee-pacureanu_data-exchange_us-storage/xray-challenge/validation-raw%22%2C%22subsources%22:%7B%22default%22:true%2C%22bounds%22:true%7D%2C%22enableDefaultSubsources%22:false%7D%2C%22tab%22:%22source%22%2C%22name%22:%22raw%22%7D%2C%7B%22type%22:%22segmentation%22%2C%22source%22:%22precomputed://https://catmaid3.hms.harvard.edu/cb2o2/staged_alignment_v3/delete_me/ng_skeletons/cutout4_230123%22%2C%22tab%22:%22source%22%2C%22segments%22:%5B%221%22%5D%2C%22segmentQuery%22:%221%22%2C%22name%22:%22skeletons_gt%22%7D%2C%7B%22type%22:%22image%22%2C%22source%22:%22precomputed://gs://lee-pacureanu_data-exchange_us-storage/xray-challenge/precomputed/baseline_validation_affs%22%2C%22tab%22:%22source%22%2C%22channelDimensions%22:%7B%22c%5E%22:%5B1%2C%22%22%5D%7D%2C%22name%22:%22baseline_affs%22%7D%2C%7B%22type%22:%22segmentation%22%2C%22source%22:%22precomputed://gs://lee-pacureanu_data-exchange_us-storage/xray-challenge/precomputed/baseline_validation_seg%22%2C%22tab%22:%22source%22%2C%22name%22:%22baseline_seg%22%7D%5D%2C%22showSlices%22:false%2C%22selectedLayer%22:%7B%22visible%22:true%2C%22layer%22:%22baseline_validation_seg%22%7D%2C%22layout%22:%22xy%22%7D)
- `test` volume: [download](https://github.com/htem/xpress-challenge-files/releases/download/baseline/baseline_seg_test.h5), [preview](https://neuroglancer-demo.appspot.com/#!%7B%22dimensions%22:%7B%22x%22:%5B3.3e-8%2C%22m%22%5D%2C%22y%22:%5B3.3e-8%2C%22m%22%5D%2C%22z%22:%5B3.3e-8%2C%22m%22%5D%7D%2C%22position%22:%5B605.1585083007812%2C605.770751953125%2C607.5%5D%2C%22crossSectionScale%22:1.82211880039051%2C%22projectionScale%22:2048%2C%22layers%22:%5B%7B%22type%22:%22image%22%2C%22source%22:%22precomputed://gs://lee-pacureanu_data-exchange_us-storage/xray-challenge/test-raw%22%2C%22tab%22:%22source%22%2C%22name%22:%22test-raw%22%7D%2C%7B%22type%22:%22image%22%2C%22source%22:%22precomputed://gs://lee-pacureanu_data-exchange_us-storage/xray-challenge/precomputed/baseline_test_affs%22%2C%22tab%22:%22source%22%2C%22channelDimensions%22:%7B%22c%5E%22:%5B1%2C%22%22%5D%7D%2C%22name%22:%22baseline_affs%22%7D%2C%7B%22type%22:%22segmentation%22%2C%22source%22:%22precomputed://gs://lee-pacureanu_data-exchange_us-storage/xray-challenge/precomputed/baseline_test_seg%22%2C%22tab%22:%22source%22%2C%22name%22:%22baseline_seg%22%7D%5D%2C%22selectedLayer%22:%7B%22layer%22:%22baseline_seg%22%7D%2C%22layout%22:%22xy%22%7D)

Conversion example:
```
python3 convert_to_zarr_h5.py baseline_seg_validation.h5 volumes/segmentation_0.550 outputs/validation/setup03/320000/output.zarr volumes/segmentation_0.550
```

## Visualize

Here we use the `neuroglancer` in the `data/` folder to check the resulting affs and segmentation outputs.

`../data/neuroglancer --file ../xpress-challenge.zarr --datasets volumes/validation_raw --file outputs/validation/reference/320000/output.zarr --datasets volumes/affs volumes/segmentation_0.550`
