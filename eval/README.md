# xray-challenge-eval

## Install Dependencies

`pip3 install -r requirements.txt`

## Run eval

We are providing the following evaluation script for you to evaluate a segmentation output against the `validation` ground-truth skeletons.

Example:
```bash=
skel=../data/skeletons/XPRESS_validation_skels.npz
fin=baseline_validation.h5
ds=submission
python eval.py $fin --segmentation_ds $ds --skeleton_file $skel --mode validation
```

### Expected outputs

By default, the script will just print the XPRESS ERL+Rand score which is used for ranking submissions:
```
0.7499562483321406
```

Use the `--show_all_scores 1` option to print more numbers which can be useful for development:
```
n_neurons: 238
Expected run-length: 12196.277300102622
Normalized ERL: 0.6666742989110856
Count results:
        Split count (total, per-neuron): 441, 1.8529411764705883
        Merge count (total, per-neuron): 49, 0.20588235294117646
Rand results (higher better):
        Rand split: 0.7121796364338487
        Rand merge: 0.9542967590725423
VOI results (lower better):
        Normalized VOI split: 0.09069016147388803
        Normalized VOI merge: 0.009372640868915904
XPRESS score (higher is better):
        ERL+VOI : 0.8083214488698418
        ERL+RAND: 0.7499562483321406
        VOI     : 0.9499685988285981
        RAND    : 0.8332381977531955
```

### Score explanations

- Normalized ERL: expected run length divided by the average ground-truth skeleton length
- XPRESS VOI: 1 - (`voi_split` + `voi_merge`) / 2
- XPRESS Rand: (`rand_split` + `rand_merge`) / 2
- XPRESS ERL+VOI: (Normalized ERL + XPRESS VOI) / 2
- XPRESS ERL+Rand: (Normalized ERL + XPRESS Rand) / 2

### Parallel evaluation

To aid participants with evaluating many checkpoints and agglomeration thresholds at once, we're also releasing simple utility scripts that we internally used. These are provided with zero supports but in hope that they will be useful as templates.

`database.py` provides the `Database` class that can store multiple score entries in a single SQL database file. This makes running and tracking potentially many different configs easier. It's hard coded to use (network names, checkpoints, thresholds) as config keys for filtering.

`run_val_parallel.py` uses `futures.ProcessPoolExecutor` to run specified configs in parallel, and uses `Database` to store them to a single `validation_results.db` SQL file.

`get_validation_results.py` reads `validation_results.db` then ranks entries to get the top-N configs and prints out a CSV table.


## Submission

To make sure that your submission of `test` is evaluated correctly, please take note of the following items:
- The submission must be a single zip file containing a single .h5 file named `submission.h5`. You can make the zip file with any common compression utility but the content must be `submission.h5`.
- The dataset name should be `submission`.
- In nm, the data offset and shape in nm should be exactly (3267,3267,3267) and (33066,33066,33066) - that is, the center 33um^3 of the provided raw volume.
    - **This is different from `validation` where the GT skels limited us to the center 23um^3 sub-volume**. Make sure that you set the ROI correctly for `test` vs `validation` when running segmentation.
- The segmentation is downsampled by 3x.
    - This means the submission resolution is 99nm, 3x larger than the raw resolution of 33nm.
    - We use downsampling to save submission bandwidth and evaluation overheads.
    - In 99nm resolution, the data offset and shape would be (33,33,33) and (334,334,334) voxels.

Check that the submission is properly formatted with this command:
```
python3 eval.py submission.h5 --check_submission 1
```
Which will print `Pass` if everything looks correct, otherwise it will give specific assertion errors.


### Conversion to HDF5

If you used our baseline segmentation flow, on running segmentation of the validation or test datasets you will have .zarr outputs.
To convert them to .h5 files for submission you can use the `convert_to_zarr_h5.py` script in the data/ folder of the repo.

For example:
```
python3 ../data/convert_to_zarr_h5.py ../segmentation/outputs/test/setup03/320000/output.zarr volumes/segmentation_0.600 submission.h5 submission --downsample 3
```

If your outputs are neither .zarr nor .h5, you'll need to write your own script to perform the conversion to the .h5 submission file format.
