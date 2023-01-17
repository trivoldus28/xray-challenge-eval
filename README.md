# xray-challenge-eval

#### Make new environment

`python3.7 -m venv ubuntu18_py37`

#### Activate

`source ubuntu18_py37/bin/activate`

#### Install Dependencies

`pip install -r requirements.txt`

Then: `pip install git+https://github.com/funkelab/funlib.evaluate@d2852b3#egg=funlib.evaluate`

#### Run eval

```bash=
skel=skeletons/100nm_Cutout4_Validation.npz
fin=seg_validation_setup01_no_bg.h5
ds=submission_0p500
python eval.py $skel $fin $ds
```

#### Expected outputs

```
n_neurons: 159
Expected run-length: 4262.811622537732
Split count (total, per-neuron): 513, 3.2264150943396226
Merge count (total, per-neuron): 33, 0.20754716981132076
VOI results:
Rand split: 0.48944144434593373
Rand merge: 0.9445481412350287
VOI split: 1.5402566264329618
VOI merge: 0.09220808711464024
Normalized VOI split: 0.18124617902875728
Normalized VOI merge: 0.010850375955715273
```
