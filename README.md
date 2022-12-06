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
skel=skeletons/100nm_Cutout6_Testing_1025.npz
fin=test_unet_vxel_skel_300_frag06_expanded.h5
ds=submission
python eval.py $skel $fin $ds
```

#### Expected outputs

```
Removing 5783 GT annotations outside of evaluated ROI
BG_NODES 3780
Expected run-length: 13050.052788437139
```
