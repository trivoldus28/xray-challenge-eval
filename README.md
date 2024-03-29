
# XPRESS challenge

See [data/](data/) for more directions on downloading datasets.

You can also preview the datasets first before downloading:

Training raw + skeleton GT + voxel GT: [link](https://neuroglancer-demo.appspot.com/#!%7B%22dimensions%22:%7B%22x%22:%5B3.3e-8%2C%22m%22%5D%2C%22y%22:%5B3.3e-8%2C%22m%22%5D%2C%22z%22:%5B3.3e-8%2C%22m%22%5D%7D%2C%22position%22:%5B604.5031127929688%2C602.7859497070312%2C600.5%5D%2C%22crossSectionScale%22:0.9323938199059489%2C%22projectionOrientation%22:%5B-0.527566134929657%2C0.579612672328949%2C-0.5048351287841797%2C0.3617522418498993%5D%2C%22projectionScale%22:2048%2C%22layers%22:%5B%7B%22type%22:%22image%22%2C%22source%22:%22precomputed://gs://lee-pacureanu_data-exchange_us-storage/xray-challenge/training-raw%22%2C%22tab%22:%22source%22%2C%22name%22:%22training-raw%22%7D%2C%7B%22type%22:%22segmentation%22%2C%22source%22:%22precomputed://gs://lee-pacureanu_data-exchange_us-storage/xray-challenge/xpress-training-voxel-labels%22%2C%22tab%22:%22source%22%2C%22name%22:%22xpress-training-voxel-labels%22%7D%2C%7B%22type%22:%22segmentation%22%2C%22source%22:%22precomputed://https://catmaid3.hms.harvard.edu/cb2o2/staged_alignment_v3/delete_me/ng_skeletons/cutout5_230123%22%2C%22tab%22:%22source%22%2C%22segments%22:%5B%221%22%5D%2C%22segmentQuery%22:%221%22%2C%22name%22:%22gt_skeletons%22%7D%5D%2C%22showSlices%22:false%2C%22selectedLayer%22:%7B%22visible%22:true%2C%22layer%22:%22xpress-training-voxel-labels%22%7D%2C%22layout%22:%224panel%22%7D)

Validation raw + skeleton GT: [link](https://neuroglancer-demo.appspot.com/#!%7B%22dimensions%22:%7B%22x%22:%5B3.3e-8%2C%22m%22%5D%2C%22y%22:%5B3.3e-8%2C%22m%22%5D%2C%22z%22:%5B3.3e-8%2C%22m%22%5D%7D%2C%22position%22:%5B627.4053344726562%2C595.3873291015625%2C597.5%5D%2C%22crossSectionScale%22:2.7319072728259264%2C%22projectionOrientation%22:%5B-0.7071067690849304%2C0%2C0%2C0.7071067690849304%5D%2C%22projectionScale%22:1571.2377855505529%2C%22layers%22:%5B%7B%22type%22:%22image%22%2C%22source%22:%7B%22url%22:%22precomputed://gs://lee-pacureanu_data-exchange_us-storage/xray-challenge/validation-raw%22%2C%22subsources%22:%7B%22default%22:true%2C%22bounds%22:true%7D%2C%22enableDefaultSubsources%22:false%7D%2C%22tab%22:%22source%22%2C%22name%22:%22raw%22%7D%2C%7B%22type%22:%22segmentation%22%2C%22source%22:%22precomputed://https://catmaid3.hms.harvard.edu/cb2o2/staged_alignment_v3/delete_me/ng_skeletons/cutout4_230123%22%2C%22tab%22:%22source%22%2C%22segments%22:%5B%221%22%5D%2C%22segmentQuery%22:%221%22%2C%22name%22:%22skeletons_gt%22%7D%5D%2C%22showSlices%22:false%2C%22selectedLayer%22:%7B%22visible%22:true%2C%22layer%22:%22s22_WM_100nm_rec_db27_400_upscaled_cutout4_3x.tif%22%7D%2C%22layout%22:%224panel%22%7D)

Test raw: [link](https://neuroglancer-demo.appspot.com/#!%7B%22dimensions%22:%7B%22x%22:%5B3.3e-8%2C%22m%22%5D%2C%22y%22:%5B3.3e-8%2C%22m%22%5D%2C%22z%22:%5B3.3e-8%2C%22m%22%5D%7D%2C%22position%22:%5B619.9406127929688%2C558.4985961914062%2C600.5%5D%2C%22crossSectionScale%22:3.3201169227365477%2C%22projectionScale%22:2048%2C%22layers%22:%5B%7B%22type%22:%22image%22%2C%22source%22:%22precomputed://gs://lee-pacureanu_data-exchange_us-storage/xray-challenge/test-raw%22%2C%22tab%22:%22source%22%2C%22name%22:%22test-raw%22%7D%5D%2C%22selectedLayer%22:%7B%22visible%22:true%2C%22layer%22:%22test-raw%22%7D%2C%22layout%22:%224panel%22%7D)

## Running baseline segmentation model

See [segmentation/](segmentation/) for running our baseline segmentation model.

## Evaluations and submission

And see [eval/](eval/) for directions on running evaluations and submission.
