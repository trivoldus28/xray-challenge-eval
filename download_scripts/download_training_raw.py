from cloudvolume import CloudVolume
import daisy
import numpy as np

cloudvol = CloudVolume(
    'gs://lee-pacureanu_data-exchange_us-storage/'
        'ls2892_LTP/2102/s22/s22_WM_100nm_rec_db27_400_upscaled_cutout5_3x.tif',
    use_https=True, parallel=True, progress=True)

# Use daisy to save data as Zarr files. Note: daisy coordinate is always in zyx
voxel_size = cloudvol.resolution[::-1]
roi_offset = cloudvol.voxel_offset*cloudvol.resolution
roi_offset = roi_offset[::-1]
roi_shape = cloudvol.volume_size*cloudvol.resolution
roi_shape = roi_shape[::-1]
raw_roi = daisy.Roi(roi_offset, roi_shape)
ds = daisy.prepare_ds(
        'xpress-challenge.zarr', 'volumes/training_raw',
        raw_roi, voxel_size, cloudvol.data_type,
        compressor={'id': 'blosc', 'clevel': 3},
        delete=True
    )

# Download
cloudvol_array = cloudvol[:]

# Remove channel dim and transpose xyz to zyx
cloudvol_array = np.squeeze(cloudvol_array, 3)
cloudvol_array = np.transpose(cloudvol_array, (2, 1, 0))

# Save to file
ds[ds.roi] = cloudvol_array
