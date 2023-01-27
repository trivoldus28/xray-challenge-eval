import argparse
import daisy
import h5py
import numpy as np
from funlib.segment.arrays import replace_values


def save_as_h5(output_file, output_dataset, nparray, resolution, offset):
    h5f = h5py.File(output_file, 'w')
    dset = h5f.create_dataset(output_dataset, data=nparray, compression="lzf")
    dset.attrs['resolution'] = resolution
    # dset.attrs['offset'] = np.array(offset)/resolution
    dset.attrs['offset'] = offset
    h5f.close()
    return


def save_using_daisy(output_file, output_dataset, nparray, resolution, offset):
    '''
    Daisy supports saving arrays as .zarr or .n5 (but not .h5)
    '''
    resolution = daisy.Coordinate(resolution)
    offset = daisy.Coordinate(offset)
    # roi_offset = offset * resolution
    roi_shape = daisy.Coordinate(nparray.shape) * resolution
    roi = daisy.Roi(offset, roi_shape)
    ds = daisy.prepare_ds(
            output_file, output_dataset,
            roi, resolution, nparray.dtype,
            compressor={'id': 'blosc', 'clevel': 3},
            delete=True
        )
    ds[roi] = nparray
    return


if __name__=="__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("input_file", type=str, help='')
    ap.add_argument("input_dataset", type=str, help='')
    ap.add_argument("output_file", type=str, help='')
    ap.add_argument("output_dataset", type=str, help='')
    ap.add_argument("--downsample", type=int, help='', default=None)
    ap.add_argument("--transpose", type=int, help='', default=0)
    # ap.add_argument("--adj_offset_xyz", type=str, help='', default='0,0,0')
    ap.add_argument("--keep_labels", type=str, help='', nargs='+', default=None)
    config = ap.parse_args()
    arg_config = vars(ap.parse_args())
    for k, v in arg_config.items():
        globals()[k] = v

    # adj_offset_xyz = [float(k) for k in adj_offset_xyz.split(',')]
    # adj_offset_xyz = np.array(adj_offset_xyz)
    # adj_offset_zyx = adj_offset_xyz[::-1]

    array = daisy.open_ds(input_file, input_dataset)

    if downsample is not None:
        assert type(downsample) is int
        ndarray = array.data[::downsample, ::downsample, ::downsample]
        ds_voxel_size = array.voxel_size * downsample
        # align ROI
        roi_begin = (array.roi.begin // ds_voxel_size) * ds_voxel_size
        roi_shape = daisy.Coordinate(ndarray.shape) * ds_voxel_size
        array = daisy.Array(data=ndarray,
                            roi=daisy.Roi(roi_begin, roi_shape),
                            voxel_size=ds_voxel_size
                            )

    data = array.to_ndarray()
    resolution = array.voxel_size
    offset = array.roi.get_begin()
    # offset -= daisy.Coordinate(adj_offset_zyx)
    print(f'resolution: {resolution}')
    print(f'offset: {offset}')
    # print(f'adj_offset_zyx: {adj_offset_zyx}')

    if transpose:
        data = np.transpose(data)
        resolution = resolution[::-1]
        offset = offset[::-1]

    if keep_labels is not None:
        assert type(keep_labels) == list
        keep_labels = [int(k) for k in keep_labels]
        labels = set(np.unique(data))
        labels = list(labels - set(keep_labels))
        zeros = [0 for k in labels]
        replace_values(
            data,
            labels,
            zeros,
            data)

    if output_file.endswith('.h5'):
        save_as_h5(output_file, output_dataset, data, resolution, offset)
        exit()

    save_using_daisy(output_file, output_dataset, data, resolution, offset)
