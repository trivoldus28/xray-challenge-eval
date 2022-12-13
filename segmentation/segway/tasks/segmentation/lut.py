import os
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class LookupTable():

    def __init__(
            self,
            filepath,
            dataset=None):
        self.filepath = filepath
        self.dataset = dataset

    def set_dataset(self, dataset):
        self.dataset = dataset

    def get_path(self, roi, dataset):
        p = os.path.join(self.filepath, dataset)
        idx = list(roi.begin)
        # return os.path.join(p, *[str(k) for k in idx])
        # return os.path.join(p, "_".join([str(k) for k in idx]))
        return os.path.join(p, "/".join([str(k) for k in idx]))

    def check(self, roi, dataset=None):
        if dataset is None:
            dataset = self.dataset
        out_file = self.get_path(roi, dataset) + '.npz'
        print(out_file)
        return os.path.exists(out_file)

    def save(self, block, datadict, dataset=None):
        if dataset is None:
            dataset = self.dataset

        out_file = self.get_path(block.write_roi, dataset)
        logger.debug(f"Saving {out_file}")
        try:
            # try saving without making OS mkdir requests
            np.savez_compressed(out_file, **datadict)
        except FileNotFoundError:
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
            np.savez_compressed(out_file, **datadict)

    def save_lut(self, block, data, dataset=None):
        data = {'fragment_segment_lut': data}
        self.save(block, data, dataset)

    def save_nodes(self, block, data, dataset=None):
        data = {'nodes': data}
        self.save(block, data, dataset)

    def save_edges(self, block, data, dataset=None):
        data = {'edges': data}
        self.save(block, data, dataset)

    def load(self, block, datadict, dataset=None):
        if dataset is None:
            dataset = self.dataset

        # note: we'll need to use write_roi for now bc
        #       read_roi is not indicative of the block in worker_04a1_find_segments
        # out_file = self.get_path(block.read_roi, dataset) + '.npz'
        out_file = self.get_path(block.write_roi, dataset) + '.npz'
        logger.debug(f"Loading {out_file}")
        return np.load(out_file)[datadict]
        # try:
        #     # try saving without making OS mkdir requests
        #     np.savez_compressed(out_file, **datadict)
        # except FileNotFoundError:
        #     os.makedirs(os.path.dirname(out_file), exist_ok=True)
        #     np.savez_compressed(out_file, **datadict)

    def load_lut(self, block, dataset=None):
        return self.load(block, 'fragment_segment_lut', dataset)

    def load_nodes(self, block, dataset=None):
        return self.load(block, 'nodes', dataset)

    def load_edges(self, block, dataset=None):
        return self.load(block, 'edges', dataset)
