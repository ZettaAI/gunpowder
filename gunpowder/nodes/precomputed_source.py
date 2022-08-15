import logging
import numpy as np

from gunpowder.batch import Batch
from gunpowder.coordinate import Coordinate
from gunpowder.profiling import Timing
from gunpowder.roi import Roi
from gunpowder.array import Array
from gunpowder.array_spec import ArraySpec
from .batch_provider import BatchProvider

logger = logging.getLogger(__name__)


class PrecomputedSource(BatchProvider):
    """An interface to Neuroglancer's Precomputed format (via CloudVolume)

    Requires a dict of keys to CloudVolumes. The user is responsible to make sure the
    CloudVolumes are synced at the appropriate resolution. The spec for each dataset
    is based on the info file of each CloudVolume. Resolution is currently hardcoded
    to (1, 1, 1), so that arbitrary cutouts can be made.

    Args:

        datasets (``dict``, :class:`ArrayKey` -> ``CloudVolume``):

            Dictionary of array keys to dataset CloudVolumes that this source offers.

    """

    def __init__(self, datasets):

        self.datasets = datasets

    def setup(self):

        for key, vol in self.datasets.items():
            spec = self.__read_spec(vol)
            self.provides(key, spec)

    def provide(self, request):

        timing = Timing(self)
        timing.start()

        batch = Batch()

        for key, request_spec in request.array_specs.items():
            # voxel_size = self.spec[self.key].voxel_size
            # dataset_roi = request_spec.roi / voxel_size
            # dataset_roi = dataset_roi - self.spec[self.key].roi.get_offset() / voxel_size
            dataset_roi = request_spec.roi

            # create array spec
            array_spec = self.spec[key].copy()
            array_spec.roi = request_spec.roi

            # add array to batch
            batch.arrays[key] = Array(
                self.__read(self.datasets[key], dataset_roi), array_spec,
            )

        logger.debug("done")

        timing.stop()
        batch.profiling_stats.add(timing)

        return batch

    def __read_spec(self, vol):
        spec = ArraySpec()
        # spec.voxel_size = Coordinate(vol.resolution)
        spec.voxel_size = Coordinate((1, 1, 1))
        offset = Coordinate(vol.voxel_offset)
        shape = Coordinate(vol.volume_size)
        spec.roi = Roi(offset, shape)
        spec.dtype = vol.dtype

        spec.interpolatable = spec.dtype in [
            np.float,
            np.float32,
            np.float64,
            np.float128,
            np.uint8,  # assuming this is not used for labels
        ]

        return spec

    def __read(self, vol, roi):
        return np.squeeze(vol[roi.to_slices()], axis=3)

    def name(self):

        return super().name() + f"[{[vol.cloudpath for vol in self.datasets.values()]}]"
