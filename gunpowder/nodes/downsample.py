from .batch_filter import BatchFilter
from gunpowder.coordinate import Coordinate
from gunpowder.volume import VolumeType, Volume
import logging
import numbers
import numpy as np

logger = logging.getLogger(__name__)

class DownSample(BatchFilter):
    '''Downsample volumes in a batch by given factors.

    Args:

        volume_factors (dict): Dictionary mapping source :class:`VolumeType` to․
            a tuple `(f, volume_type)` of downsampling factor `f` and target․
            :class:`VolumeType`. `f` can be a single integer or a tuple of 
            integers, one for each dimension of the volume to downsample.
    ''' 

    def __init__(self, volume_factors):

        self.volume_factors = volume_factors
        self.outputs = []

        for input_volume, downsample in volume_factors.items():

            assert isinstance(input_volume, VolumeType)
            assert isinstance(downsample, tuple)
            assert len(downsample) == 2
            f, output_volume = downsample
            assert isinstance(output_volume, VolumeType)
            assert isinstance(f, numbers.Number) or isinstance(f, tuple), "Scaling factor should be a number or a tuple of numbers."
            assert output_volume not in self.outputs, "Output volume type %s is used twice."%output_volume

            self.outputs.append(output_volume)


    def prepare(self, request):

        for input_volume, downsample in self.volume_factors.items():

            f, output_volume = downsample

            if output_volume not in request.volumes:
                continue

            logger.debug("preparing downsampling of " + str(input_volume))

            request_roi = request.volumes[output_volume]
            scaled_roi = self.__scale_roi(request_roi, f)

            logger.debug("needed request for %s in %s is %s in %s"%(request_roi, output_volume, scaled_roi, input_volume))

            # add or merge to batch request
            if input_volume in request.volumes:
                request.volumes[input_volume] = request.volumes[input_volume].union(scaled_roi)
                logger.debug("merging with existing request to %s"%request.volumes[input_volume])
            else:
                request.volumes[input_volume] = scaled_roi
                logger.debug("adding as new request")

            # remove volume type provided by us
            del request.volumes[output_volume]

    def process(self, batch, request):

        for input_volume, downsample in self.volume_factors.items():

            f, output_volume = downsample

            if output_volume not in request.volumes:
                continue

            request_roi = request.volumes[output_volume]
            scaled_roi = self.__scale_roi(request_roi, f)
            input_roi = batch.volumes[input_volume].roi

            assert input_roi.contains(scaled_roi)

            # get data corresponding to scaled roi
            data = batch.volumes[input_volume].data[(scaled_roi - input_roi.get_begin()).get_bounding_box()]

            # downsample
            if isinstance(f, tuple):
                slices = tuple(slice(None, None, k) for k in f)
            else:
                slices = tuple(slice(None, None, f) for i in range(input_roi.dims()))

            logger.debug("downsampling %s with %s"%(input_volume, slices))

            data = data[slices]

            assert data.shape == request_roi.get_shape()

            # create output volume
            batch.volumes[output_volume] = Volume(
                    data,
                    request_roi,
                    batch.volumes[input_volume].resolution*f)

        # restore requested rois
        for input_volume, downsample in self.volume_factors.items():

            if input_volume not in batch.volumes:
                continue

            input_roi = batch.volumes[input_volume].roi
            request_roi = request.volumes[input_volume]

            if input_roi != request_roi:

                assert input_roi.contains(request_roi)

                logger.debug("restoring original request roi %s of %s from %s"%(request_roi, input_volume, input_roi))

                data = batch.volumes[input_volume].data[(request_roi - input_roi.get_begin()).get_bounding_box()]
                batch.volumes[input_volume].data = data
                batch.volumes[input_volume].roi = request_roi

    def __scale_roi(self, roi, f):

        prev_center = roi.get_center()
        scaled_roi = roi*f
        new_center = scaled_roi.get_center()
        scaled_roi = scaled_roi - (new_center - prev_center)

        return scaled_roi
