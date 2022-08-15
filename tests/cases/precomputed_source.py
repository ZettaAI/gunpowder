import argparse
import gunpowder as gp
from gunpowder.coordinate import Coordinate
from cloudvolume import CloudVolume
import tifffile


def write_precomputed_cutout(src_path, bbox_start, bbox_shape, dst_path):
    raw = gp.ArrayKey("RAW")
    vol = CloudVolume(src_path)
    source = gp.PrecomputedSource({raw: vol})
    pipeline = source
    request = gp.BatchRequest()
    request[raw] = gp.Roi(bbox_start, bbox_shape)  # (50800, 133800, 2200), (64, 64, 1))
    with gp.build(pipeline):
        batch = pipeline.request_batch(request)

    tifffile.imwrite(dst_path, batch[raw].data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_path", type=str, help="path to directory of images")
    parser.add_argument(
        "--dst_path", type=str, help="path to CloudVolume where images will be stored"
    )
    parser.add_argument("--bbox_start", type=int, nargs=3, help="pixels of start")
    parser.add_argument(
        "--bbox_shape", type=int, nargs=3, help="pixel dimensions of bbox"
    )
    args = parser.parse_args()
    write_precomputed_cutout(**vars(args))
