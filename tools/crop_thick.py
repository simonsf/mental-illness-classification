from utils.python.image3d_tools import resample_nn, resample_trilinear
from utils.python.image3d import Image3d
from utils.python.frame3d import Frame3d
import numpy as np


def center_crop_thick(image, coord, spacing, size, padtype=0, is_world=False,
                padvalue=0, method='NN'):
    """
    crop a sub-volume centered at voxel.
    :param image:       an image3d object
    :param coord:       the coordinate of center voxel
    :param spacing:     spacing of output volume
    :param size:        size of output volume
    :param padtype:     padding type, 0 for value padding and 1 for edge padding
    :param is_world:    whether the coord is world coordinate
    :param padvalue:    the default padding value for value padding
    :param method:      interpolation method, 'NN' for nearest neighbor
                        interpolation, and 'LINEAR' for linear interpolation
    :return: a sub-volume image3d object
    """
    assert isinstance(image, Image3d), 'image must be an image3d object'

    if size[0] <= 0 or size[1] <= 0 or size[2] <= 0:
        raise ValueError('[error] negative image size')

    coord = np.array(coord, dtype=np.double)
    spacing = np.array(spacing, dtype=np.double)

    if not is_world:
        origin = image.voxel_to_world(coord)
    else:
        origin = coord

    if not is_world:
        for i in range(3):
            origin -= image.axis(i) * spacing[i] * size[i] / 2.0
    else:
        for i in range(3):
            origin -= image.axis(i) * size[i] / 2.0

        for i in range(3):
            size[i] = int(size[i] / spacing[i])

    coord = np.rint(image.world_to_voxel(origin)).astype(np.int)
    origin = image.voxel_to_world(coord)

    frame = image.frame().deep_copy()
    assert isinstance(frame, Frame3d)
    frame.set_origin(origin)
    frame.set_spacing(spacing)

    if method == 'NN':
        return resample_nn(image, frame, size, padtype, padvalue)
    elif method == 'LINEAR':
        return resample_trilinear(image, frame, size, padtype, padvalue)
    else:
        raise ValueError('Unsupported Interpolation Method')