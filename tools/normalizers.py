import numpy as np
import utils.python.image3d_tools as ctools
from utils.python.image3d import Image3d


class AdaptiveNormalizer(object):
    """
    use the minimum and maximum percentiles to normalize image intensities
    """
    def __init__(self, min_p=0.001, max_p=0.999, clip=True, min_rand=0, max_rand=0):
        """
        constructor
        :param min_p: percentile for computing minimum value
        :param max_p: percentile for computing maximum value
        :param clip: whether to clip the intensity between min and max
        :param min_rand: the random perturbation (%) of minimum value (0-1)
        :param max_rand: the random perturbation (%) of maximum value (0-1)
        """
        assert 1 >= min_p >= 0, 'min_p must be between 0 and 1'
        assert 1 >= max_p >= 0, 'max_p must be between 0 and 1'
        assert max_p > min_p, 'max_p must be > min_p'
        assert 1 >= min_rand >= 0, 'min_rand must be between 0 and 1'
        assert 1 >= max_rand >= 0, 'max_rand must be between 0 and 1'
        assert isinstance(clip, bool), 'clip must be a boolean'
        self.min_p = min_p
        self.max_p = max_p
        self.clip = clip
        self.min_rand = min_rand
        self.max_rand = max_rand

    def normalize(self, single_image):

        assert isinstance(single_image, Image3d), 'image must be an image3d object'
        normalize_min, normalize_max = ctools.percentiles(single_image, [self.min_p, self.max_p])

        if self.min_rand > 0:
            offset = np.abs(normalize_min) * self.min_rand
            offset = np.random.uniform(-offset, offset)
            normalize_min += offset

        if self.max_rand > 0:
            offset = np.abs(normalize_max) * self.max_rand
            offset = np.random.uniform(-offset, offset)
            normalize_max += offset

        normalize_mean = (normalize_min + normalize_max) / 2.0
        normalize_stddev = (normalize_max - normalize_min) / 2.0
        ctools.intensity_normalize(single_image, normalize_mean, normalize_stddev, clip=self.clip)

    def __call__(self, image):
        """ normalize image """
        if isinstance(image, Image3d):
            self.normalize(image)
        elif isinstance(image, (list, tuple)):
            for im in image:
                assert isinstance(im, Image3d)
                self.normalize(im)
        else:
            raise ValueError('Unknown type of input. Normalizer only supports Image3d or Image3d list/tuple')

    def static_obj(self):
        """ get a static normalizer object by removing randomness """
        obj = AdaptiveNormalizer(self.min_p, self.max_p, self.clip, min_rand=0, max_rand=0)
        return obj

    def to_dict(self):
        """ convert parameters to dictionary """
        obj = {'type': 1, 'min_p': self.min_p, 'max_p': self.max_p, 'clip': self.clip}
        return obj