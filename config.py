from easydict import EasyDict as edict
from tools import AdaptiveNormalizer
import numpy as np

__C = edict()
cfg = __C


##################################
# general parameters
##################################

__C.general = {}

# image-mask-label pair list
# multi-modality image training, use csv annotation file
__C.general.train_list = "/home/yichu/brain_classification/train_template.csv"
__C.general.validation_list = None

# the output of training models and logs
__C.general.save_dir = 'training'

# continue training from certain epoch, -1 to train from scratch
__C.general.resume_epoch = -1

# the number of GPUs used in training
__C.general.num_gpus = 1

# random seed used in training (debugging purpose)
__C.general.seed = 0


##################################
# data set parameters
##################################

__C.dataset = {}

# the number of classes
__C.dataset.num_classes = 2

# the resolution on which segmentation is performed
__C.dataset.spacing = [0.9, 0.9]

# the sampling crop size, e.g., determine the context information
__C.dataset.crop_size = [208, 208]

# the number of instances in one bag, e.g., how many slices will be sampled
__C.dataset.bag_size = 23

# the default padding value list
__C.dataset.default_values = [-1, -1]

# translation augmentation (unit: mm)
__C.dataset.random_translation = [6, 6]

# linear interpolation method:
# 1) NN: nearest neighbor interpolation
# 2) LINEAR: linear interpolation
__C.dataset.interpolation = 'LINEAR'

# crop intensity normalizers (to [-1,1])
# one normalizer corresponds to one input modality
# 1) FixedNormalizer: use fixed mean and standard deviation to normalize intensity
# 2) AdaptiveNormalizer: use minimum and maximum intensity of crop to normalize intensity
__C.dataset.crop_normalizers = [AdaptiveNormalizer(0.001, 0.999), AdaptiveNormalizer(0.001, 0.999)]


####################################
# training loss
####################################

__C.loss = {}
# weight for CAM loss 
__C.loss.cam_weight = 1

####################################
# net
####################################

__C.net = {}
__C.net.name = 'MIL' 

######################################
# training parameters
######################################

__C.train = {}

# the number of training epochs
__C.train.epochs = 1001

# the number of samples in a batch
__C.train.batchsize = 8  

# the number of threads for IO
__C.train.num_threads = 4

# the learning rate
__C.train.lr = 1e-3
__C.train.momentum = 0.95
__C.train.weight_decay = 7e-4

# the beta in Adam optimizer
__C.train.betas = (0.9, 0.999)

# the number of batches to update loss curve
__C.train.plot_snapshot = 10

# the number of batches to save model
__C.train.save_epochs = 5


########################################
# debug parameters
########################################

__C.debug = {}

# whether to save input crops
__C.debug.save_inputs = False

__C.test = {}
__C.test.batchsize = 1
__C.test.epochs = 1