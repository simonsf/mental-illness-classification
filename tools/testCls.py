from models.model_new import MILinMIL_Cls
from .dataset import *
from utils.model_io import load_pytorch_model
import torch.nn as nn
import itertools
from collections import Counter
import matplotlib.pyplot as plt
from .normalizers import AdaptiveNormalizer
import os


def __normalizer_from_dict(crop_normalizer):
    """ convert dictionary to crop normalizer """

    if crop_normalizer['type'] == 1:
        ret = AdaptiveNormalizer(crop_normalizer['min_p'], crop_normalizer['max_p'], crop_normalizer['clip'])
    else:
        raise ValueError('unknown normalizer type: {}'.format(crop_normalizer['type']))
    return ret


def load_network(path, gpu_idx=0):
    #os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpu_idx)
    state = load_pytorch_model(path)
    out_channels = state['out_channels']
    in_channels = state['in_channels']
    bag_size = state['bag_size']

    model = MILinMIL_Cls(img_ch=in_channels, out_cls_ch=out_channels, bag_size=bag_size)
    model = nn.DataParallel(model, device_ids=[gpu_idx])
    model = model.cuda()

    preprocess = {}

    model.load_state_dict(state['state_dict'])
    model = model.eval()

    preprocess['spacing'] = np.array(state['spacing'], dtype=np.double)
    preprocess['crop_normalizers'] = []
    for crop_normalizer in state['crop_normalizers']:
        preprocess['crop_normalizers'].append(__normalizer_from_dict(crop_normalizer))
    preprocess['interpolation'] = state['interpolation']
    preprocess['default_values'] = np.array(state['default_values'], dtype=np.double)
    preprocess['bag_size'] = bag_size
    preprocess['crop_size'] = state['crop_size']
    preprocess['num_class'] = out_channels
    return model, preprocess


def plot_confusion_matrix(cm, classes, save_path,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(save_path, dpi=100)
    plt.close()


def predict_proba(model, imgs, t2_imgs=None, num_class=2):
    softmax = nn.Softmax(dim=1)
    inp = torch.cat([imgs[i] for i in range(imgs.shape[0])], dim=0)
    if t2_imgs is not None:
        inp_t2 = torch.cat([t2_imgs[i] for i in range(t2_imgs.shape[0])], dim=0)
        inp = torch.cat([inp, inp_t2], dim=1)
    with torch.no_grad():
        cls_pred, att, indices, feat = model(inp, return_feat=True)
    probas = []
    for i in range(num_class):
        proba = softmax(cls_pred)[:, i, :]
        probas.append(proba.cpu())
    indices = indices[0, :, 0]
    indices = sorted(dict(Counter(indices.cpu().detach().numpy())).items(), key=lambda x: x[1], reverse=True)
    att = att.permute(1, 0, 2, 3)
    feat = feat[:, :, 0]
    return cls_pred.cpu(), probas, indices, att.cpu(), inp.cpu(), feat.cpu()


def test(model, test_dataloader, gpu_idx=0, num_class=2):
    resultPred = None
    resultLabel = None
    resultProbas = None
    inds = []
    inps = []
    atts = []
    feats = []
    device = torch.device("cuda:%d"%(gpu_idx) if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for i_batch, data in enumerate(test_dataloader):
            imgs = data['t1'].to(device).float()
            if 't2' in data:
                t2_imgs = data['t2'].to(device).float()
            else:
                t2_imgs = None

            #print(imgs.shape)
            if 'label' in data:
                cls_labs = torch.tensor(np.array(data['label']).astype(np.int8)).long().unsqueeze(1)
            else:
                cls_labs = None
            cls_pred, probas, indices, att, inp, feat = predict_proba(model, imgs, t2_imgs, num_class)
            res = cls_pred.argmax(1).long()
            inds.append(indices)
            inps.append(inp)
            atts.append(att)
            feats.append(feat)
            if resultPred is None:
                resultPred = res
                resultLabel = cls_labs
                resultProbas = probas
            else:
                resultPred = torch.cat((resultPred, res), 0)
                if cls_labs is not None:
                    resultLabel = torch.cat((resultLabel, cls_labs), 0)
                else:
                    resultLabel = None
                for i in range(len(probas)):
                    resultProbas[i] = torch.cat((resultProbas[i], probas[i]), 0)
            torch.cuda.empty_cache()
    feats = torch.cat(feats, dim=0)

    return resultPred, resultProbas, resultLabel, inds, inps, atts, feats





