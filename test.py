from tools.testCls import *
from tools.dataset import ClassificationDataset
from tools.grad_cam import *
from sklearn import metrics
from utils.gpu_switcher import use_gpu
import os
import argparse


def classify(gpu_idx, model_path, image_list_path, result_list_path, **kwargs):
    """
    :param gpu_idx: which gpu to run classify model
    :param model_path: the folder that contains classify model
    :param image_list_path: input images list csv path
    :param result_list_path: output label csv path
    :param kwargs: reserve extended params, contains param: result_folder

    """
    df = pd.read_csv(image_list_path, header=None)
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpu_idx)
    model, preprocess = load_network(model_path)
    if len(df.columns) == 4:
        with_label = True
    else:
        with_label = False
    test_dataset = ClassificationDataset(
        imlist_file=image_list_path,
        num_classes=preprocess['num_class'],
        spacing=preprocess['spacing'],
        crop_size=preprocess['crop_size'],
        bag_size=preprocess['bag_size'],
        default_values=preprocess['default_values'],
        random_translation=[0, 0],
        interpolation=preprocess['interpolation'],
        crop_normalizers=preprocess['crop_normalizers'],
        with_label=with_label
    )
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                                  shuffle=False, num_workers=8, drop_last=False)
    resultPred, resultProbas, resultLabel, inds, inps, atts, feats = test(model, test_dataloader,
                                                                   num_class=preprocess['num_class'])
    #print('done')
    if kwargs.get('result_folder') and resultLabel is not None:
        if not os.path.isdir(kwargs['result_folder']):
            os.makedirs(kwargs['result_folder'])
        cm = metrics.confusion_matrix(y_true=resultLabel.numpy(), y_pred=resultPred.numpy())
        plot_confusion_matrix(cm, ['normal', 'abnormal'],
                              os.path.join(kwargs['result_folder'], 'cm_thres0.5.png'))
    for i in range(len(resultProbas)):
        df['Result_Prob_%d' % i] = resultProbas[i].numpy()
    df['Result_Pred'] = resultPred.numpy()
    df.to_csv(result_list_path, index=False)
    if not kwargs.get('result_folder'):
        return
    for i, inp in enumerate(inps):
        pred = resultPred[i].item()
        name = str(i).zfill(5) + '-pred%d' % pred
        if resultLabel is not None:
            label = resultLabel[i].item()
            name = name + '-label%d' % label
        save_dir = os.path.join(kwargs['result_folder'], name)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        forward_cam(model, inp, save_dir)
    np.save(os.path.join(kwargs['result_folder'], 'features.npy'), feats.numpy())
    if resultLabel is not None:
        np.save(os.path.join(kwargs['result_folder'], 'labels.npy'), resultLabel.numpy())

    del os.environ['CUDA_VISIBLE_DEVICES']
    torch.cuda.empty_cache()


def main():

    from argparse import RawTextHelpFormatter

    long_description = 'UII Brain Segmentation2d Batch Testing Engine\n\n' \
                       'It supports multiple kinds of input:\n' \
                       '1. Image list txt file\n' \
                       '2. Single image file\n' \
                       '3. A folder that contains all testing images\n'

    parser = argparse.ArgumentParser(description=long_description,
                                     formatter_class=RawTextHelpFormatter)

    parser.add_argument('-i', '--input', type=str, help='input images list csv path')
    parser.add_argument('-m', '--model', type=str, help='model checkpoint path')
    parser.add_argument('-r', '--result', type=str, help='output label csv path')
    parser.add_argument('-o', '--output', type=str, help='output folder for CAM')
    parser.add_argument('-g', '--gpu_id', type=int, default='0', help='the gpu id to run model')
    args = parser.parse_args()
    classify(args.gpu_id, args.model, args.input, args.result, result_folder=args.output)


if __name__ == '__main__':
    main()

