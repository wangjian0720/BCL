import argparse
import os
import shutil

import h5py
import numpy as np
import torch
import SimpleITK as sitk
from medpy import metric
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm

from networks.net_factory import net_factory

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/data/wj/promise/npy_image/', help='Patch to dataset')
parser.add_argument('--res_path', type=str, default='./results', help='Path to results')
parser.add_argument('--exp', type=str, default='baseline/v0', help='experiment_name')
parser.add_argument('--model', type=str, default='unet', help='model_name')
parser.add_argument('--num_classes', type=int,  default=2, help='output channel of network')
parser.add_argument('--labeled_num', type=int, default=7, help='labeled data')
parser.add_argument("--flag_check_with_best_stu", default='True', 
                    help="True 使用学生模型进行预测, False 使用教师模型进行预测")
parser.add_argument('--gpu_id', type=int, default=0,
                        help='the id of gpu used to train the model')


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    asd = metric.binary.asd(pred, gt)

    return dice, asd

def test_single_volume(case, net, test_save_path, FLAGS):

    np_data_path = os.path.join(FLAGS.root_path)

    image = np.load(os.path.join(np_data_path, '{}.npy'.format(case)))
    label = np.load(os.path.join(np_data_path, '{}_segmentation.npy'.format(case)))

    prediction = np.zeros_like(label)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()

        net.eval()
        with torch.no_grad():
            out = net(input)
            out = torch.argmax(torch.softmax(out, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            prediction[ind] = out

    if np.sum(prediction == 1)==0:
        first_metric = 0,0,0,0
    else:
        first_metric = calculate_metric_percase(prediction == 1, label == 1)

    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.SetSpacing((1, 1, 10))
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.SetSpacing((1, 1, 10))
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    lab_itk.SetSpacing((1, 1, 10))
    sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
    sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
    sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")

    return first_metric


def Inference(FLAGS):
    with open("/data/wj/promise/" + '/test.list', 'r') as f:
        image_list = f.readlines()
        
    image_list = sorted([item.replace('\n', '').split(".")[0] for item in image_list])

    snapshot_path = "{}/{}_{}_labeled/{}".format(FLAGS.res_path, FLAGS.exp, 
                                                 FLAGS.labeled_num, FLAGS.model)
    
    if FLAGS.flag_check_with_best_stu == "True":
        test_save_path = "{}/{}_{}_labeled/{}_predictions_stu/".format(FLAGS.res_path, FLAGS.exp, FLAGS.labeled_num, FLAGS.model)
    else:
        test_save_path = "{}/{}_{}_labeled/{}_predictions_tea/".format(FLAGS.res_path, FLAGS.exp, FLAGS.labeled_num, FLAGS.model)

    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    
    net = net_factory(net_type=FLAGS.model, in_chns=1, class_num=FLAGS.num_classes)

    if FLAGS.flag_check_with_best_stu == "True":
        print("Use student model")
        save_model_path = os.path.join(snapshot_path, '{}_best_stu_model.pth'.format(FLAGS.model))
    else:
        print("Use teacher model")
        save_model_path = os.path.join(snapshot_path, '{}_best_tea_model.pth'.format(FLAGS.model))
    net.load_state_dict(torch.load(save_model_path, weights_only=False))
    print("init weight from {}".format(save_model_path))
    net.eval()

    first_total = 0.0
    for case in tqdm(image_list):
        first_metric = test_single_volume(case, net, test_save_path, FLAGS)
        first_total += np.asarray(first_metric)

    avg_metric = [first_total / len(image_list)]

    return avg_metric, test_save_path


if __name__ == '__main__':
    import pprint
    FLAGS = parser.parse_args()
    
    # 3. setup gpus and randomness
    # if FLAGS.gpu_id in range(8):
    if FLAGS.gpu_id in range(10):
        gid = FLAGS.gpu_id
    else:
        gid = 0
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gid)
    print("{}".format(pprint.pformat(FLAGS)))

    metric, test_save_path = Inference(FLAGS)
    print(metric)
    print(metric[0])
    with open(test_save_path+'../performance.txt', 'w') as f:
        f.writelines('metric is {} \n'.format(metric))
        f.writelines('average metric is {}\n'.format(metric[0]))
