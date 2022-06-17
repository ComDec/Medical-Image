import argparse
import sys
from collections import OrderedDict
import torch
from torch.utils.data import DataLoader

from dataset import Medicine_Dataset
from model import MResNet, SNet, VNet

if __name__ == '__main__':
    print(torch.__version__)
    torch.backends.cudnn.benchmark = False
    sys.path.append('./')
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='resnet18', help='different models')
    parser.add_argument('--model_path', type=str, default='./plt/ResNet.plt')
    parser.add_argument('--infer_path', type=str, default='./data/test/val.png')
    parser.add_argument('--data_dir', type=str, default='./data')

    opt = parser.parse_args()
    model_name = opt.model_type
    print('Current model: {}'.format(model_name))
    test_set = Medicine_Dataset(opt.data_dir, 'val')
    test_dataloader = DataLoader(test_set, batch_size=1, shuffle=False)

    Pmodel = MResNet(dtype='Res18')

    if model_name == 'resnet18':
        Pmodel = MResNet(dtype='Res18')

    if model_name == 'resnet34':
        Pmodel = MResNet(dtype='Res34')

    if model_name == 'squeeze_1':
        Pmodel = SNet(dtype='v1')

    if model_name == 'squeeze_2':
        Pmodel = SNet(dtype='v2')

    if model_name == 'vgg16':
        Pmodel = VNet(dtype='v1')

    if model_name == 'vgg34':
        model = VNet(dtype='v2')

    checkpoint = torch.load(opt.model_path, map_location=lambda storage, loc: storage)
    new_state_dict = OrderedDict()

    for key, value in checkpoint["state_dict"].items():
        key = key[6:]  # remove `att.`
        new_state_dict[key] = value
    Pmodel.load_state_dict(new_state_dict)

    Pmodel.eval()

    step = 0
    t_acc = .0
    for sample in test_dataloader:
        step += 1
        img = sample['img']
        label = sample['label']
        pred = Pmodel(img)
        acc = (torch.argmax(pred, dim=-1) == label).float().mean()
        t_acc += acc

    print('Accuracy in val set: {}'.format(t_acc/step))