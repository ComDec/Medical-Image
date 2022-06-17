import argparse
import sys

import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from model import MResNet, SNet, VNet, LNet, LSTMNet
from dataset import Medicine_Dataset
from torch.utils.data import DataLoader
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS


class pl_dataset(pl.LightningDataModule):

    def __init__(self, root_dir, params):
        super(pl_dataset, self).__init__()
        self.val_data = None
        self.test_set = None
        self.params = params
        self.test_data = None
        self.train_data = None
        self.val_set = None
        self.train_set = None
        self.data_dir = root_dir

    def prepare_data(self) -> None:
        self.train_data = Medicine_Dataset(root_dir=self.data_dir, dtype='train')
        self.val_data = Medicine_Dataset(root_dir=self.data_dir, dtype='val')
        self.test_data = Medicine_Dataset(root_dir=self.data_dir, dtype='test')

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_set, self.val_set, self.test_set = self.train_data, self.val_data, self.test_data

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_set, batch_size=self.params.batch_size, num_workers=6)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_set, batch_size=self.params.batch_size, num_workers=6)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_set, batch_size=1, num_workers=6)


class pl_model(pl.LightningModule):
    def __init__(self, params):
        super(pl_model, self).__init__()

        model_name = params.model_type
        self.params = params
        # self.automatic_optimization =True
        self.save_hyperparameters()

        self.model = MResNet(dtype='Res18')

        if model_name == 'resnet18':
            self.model = MResNet(dtype='Res18')

        if model_name == 'resnet34':
            self.model = MResNet(dtype='Res34')

        if model_name == 'squeeze_1':
            self.model = SNet(dtype='v1')

        if model_name == 'squeeze_2':
            self.model = SNet(dtype='v2')

        if model_name == 'vgg16':
            self.model = VNet(dtype='v1')

        if model_name == 'vgg34':
            self.model = VNet(dtype='v2')

        if model_name == 'linearNet':
            self.model = LNet()

        if model_name == 'LSTM':
            self.model = LSTMNet()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.params.lr)
        if self.params.optimizer == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.params.lr)
        elif self.params.optimizer == 'SGD':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.params.lr)

        return optimizer

    def training_step(self, train_batch, batch_idx):
        sample = train_batch
        img = sample['img']
        label = sample['label']
        pred = self(img)
        loss = F.cross_entropy(pred, label)
        acc = (torch.argmax(pred, dim=-1) == label).float().mean()
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)

        return loss

    def validation_step(self, val_batch, batch_index):
        sample = val_batch
        img = sample['img']
        label = sample['label']
        pred = self(img)
        loss = F.cross_entropy(pred, label)
        acc = (torch.argmax(pred, dim=-1) == label).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

        return loss


if __name__ == "__main__":
    print(torch.__version__)
    torch.backends.cudnn.benchmark = False
    sys.path.append('./')
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='resnet18', help='different models')
    parser.add_argument('--train', type=bool, default=True, help='train models from null')
    parser.add_argument('--val', type=bool, default=False, help='val models from model1.pt')
    parser.add_argument('--test', type=bool, default=False, help='choose your images to test')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--model_path', type=str, default='./plt/ResNet.plt')
    parser.add_argument('--infer_path', type=str, default='./data/test/val.png')
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--log_every_step', type=int, default=50)
    parser.add_argument('--val_every_step', type=int, default=1)
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--nums_gpu', type=int, default=1)
    opt = parser.parse_args()
    print('Current model: {}'.format(opt.model_type))
    data_meta = pl_dataset('./data', opt)
    m_model = pl_model(opt)

    data_meta.prepare_data()
    data_meta.setup('fit')
    train_loader = data_meta.train_dataloader()
    val_loader = data_meta.val_dataloader()
    trainer = pl.Trainer(accelerator='gpu', devices=opt.nums_gpu, amp_backend="apex", amp_level="O1", max_epochs=opt.max_epoch, log_every_n_steps=opt.log_every_step, check_val_every_n_epoch=opt.val_every_step)
    trainer.fit(model=m_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

