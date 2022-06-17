import os.path
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision import datasets

class Medicine_Dataset(Dataset):
    def __init__(self, root_dir, dtype='train'):
        super(Medicine_Dataset, self).__init__()
        self.data_dir = os.path.join(root_dir, dtype)
        self.ImageSet = datasets.ImageFolder(self.data_dir)

    def __getitem__(self, item):

        img_compose = [
            transforms.Resize([224, 224]),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]

        img_composer = transforms.Compose(img_compose)
        img, label = self.ImageSet[item]
        img = img_composer(img)

        return {'img': img, 'label': label}

    def __len__(self):
        return len(self.ImageSet)


if __name__ == "__main__":
    myset = Medicine_Dataset('./data')
    print(myset[1]['img'].shape)