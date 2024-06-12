import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from config import root_path, mri_2d_path


class MRI2D(Dataset):

    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.data = pd.read_csv(csv_path)
        self.filenames = self.data['filename'].values
        self.dir_paths = self.data['dir_path'].values
        self.labels = self.data['AD'].values

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        fn = self.filenames[index]
        fp = os.path.join(self.dir_paths[index], fn)
        image = (np.load(fp) / 255).astype('float32')
        if image.shape[2] == 3:
            image = image.transpose((2, 0, 1))
        label = self.labels[index]
        return image, label


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    train_set = MRI2D(os.path.join(root_path, 'datasets/train.csv'))
    train_set = DataLoader(train_set, batch_size=16, shuffle=True, drop_last=False)
    for batch_image, batch_label in train_set:
        break
