import torch
from datasets.dataset_h5 import HDF5Dataset
import torchvision.transforms as transforms
import tqdm

augmentation = [
            transforms.ToTensor()
        ]

train_dataset = HDF5Dataset(
        '../../data',
        '../../../../data/department/aubreville/datasets/C-KIT/C-KIT-11',
        csv_path='dataset_csv/ckit_data_pretraining.csv',
        recursive=True,
        transform=transforms.Compose(augmentation))

train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, pin_memory=True, drop_last=True,
        sampler=torch.utils.data.RandomSampler(train_dataset, num_samples=20000, replacement=True))

####### COMPUTE MEAN / STD

# placeholders
psum    = torch.tensor([0.0, 0.0, 0.0])
psum_sq = torch.tensor([0.0, 0.0, 0.0])

# loop through images
for image in tqdm.tqdm(train_loader):
    psum    += image.sum(axis        = [0, 2, 3])
    psum_sq += (image ** 2).sum(axis = [0, 2, 3])
    
# pixel count
count = len(train_loader) * 256 * 256 * 64

# mean and std
total_mean = psum / count
total_var  = (psum_sq / count) - (total_mean ** 2)
total_std  = torch.sqrt(total_var)

# output
print('mean: '  + str(total_mean))
print('std:  '  + str(total_std))