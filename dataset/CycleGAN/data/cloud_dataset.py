import os
import warnings
import numpy as np
from tqdm import tqdm
from natsort import natsorted
import torch
from datetime import datetime

import os
import torch
from skimage.transform import resize
from pathlib import Path
import random

import rasterio
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset



import os
import numpy as np
import torch
from torch.utils.data import Dataset

class NPYCycleGANDataset(Dataset):
    def __init__(
        self,
        root,
        domain_A="target/S2",
        domain_B="input/S1",
        transform=None
    ):
        """
        root: base directory (asiaWest_test_npy_3)
        domain_A: subfolder for domain A
        domain_B: subfolder for domain B
        """

        self.dir_A = os.path.join(root, domain_A)
        self.dir_B = os.path.join(root, domain_B)

        self.A_files = sorted([
            f for f in os.listdir(self.dir_A) if f.endswith(".npy")
        ])
        self.B_files = sorted([
            f for f in os.listdir(self.dir_B) if f.endswith(".npy")
        ])

        assert len(self.A_files) == len(self.B_files), \
            "Domain A and B must have same number of files"

        self.transform = transform

    def __len__(self):
        return len(self.A_files)

    def __getitem__(self, index):
        A_path = os.path.join(self.dir_A, self.A_files[index])
        B_path = os.path.join(self.dir_B, self.B_files[index])

        A = np.load(A_path)   # (C,H,W) or (1,C,H,W)
        B = np.load(B_path)

        # remove time dimension if exists
        if A.ndim == 4:
            A = A[0]
        if B.ndim == 4:
            B = B[0]

        A = torch.from_numpy(A).float()
        B = torch.from_numpy(B).float()
        # print("A shape:", A.shape)
        A = A[[3,2,1],:,:]  # convert from BGR to RGB by reordering the channels
        
        B = torch.cat((B[0].unsqueeze(0), B[0].unsqueeze(0), B[0].unsqueeze(0)), 0)  # convert SAR to 3 channels by repeating the single channel
        # print("B shape:", B.shape)

        if self.transform:
            A = self.transform(A)
            B = self.transform(B)

        return {
            "A": A,
            "B": B,
            "A_paths": A_path,
            "B_paths": B_path
        }

class SEN12MSCR_new(Dataset):
    # SEN12MSCR_new is a new version of SEN12MSCR dataset, which is used for the new model set up
    # The dataset needs to have the following structure:
    
    # The dataset is split into train, test, and val sets
    # To use the dataset, the user needs to specify the root directory and the split type
    # The dataset will return a dictionary containing the input and target images
    # Input images are stored in the 'input' key, and target images are stored in the 'target' key
    # For example, sample = {'input': input_img, 'target': target_img}
    # input_img = sample['input']
    # target_img = sample['target']
    
    def __init__(self, root, split="all", rescale_method='default', data_mode = 'partial'):
        self.root_dir = root
        self.split = split
        self.rescale_method = rescale_method
        self.data_mode = data_mode
        
        if self.split == 'train':
            if self.data_mode == 'full':
                self.input_dir = os.path.join(self.root_dir, 'train_masked_full_npy')
                self.target_dir = os.path.join(self.root_dir, 'train_gt_full_npy')
                self.sar_dir = os.path.join(self.root_dir, 'train_sar_full_npy')
            elif self.data_mode == 'partial':
                self.input_dir = os.path.join(self.root_dir, 'train_masked_npy')
                self.target_dir = os.path.join(self.root_dir, 'train_gt_npy')
                self.sar_dir = os.path.join(self.root_dir, 'train_sar_npy')
            else:
                raise ValueError("Invalid data_mode. Choose either 'partial' or 'full'.")
        elif self.split == 'test':
            if self.data_mode == 'full':
                self.input_dir = os.path.join(self.root_dir, 'test_masked_full_npy')
                self.target_dir = os.path.join(self.root_dir, 'test_gt_full_npy')
                self.sar_dir = os.path.join(self.root_dir, 'test_sar_full_npy')
            elif self.data_mode == 'partial':
                self.input_dir = os.path.join(self.root_dir, 'test_masked_100_npy')
                self.target_dir = os.path.join(self.root_dir, 'test_gt_100_npy')
                self.sar_dir = os.path.join(self.root_dir, 'test_sar_100_npy')
            else:
                raise ValueError("Invalid data_mode. Choose either 'partial' or 'full'.")
        elif self.split == 'val':
            self.input_dir = os.path.join(self.root_dir, 'val_masked_npy')
            self.target_dir = os.path.join(self.root_dir, 'val_gt_npy')
            self.sar_dir = os.path.join(self.root_dir, 'val_sar_npy')
        else:
            raise ValueError("Invalid split type.")

    def __len__(self):
        return len(os.listdir(self.input_dir))

    def __getitem__(self, idx):
        if self.split == 'train':
            filename = os.listdir(self.input_dir)[idx]
            input_path = os.path.join(self.input_dir, filename)
            target_path = os.path.join(self.target_dir, filename)
            sar_path = os.path.join(self.sar_dir, filename)
            input_img = torch.from_numpy(np.load(input_path).astype('float32'))
            target_img = torch.from_numpy(np.load(target_path).astype('float32'))
            sar_img = torch.from_numpy(np.load(sar_path).astype('float32'))
            sample = {'cloudy': input_img, 'A': target_img, 'B': sar_img, 'A_paths': target_path, 'B_paths': sar_path, 'filename': filename}
            return sample
        elif self.split == 'test':
            filename = os.listdir(self.input_dir)[idx]
            input_path = os.path.join(self.input_dir, filename)
            target_path = os.path.join(self.target_dir, filename)
            sar_path = os.path.join(self.sar_dir, filename)
            input_img = torch.from_numpy(np.load(input_path).astype('float32'))
            target_img = torch.from_numpy(np.load(target_path).astype('float32'))
            sar_img = torch.from_numpy(np.load(sar_path).astype('float32'))
            sample = {'cloudy': input_img, 'A': target_img, 'B': sar_img, 'A_paths': target_path, 'B_paths': sar_path, 'filename': filename}
            return sample
        elif self.split == 'val':
            filename = os.listdir(self.input_dir)[idx]
            input_path = os.path.join(self.input_dir, filename)
            target_path = os.path.join(self.target_dir, filename)
            sar_path = os.path.join(self.sar_dir, filename)
            input_img = torch.from_numpy(np.load(input_path).astype('float32'))
            target_img = torch.from_numpy(np.load(target_path).astype('float32'))
            sar_img = torch.from_numpy(np.load(sar_path).astype('float32'))
            sample = {'cloudy': input_img, 'A': target_img, 'B': sar_img, 'A_paths': target_path, 'B_paths': sar_path, 'filename': filename}
            return sample
        else:
            raise ValueError("Unknown split type!")

    def load_pt(self, input_path):
        input_img = torch.load(input_path)
        return input_img

class SEN12MSCR_new_rgb(Dataset):
    # SEN12MSCR_new is a new version of SEN12MSCR dataset, which is used for the new model set up
    # The dataset needs to have the following structure:
    
    # The dataset is split into train, test, and val sets
    # To use the dataset, the user needs to specify the root directory and the split type
    # The dataset will return a dictionary containing the input and target images
    # Input images are stored in the 'input' key, and target images are stored in the 'target' key
    # For example, sample = {'input': input_img, 'target': target_img}
    # input_img = sample['input']
    # target_img = sample['target']
    
    def __init__(self, root, split="all", rescale_method='default', data_mode = 'partial'):
        self.root_dir = root
        self.split = split
        self.rescale_method = rescale_method
        self.data_mode = data_mode
        
        if self.split == 'train':
            if self.data_mode == 'full':
                self.input_dir = os.path.join(self.root_dir, 'train_masked_full_npy')
                self.target_dir = os.path.join(self.root_dir, 'train_gt_full_npy')
                self.sar_dir = os.path.join(self.root_dir, 'train_sar_full_npy')
            elif self.data_mode == 'partial':
                self.input_dir = os.path.join(self.root_dir, 'train_masked_npy')
                self.target_dir = os.path.join(self.root_dir, 'train_gt_npy')
                self.sar_dir = os.path.join(self.root_dir, 'train_sar_npy')
            else:
                raise ValueError("Invalid data_mode. Choose either 'partial' or 'full'.")
        elif self.split == 'test':
            if self.data_mode == 'full':
                self.input_dir = os.path.join(self.root_dir, 'test_masked_full_npy')
                self.target_dir = os.path.join(self.root_dir, 'test_gt_full_npy')
                self.sar_dir = os.path.join(self.root_dir, 'test_sar_full_npy')
            elif self.data_mode == 'partial':
                self.input_dir = os.path.join(self.root_dir, 'test_masked_100_npy')
                self.target_dir = os.path.join(self.root_dir, 'test_gt_100_npy')
                self.sar_dir = os.path.join(self.root_dir, 'test_sar_100_npy')
            else:
                raise ValueError("Invalid data_mode. Choose either 'partial' or 'full'.")
        elif self.split == 'val':
            self.input_dir = os.path.join(self.root_dir, 'val_masked_npy')
            self.target_dir = os.path.join(self.root_dir, 'val_gt_npy')
            self.sar_dir = os.path.join(self.root_dir, 'val_sar_npy')
        else:
            raise ValueError("Invalid split type.")

    def __len__(self):
        return len(os.listdir(self.input_dir))

    def __getitem__(self, idx):
        if self.split == 'train':
            filename = os.listdir(self.input_dir)[idx]
            input_path = os.path.join(self.input_dir, filename)
            target_path = os.path.join(self.target_dir, filename)
            sar_path = os.path.join(self.sar_dir, filename)
            input_img = torch.from_numpy(np.load(input_path).astype('float32'))
            target_img = torch.from_numpy(np.load(target_path).astype('float32'))
            sar_img = torch.from_numpy(np.load(sar_path).astype('float32'))
            sar_img = torch.cat((sar_img[0].unsqueeze(0), sar_img[0].unsqueeze(0), sar_img[0].unsqueeze(0)), 0)
            sample = {'cloudy': input_img, 'A': target_img[[3,2,1],:,:], 'B': sar_img, 'A_paths': target_path, 'B_paths': sar_path, 'filename': filename}
            return sample
        elif self.split == 'test':
            filename = os.listdir(self.input_dir)[idx]
            input_path = os.path.join(self.input_dir, filename)
            target_path = os.path.join(self.target_dir, filename)
            sar_path = os.path.join(self.sar_dir, filename)
            input_img = torch.from_numpy(np.load(input_path).astype('float32'))
            target_img = torch.from_numpy(np.load(target_path).astype('float32'))
            sar_img = torch.from_numpy(np.load(sar_path).astype('float32'))
            sar_img = torch.cat((sar_img[0].unsqueeze(0), sar_img[0].unsqueeze(0), sar_img[0].unsqueeze(0)), 0)
            sample = {'cloudy': input_img, 'A': target_img[[3,2,1],:,:], 'B': sar_img, 'A_paths': target_path, 'B_paths': sar_path, 'filename': filename}
            return sample
        elif self.split == 'val':
            filename = os.listdir(self.input_dir)[idx]
            input_path = os.path.join(self.input_dir, filename)
            target_path = os.path.join(self.target_dir, filename)
            sar_path = os.path.join(self.sar_dir, filename)
            input_img = torch.from_numpy(np.load(input_path).astype('float32'))
            target_img = torch.from_numpy(np.load(target_path).astype('float32'))
            sar_img = torch.from_numpy(np.load(sar_path).astype('float32'))
            sar_img = torch.cat((sar_img[0].unsqueeze(0), sar_img[0].unsqueeze(0), sar_img[0].unsqueeze(0)), 0)
            sample = {'cloudy': input_img, 'A': target_img[[3,2,1],:,:], 'B': sar_img, 'A_paths': target_path, 'B_paths': sar_path, 'filename': filename}
            return sample
        else:
            raise ValueError("Unknown split type!")

    def load_pt(self, input_path):
        input_img = torch.load(input_path)
        return input_img
    
class SEN12MSCR_new_rgb_2(Dataset):
    # SEN12MSCR_new is a new version of SEN12MSCR dataset, which is used for the new model set up
    # The dataset needs to have the following structure:
    
    # The dataset is split into train, test, and val sets
    # To use the dataset, the user needs to specify the root directory and the split type
    # The dataset will return a dictionary containing the input and target images
    # Input images are stored in the 'input' key, and target images are stored in the 'target' key
    # For example, sample = {'input': input_img, 'target': target_img}
    # input_img = sample['input']
    # target_img = sample['target']
    
    def __init__(self, root, split="all", rescale_method='default', data_mode = 'partial'):
        self.root_dir = root
        self.split = split
        self.rescale_method = rescale_method
        self.data_mode = data_mode
        
        if self.split == 'train':
            if self.data_mode == 'full':
                self.input_dir = os.path.join(self.root_dir, 'SEN12MSCR_train_input_npy_full')
                self.target_dir = os.path.join(self.root_dir, 'SEN12MSCR_train_gt_npy_full')
                self.sar_dir = os.path.join(self.root_dir, 'SEN12MSCR_train_sar_npy_full')
            elif self.data_mode == 'partial':
                self.input_dir = os.path.join(self.root_dir, 'SEN12MSCR_train_input_npy')
                self.target_dir = os.path.join(self.root_dir, 'SEN12MSCR_train_gt_npy')
                self.sar_dir = os.path.join(self.root_dir, 'SEN12MSCR_train_sar_npy')
            else:
                raise ValueError("Invalid data_mode. Choose either 'partial' or 'full'.")
        elif self.split == 'test':
            if self.data_mode == 'full':
                self.input_dir = os.path.join(self.root_dir, 'SEN12MSCR_test_input_npy_full')
                self.target_dir = os.path.join(self.root_dir, 'SEN12MSCR_test_gt_npy_full')
                self.sar_dir = os.path.join(self.root_dir, 'SEN12MSCR_test_sar_npy_full')
            elif self.data_mode == 'partial':
                self.input_dir = os.path.join(self.root_dir, 'SEN12MSCR_test_input_npy')
                self.target_dir = os.path.join(self.root_dir, 'SEN12MSCR_test_gt_npy')
                self.sar_dir = os.path.join(self.root_dir, 'SEN12MSCR_test_sar_npy')
            else:
                raise ValueError("Invalid data_mode. Choose either 'partial' or 'full'.")
        elif self.split == 'val':
            self.input_dir = os.path.join(self.root_dir, 'SEN12MSCR_val_input_npy')
            self.target_dir = os.path.join(self.root_dir, 'SEN12MSCR_val_gt_npy')
            self.sar_dir = os.path.join(self.root_dir, 'SEN12MSCR_val_sar_npy')
        else:
            raise ValueError("Invalid split type.")

    def __len__(self):
        return len(os.listdir(self.input_dir))

    def __getitem__(self, idx):
        if self.split == 'train':
            filename = os.listdir(self.input_dir)[idx]
            input_path = os.path.join(self.input_dir, filename)
            target_path = os.path.join(self.target_dir, filename)
            sar_path = os.path.join(self.sar_dir, filename)
            input_img = torch.from_numpy(np.load(input_path).astype('float32'))
            target_img = torch.from_numpy(np.load(target_path).astype('float32'))
            sar_img = torch.from_numpy(np.load(sar_path).astype('float32'))
            sar_img = torch.cat((sar_img[0].unsqueeze(0), sar_img[0].unsqueeze(0), sar_img[0].unsqueeze(0)), 0)
            sample = {'cloudy': input_img, 'A': target_img[[3,2,1],:,:], 'B': sar_img, 'A_paths': target_path, 'B_paths': sar_path, 'filename': filename}
            return sample
        elif self.split == 'test':
            filename = os.listdir(self.input_dir)[idx]
            input_path = os.path.join(self.input_dir, filename)
            target_path = os.path.join(self.target_dir, filename)
            sar_path = os.path.join(self.sar_dir, filename)
            input_img = torch.from_numpy(np.load(input_path).astype('float32'))
            target_img = torch.from_numpy(np.load(target_path).astype('float32'))
            sar_img = torch.from_numpy(np.load(sar_path).astype('float32'))
            sar_img = torch.cat((sar_img[0].unsqueeze(0), sar_img[0].unsqueeze(0), sar_img[0].unsqueeze(0)), 0)
            sample = {'cloudy': input_img, 'A': target_img[[3,2,1],:,:], 'B': sar_img, 'A_paths': target_path, 'B_paths': sar_path, 'filename': filename}
            return sample
        elif self.split == 'val':
            filename = os.listdir(self.input_dir)[idx]
            input_path = os.path.join(self.input_dir, filename)
            target_path = os.path.join(self.target_dir, filename)
            sar_path = os.path.join(self.sar_dir, filename)
            input_img = torch.from_numpy(np.load(input_path).astype('float32'))
            target_img = torch.from_numpy(np.load(target_path).astype('float32'))
            sar_img = torch.from_numpy(np.load(sar_path).astype('float32'))
            sar_img = torch.cat((sar_img[0].unsqueeze(0), sar_img[0].unsqueeze(0), sar_img[0].unsqueeze(0)), 0)
            sample = {'cloudy': input_img, 'A': target_img[[3,2,1],:,:], 'B': sar_img, 'A_paths': target_path, 'B_paths': sar_path, 'filename': filename}
            return sample
        else:
            raise ValueError("Unknown split type!")

    def load_pt(self, input_path):
        input_img = torch.load(input_path)
        return input_img

class SEN12MSCR_new_SEN12(Dataset):
    # SEN12MSCR_new is a new version of SEN12MSCR dataset, which is used for the new model set up
    # The dataset needs to have the following structure:
    
    # The dataset is split into train, test, and val sets
    # To use the dataset, the user needs to specify the root directory and the split type
    # The dataset will return a dictionary containing the input and target images
    # Input images are stored in the 'input' key, and target images are stored in the 'target' key
    # For example, sample = {'input': input_img, 'target': target_img}
    # input_img = sample['input']
    # target_img = sample['target']
    
    def __init__(self, root, split="all", rescale_method='default', data_mode = 'partial'):
        self.root_dir = root
        self.split = split
        self.rescale_method = rescale_method
        self.data_mode = data_mode
        
        if self.split == 'train':
            if self.data_mode == 'full':
                self.input_dir = os.path.join(self.root_dir, 'train_masked_full_npy')
                self.target_dir = os.path.join(self.root_dir, 'train_gt_full_npy')
                self.sar_dir = os.path.join(self.root_dir, 'train_sar_full_npy')
            elif self.data_mode == 'partial':
                self.input_dir = os.path.join(self.root_dir, 'train_masked_npy')
                self.target_dir = os.path.join(self.root_dir, 'train_gt_npy')
                self.sar_dir = os.path.join(self.root_dir, 'train_sar_npy')
            else:
                raise ValueError("Invalid data_mode. Choose either 'partial' or 'full'.")
        elif self.split == 'test':
            if self.data_mode == 'full':
                self.input_dir = os.path.join(self.root_dir, 'test_masked_full_npy')
                self.target_dir = os.path.join(self.root_dir, 'test_gt_full_npy')
                self.sar_dir = os.path.join(self.root_dir, 'test_sar_full_npy')
            elif self.data_mode == 'partial':
                self.input_dir = os.path.join(self.root_dir, 'test_masked_100_npy')
                self.target_dir = os.path.join(self.root_dir, 'test_gt_100_npy')
                self.sar_dir = os.path.join(self.root_dir, 'test_sar_100_npy')
            else:
                raise ValueError("Invalid data_mode. Choose either 'partial' or 'full'.")
        elif self.split == 'val':
            self.input_dir = os.path.join(self.root_dir, 'val_masked_npy')
            self.target_dir = os.path.join(self.root_dir, 'val_gt_npy')
            self.sar_dir = os.path.join(self.root_dir, 'val_sar_npy')
        else:
            raise ValueError("Invalid split type.")

    def __len__(self):
        return len(os.listdir(self.input_dir))

    def __getitem__(self, idx):
        if self.split == 'train':
            filename = os.listdir(self.input_dir)[idx]
            input_path = os.path.join(self.input_dir, filename)
            target_path = os.path.join(self.target_dir, filename)
            sar_path = os.path.join(self.sar_dir, filename)
            input_img = torch.from_numpy(np.load(input_path).astype('float32'))
            target_img = torch.from_numpy(np.load(target_path).astype('float32'))
            sar_img = torch.from_numpy(np.load(sar_path).astype('float32'))
            sar_img = torch.cat((sar_img[0].unsqueeze(0), sar_img[0].unsqueeze(0), sar_img[0].unsqueeze(0)), 0)
            sample = {'cloudy': input_img, 'A': target_img[[3,2,1],:,:], 'B': sar_img, 'A_paths': target_path, 'B_paths': sar_path, 'filename': filename}
            return sample
        elif self.split == 'test':
            filename = os.listdir(self.input_dir)[idx]
            input_path = os.path.join(self.input_dir, filename)
            target_path = os.path.join(self.target_dir, filename)
            sar_path = os.path.join(self.sar_dir, filename)
            input_img = torch.from_numpy(np.load(input_path).astype('float32'))
            target_img = torch.from_numpy(np.load(target_path).astype('float32'))
            sar_img = torch.from_numpy(np.load(sar_path).astype('float32'))
            sar_img = torch.cat((sar_img[0].unsqueeze(0), sar_img[0].unsqueeze(0), sar_img[0].unsqueeze(0)), 0)
            sample = {'cloudy': input_img, 'A': target_img[[3,2,1],:,:], 'B': sar_img, 'A_paths': target_path, 'B_paths': sar_path, 'filename': filename}
            return sample
        elif self.split == 'val':
            filename = os.listdir(self.input_dir)[idx]
            input_path = os.path.join(self.input_dir, filename)
            target_path = os.path.join(self.target_dir, filename)
            sar_path = os.path.join(self.sar_dir, filename)
            input_img = torch.from_numpy(np.load(input_path).astype('float32'))
            target_img = torch.from_numpy(np.load(target_path).astype('float32'))
            sar_img = torch.from_numpy(np.load(sar_path).astype('float32'))
            sar_img = torch.cat((sar_img[0].unsqueeze(0), sar_img[0].unsqueeze(0), sar_img[0].unsqueeze(0)), 0)
            sample = {'cloudy': input_img, 'A': target_img[[3,2,1],:,:], 'B': sar_img, 'A_paths': target_path, 'B_paths': sar_path, 'filename': filename}
            return sample
        else:
            raise ValueError("Unknown split type!")

    def load_pt(self, input_path):
        input_img = torch.load(input_path)
        return input_img

class SEN12MSCR_new_SEN12_real(Dataset):
    # SEN12MSCR_new is a new version of SEN12MSCR dataset, which is used for the new model set up
    # The dataset needs to have the following structure:
    
    # The dataset is split into train, test, and val sets
    # To use the dataset, the user needs to specify the root directory and the split type
    # The dataset will return a dictionary containing the input and target images
    # Input images are stored in the 'input' key, and target images are stored in the 'target' key
    # For example, sample = {'input': input_img, 'target': target_img}
    # input_img = sample['input']
    # target_img = sample['target']
    
    def __init__(self, root, split="all", rescale_method='default', data_mode = 'partial'):
        self.root_dir = root
        self.split = split
        self.rescale_method = rescale_method
        self.data_mode = data_mode
        
        if self.split == 'train':
            if self.data_mode == 'full':
                self.input_dir = os.path.join(self.root_dir, 'SEN12MSCR_train_input_npy_full')
                self.target_dir = os.path.join(self.root_dir, 'SEN12MSCR_train_gt_npy_full')
                self.sar_dir = os.path.join(self.root_dir, 'SEN12MSCR_train_sar_npy_full')
            elif self.data_mode == 'partial':
                self.input_dir = os.path.join(self.root_dir, 'SEN12MSCR_train_input_npy')
                self.target_dir = os.path.join(self.root_dir, 'SEN12MSCR_train_gt_npy')
                self.sar_dir = os.path.join(self.root_dir, 'SEN12MSCR_train_sar_npy')
            else:
                raise ValueError("Invalid data_mode. Choose either 'partial' or 'full'.")
        elif self.split == 'test':
            if self.data_mode == 'full':
                self.input_dir = os.path.join(self.root_dir, 'SEN12MSCR_test_input_npy_full')
                self.target_dir = os.path.join(self.root_dir, 'SEN12MSCR_test_gt_npy_full')
                self.sar_dir = os.path.join(self.root_dir, 'SEN12MSCR_test_sar_npy_full')
            elif self.data_mode == 'partial':
                self.input_dir = os.path.join(self.root_dir, 'SEN12MSCR_test_input_npy')
                self.target_dir = os.path.join(self.root_dir, 'SEN12MSCR_test_gt_npy')
                self.sar_dir = os.path.join(self.root_dir, 'SEN12MSCR_test_sar_npy')
            else:
                raise ValueError("Invalid data_mode. Choose either 'partial' or 'full'.")
        elif self.split == 'val':
            self.input_dir = os.path.join(self.root_dir, 'SEN12MSCR_val_input_npy')
            self.target_dir = os.path.join(self.root_dir, 'SEN12MSCR_val_gt_npy')
            self.sar_dir = os.path.join(self.root_dir, 'SEN12MSCR_val_sar_npy')
        else:
            raise ValueError("Invalid split type.")

    def __len__(self):
        return len(os.listdir(self.input_dir))

    def __getitem__(self, idx):
        if self.split == 'train':
            filename = os.listdir(self.input_dir)[idx]
            input_path = os.path.join(self.input_dir, filename)
            target_path = os.path.join(self.target_dir, filename)
            sar_path = os.path.join(self.sar_dir, filename)
            input_img = torch.from_numpy(np.load(input_path).astype('float32'))
            target_img = torch.from_numpy(np.load(target_path).astype('float32'))
            sar_img = torch.from_numpy(np.load(sar_path).astype('float32'))
            sar_img = torch.cat((sar_img[0].unsqueeze(0), sar_img[0].unsqueeze(0), sar_img[0].unsqueeze(0)), 0)
            sample = {'cloudy': input_img, 'A': target_img[[3,2,1],:,:], 'B': sar_img, 'A_paths': target_path, 'B_paths': sar_path, 'filename': filename}
            return sample
        elif self.split == 'test':
            filename = os.listdir(self.input_dir)[idx]
            input_path = os.path.join(self.input_dir, filename)
            target_path = os.path.join(self.target_dir, filename)
            sar_path = os.path.join(self.sar_dir, filename)
            input_img = torch.from_numpy(np.load(input_path).astype('float32'))
            target_img = torch.from_numpy(np.load(target_path).astype('float32'))
            sar_img = torch.from_numpy(np.load(sar_path).astype('float32'))
            sar_img = torch.cat((sar_img[0].unsqueeze(0), sar_img[0].unsqueeze(0), sar_img[0].unsqueeze(0)), 0)
            sample = {'cloudy': input_img, 'A': target_img[[3,2,1],:,:], 'B': sar_img, 'A_paths': target_path, 'B_paths': sar_path, 'filename': filename}
            return sample
        elif self.split == 'val':
            filename = os.listdir(self.input_dir)[idx]
            input_path = os.path.join(self.input_dir, filename)
            target_path = os.path.join(self.target_dir, filename)
            sar_path = os.path.join(self.sar_dir, filename)
            input_img = torch.from_numpy(np.load(input_path).astype('float32'))
            target_img = torch.from_numpy(np.load(target_path).astype('float32'))
            sar_img = torch.from_numpy(np.load(sar_path).astype('float32'))
            sar_img = torch.cat((sar_img[0].unsqueeze(0), sar_img[0].unsqueeze(0), sar_img[0].unsqueeze(0)), 0)
            sample = {'cloudy': input_img, 'A': target_img[[3,2,1],:,:], 'B': sar_img, 'A_paths': target_path, 'B_paths': sar_path, 'filename': filename}
            return sample
        else:
            raise ValueError("Unknown split type!")

    def load_pt(self, input_path):
        input_img = torch.load(input_path)
        return input_img
    
class SEN12MSCR_CycleGAN(Dataset):
    """
    CycleGAN-compatible unpaired dataset
    Domain A: Optical (RGB)
    Domain B: SAR (replicated to 3 channels)
    """

    def __init__(self, root, split="train", data_mode="partial"):
        self.root = Path(root)
        self.split = split
        self.data_mode = data_mode

        if split not in ["train", "test", "val"]:
            raise ValueError("split must be train | test | val")

        if split == "train":
            suffix = "_full" if data_mode == "full" else ""
            self.A_dir = self.root / f"SEN12MSCR_train_gt_npy{suffix}"
            self.B_dir = self.root / f"SEN12MSCR_train_sar_npy{suffix}"
        elif split == "test":
            suffix = "_full" if data_mode == "full" else ""
            self.A_dir = self.root / f"SEN12MSCR_test_gt_npy{suffix}"
            self.B_dir = self.root / f"SEN12MSCR_test_sar_npy{suffix}"
        else:  # val
            self.A_dir = self.root / "SEN12MSCR_val_gt_npy"
            self.B_dir = self.root / "SEN12MSCR_val_sar_npy"

        self.A_paths = sorted(self.A_dir.glob("*.npy"))
        self.B_paths = sorted(self.B_dir.glob("*.npy"))

        assert len(self.A_paths) > 0, "No optical files found"
        assert len(self.B_paths) > 0, "No SAR files found"

    def __len__(self):
        return max(len(self.A_paths), len(self.B_paths))

    def _resize(self, x):
        if x.shape[-2:] != (256, 256):
            x = resize(
                x,
                (*x.shape[:-2], 256, 256),
                order=1,
                preserve_range=True,
                anti_aliasing=True
            ).astype(x.dtype)
        return x

    def __getitem__(self, idx):
        # --- UNPAIRED RANDOM SAMPLING ---
        A_path = self.A_paths[idx % len(self.A_paths)]
        B_path = random.choice(self.B_paths)

        # --- LOAD OPTICAL (A) ---
        A = np.load(A_path).astype("float32")
        A = self._resize(A)
        A = A[[3, 2, 1], :, :]          # → RGB
        A = torch.from_numpy(A)

        # --- LOAD SAR (B) ---
        B = np.load(B_path).astype("float32")
        B = self._resize(B)
        B = torch.from_numpy(B)
        B = B[0].unsqueeze(0).repeat(3, 1, 1)

        # --- SAFETY CHECK ---
        if torch.isnan(A).any() or torch.isnan(B).any():
            return self.__getitem__((idx + 1) % len(self))

        return {
            "A": A,
            "B": B,
            "A_paths": str(A_path),
            "B_paths": str(B_path),
        }

# s2cloudless: see https://github.com/sentinel-hub/sentinel2-cloud-detector
from s2cloudless import S2PixelCloudDetector
# from detect_cloudshadow import get_cloud_mask, get_shadow_mask

import numpy as np
import scipy
import scipy.signal as scisig


def rescale(data, limits):
    return (data - limits[0]) / (limits[1] - limits[0])


def normalized_difference(channel1, channel2):
    subchan = channel1 - channel2
    sumchan = channel1 + channel2
    sumchan[sumchan == 0] = 0.001  # checking for 0 divisions
    return subchan / sumchan


def get_shadow_mask(data_image):
    data_image = data_image / 10000.

    (ch, r, c) = data_image.shape
    shadowmask = np.zeros((r, c)).astype('float32')

    BB     = data_image[1]
    BNIR   = data_image[7]
    BSWIR1 = data_image[11]

    CSI = (BNIR + BSWIR1) / 2.

    t3 = 3/4 # cloud-score index threshold
    T3 = np.min(CSI) + t3 * (np.mean(CSI) - np.min(CSI))

    t4 = 5 / 6  # water-body index threshold
    T4 = np.min(BB) + t4 * (np.mean(BB) - np.min(BB))

    shadow_tf = np.logical_and(CSI < T3, BB < T4)

    shadowmask[shadow_tf] = -1
    shadowmask = scisig.medfilt2d(shadowmask, 5)

    return shadowmask


def get_cloud_mask(data_image, cloud_threshold, binarize=False, use_moist_check=False):
    '''Adapted from https://github.com/samsammurphy/cloud-masking-sentinel2/blob/master/cloud-masking-sentinel2.ipynb'''

    data_image = data_image / 10000.
    (ch, r, c) = data_image.shape

    # Cloud until proven otherwise
    score = np.ones((r, c)).astype('float32')
    # Clouds are reasonably bright in the blue and aerosol/cirrus bands.
    score = np.minimum(score, rescale(data_image[1], [0.1, 0.5]))
    score = np.minimum(score, rescale(data_image[0], [0.1, 0.3]))
    score = np.minimum(score, rescale((data_image[0] + data_image[10]), [0.4, 0.9]))
    score = np.minimum(score, rescale((data_image[3] + data_image[2] + data_image[1]), [0.2, 0.8]))

    if use_moist_check:
        # Clouds are moist
        ndmi = normalized_difference(data_image[7], data_image[11])
        score = np.minimum(score, rescale(ndmi, [-0.1, 0.1]))

    # However, clouds are not snow.
    ndsi = normalized_difference(data_image[2], data_image[11])
    score = np.minimum(score, rescale(ndsi, [0.8, 0.6]))

    boxsize = 7
    box = np.ones((boxsize, boxsize)) / (boxsize ** 2)

    score = scipy.ndimage.morphology.grey_closing(score, size=(5, 5))
    score = scisig.convolve2d(score, box, mode='same')

    score = np.clip(score, 0.00001, 1.0)

    if binarize:
        score[score >= cloud_threshold] = 1
        score[score < cloud_threshold]  = 0

    return score

# IN: [13 x H x W] S2 image (of arbitrary resolution H,W), scalar cloud detection threshold
# OUT: cloud & shadow segmentation mask (of same resolution)
# the multispectral S2 images are expected to have their default ranges and not be value-standardized yet
# cloud_threshold: the higher the more conservative the masks (i.e. less pixels labeled clouds/shadows)
def get_cloud_cloudshadow_mask(data_image, cloud_threshold):
    cloud_mask = get_cloud_mask(data_image, cloud_threshold, binarize=True)
    shadow_mask = get_shadow_mask(data_image)

    # encode clouds and shadows as segmentation masks
    cloud_cloudshadow_mask = np.zeros_like(cloud_mask)
    cloud_cloudshadow_mask[shadow_mask < 0] = -1
    cloud_cloudshadow_mask[cloud_mask > 0]  = 1

    return cloud_cloudshadow_mask


# utility functions used in the dataloaders of SEN12MS-CR and SEN12MS-CR-TS
def read_tif(path_IMG):
    tif = rasterio.open(path_IMG)
    return tif

def read_img(tif):
    return tif.read().astype(np.float32)

def rescale(img, oldMin, oldMax):
    oldRange = oldMax - oldMin
    img      = (img - oldMin) / oldRange
    return img

def process_MS(img, method):
    if method=='default':
        intensity_min, intensity_max = 0, 10000            # define a reasonable range of MS intensities
        img = np.clip(img, intensity_min, intensity_max)   # intensity clipping to a global unified MS intensity range
        img = rescale(img, intensity_min, intensity_max)   # project to [0,1], preserve global intensities (across patches), gets mapped to [-1,+1] in wrapper
    if method=='resnet':
        intensity_min, intensity_max = 0, 10000            # define a reasonable range of MS intensities
        img = np.clip(img, intensity_min, intensity_max)   # intensity clipping to a global unified MS intensity range
        img /= 2000                                        # project to [0,5], preserve global intensities (across patches)
    return img

def process_MS2(img, method):
    if method=='default':
        intensity_min, intensity_max = 0, 10000            # define a reasonable range of MS intensities
        img = np.clip(img, intensity_min, intensity_max)   # intensity clipping to a global unified MS intensity range
        img = rescale(img, intensity_min, intensity_max)   # project to [0,1], preserve global intensities (across patches), gets mapped to [-1,+1] in wrapper
    if method=='resnet':
        intensity_min, intensity_max = 0, 10000           # define a reasonable range of MS intensities
        img = np.clip(img, intensity_min, intensity_max)   # intensity clipping to a global unified MS intensity range
        img /= 2000                                        # project to [0,5], preserve global intensities (across patches)
    return img

def process_SAR(img, method):
    if method=='default':
        dB_min, dB_max = -25, 0                            # define a reasonable range of SAR dB
        img = np.clip(img, dB_min, dB_max)                 # intensity clipping to a global unified SAR dB range
        img = rescale(img, dB_min, dB_max)                 # project to [0,1], preserve global intensities (across patches), gets mapped to [-1,+1] in wrapper
    if method=='resnet':
        # project SAR to [0, 2] range
        dB_min, dB_max = [-25.0, -32.5], [0, 0]
        img = np.concatenate([(2 * (np.clip(img[0], dB_min[0], dB_max[0]) - dB_min[0]) / (dB_max[0] - dB_min[0]))[None, ...],
                              (2 * (np.clip(img[1], dB_min[1], dB_max[1]) - dB_min[1]) / (dB_max[1] - dB_min[1]))[None, ...]], axis=0)
    return img

def get_cloud_cloudshadow_mask(img, cloud_threshold=0.2):
    cloud_mask = get_cloud_mask(img, cloud_threshold, binarize=True)
    shadow_mask = get_shadow_mask(img)

    # encode clouds and shadows as segmentation masks
    cloud_cloudshadow_mask = np.zeros_like(cloud_mask)
    cloud_cloudshadow_mask[shadow_mask < 0] = -1
    cloud_cloudshadow_mask[cloud_mask > 0] = 1

    # label clouds and shadows
    cloud_cloudshadow_mask[cloud_cloudshadow_mask != 0] = 1
    return cloud_cloudshadow_mask

def get_cloud_map(img, detector, instance=None):

    if detector == 'cloud_cloudshadow_mask':
        threshold = 0.2  # set to e.g. 0.2 or 0.4
        mask = get_cloud_cloudshadow_mask(np.clip(img, 0, 10000), threshold)
    elif detector== 's2cloudless_map':
        threshold = 0.5
        mask = instance.get_cloud_probability_maps(np.moveaxis(np.clip(img, 0, 10000)/10000, 0, -1)[None, ...])[0, ...]
        mask[mask < threshold] = 0
        mask = gaussian_filter(mask, sigma=2).astype(np.float32)
    elif detector == 's2cloudless_mask':
        mask = instance.get_cloud_masks(np.moveaxis(np.clip(img, 0, 10000)/10000, 0, -1)[None, ...])[0, ...]
    else:
        mask = np.ones((img.shape[-1], img.shape[-1]))
    return mask


class SEN12MSCR_new0(Dataset):
    def __init__(self, root, split="all", region='all', cloud_masks='s2cloudless_mask', sample_type='pretrain', rescale_method='default'):

        self.root_dir = root   # set root directory which contains all ROI
        self.region   = region # region according to which the ROI are selected # TODO: currently only supporting 'all'
        self.ROI      = {'ROIs1158': ['106'],
                         'ROIs1868': ['17', '36', '56', '73', '85', '100', '114', '119', '121', '126', '127', '139', '142', '143'],
                         'ROIs1970': ['20', '21', '35', '40', '57', '65', '71', '82', '83', '91', '112', '116', '119', '128', '132', '133', '135', '139', '142', '144', '149'],
                         'ROIs2017': ['8', '22', '25', '32', '49', '61', '63', '69', '75', '103', '108', '115', '116', '117', '130', '140', '146']}
        
        # define splits conform with SEN12MS-CR-TS
        self.splits         = {}
        self.splits['train']= ['ROIs1970_fall_s1/s1_3', 'ROIs1970_fall_s1/s1_22', 'ROIs1970_fall_s1/s1_148', 'ROIs1970_fall_s1/s1_107', 'ROIs1970_fall_s1/s1_1', 'ROIs1970_fall_s1/s1_114', 
                               'ROIs1970_fall_s1/s1_135', 'ROIs1970_fall_s1/s1_40', 'ROIs1970_fall_s1/s1_42', 'ROIs1970_fall_s1/s1_31', 'ROIs1970_fall_s1/s1_149', 'ROIs1970_fall_s1/s1_64', 
                               'ROIs1970_fall_s1/s1_28', 'ROIs1970_fall_s1/s1_144', 'ROIs1970_fall_s1/s1_57', 'ROIs1970_fall_s1/s1_35', 'ROIs1970_fall_s1/s1_133', 'ROIs1970_fall_s1/s1_30', 
                               'ROIs1970_fall_s1/s1_134', 'ROIs1970_fall_s1/s1_141', 'ROIs1970_fall_s1/s1_112', 'ROIs1970_fall_s1/s1_116', 'ROIs1970_fall_s1/s1_37', 'ROIs1970_fall_s1/s1_26', 
                               'ROIs1970_fall_s1/s1_77', 'ROIs1970_fall_s1/s1_100', 'ROIs1970_fall_s1/s1_83', 'ROIs1970_fall_s1/s1_71', 'ROIs1970_fall_s1/s1_93', 'ROIs1970_fall_s1/s1_119', 
                               'ROIs1970_fall_s1/s1_104', 'ROIs1970_fall_s1/s1_136', 'ROIs1970_fall_s1/s1_6', 'ROIs1970_fall_s1/s1_41', 'ROIs1970_fall_s1/s1_125', 'ROIs1970_fall_s1/s1_91', 
                               'ROIs1970_fall_s1/s1_131', 'ROIs1970_fall_s1/s1_120', 'ROIs1970_fall_s1/s1_110', 'ROIs1970_fall_s1/s1_19', 'ROIs1970_fall_s1/s1_14', 'ROIs1970_fall_s1/s1_81', 
                               'ROIs1970_fall_s1/s1_39', 'ROIs1970_fall_s1/s1_109', 'ROIs1970_fall_s1/s1_33', 'ROIs1970_fall_s1/s1_88', 'ROIs1970_fall_s1/s1_11', 'ROIs1970_fall_s1/s1_128', 
                               'ROIs1970_fall_s1/s1_142', 'ROIs1970_fall_s1/s1_122', 'ROIs1970_fall_s1/s1_4', 'ROIs1970_fall_s1/s1_27', 'ROIs1970_fall_s1/s1_147', 'ROIs1970_fall_s1/s1_85', 
                               'ROIs1970_fall_s1/s1_82', 'ROIs1970_fall_s1/s1_105', 'ROIs1158_spring_s1/s1_9', 'ROIs1158_spring_s1/s1_1', 'ROIs1158_spring_s1/s1_124', 'ROIs1158_spring_s1/s1_40', 
                               'ROIs1158_spring_s1/s1_101', 'ROIs1158_spring_s1/s1_21', 'ROIs1158_spring_s1/s1_134', 'ROIs1158_spring_s1/s1_145', 'ROIs1158_spring_s1/s1_141', 'ROIs1158_spring_s1/s1_66', 
                               'ROIs1158_spring_s1/s1_8', 'ROIs1158_spring_s1/s1_26', 'ROIs1158_spring_s1/s1_77', 'ROIs1158_spring_s1/s1_113', 'ROIs1158_spring_s1/s1_100', 
                               'ROIs1158_spring_s1/s1_117', 'ROIs1158_spring_s1/s1_119', 'ROIs1158_spring_s1/s1_6', 'ROIs1158_spring_s1/s1_58', 'ROIs1158_spring_s1/s1_120', 'ROIs1158_spring_s1/s1_110', 
                               'ROIs1158_spring_s1/s1_126', 'ROIs1158_spring_s1/s1_115', 'ROIs1158_spring_s1/s1_121', 'ROIs1158_spring_s1/s1_39', 'ROIs1158_spring_s1/s1_109', 'ROIs1158_spring_s1/s1_63', 
                               'ROIs1158_spring_s1/s1_75', 'ROIs1158_spring_s1/s1_132', 'ROIs1158_spring_s1/s1_128', 'ROIs1158_spring_s1/s1_142', 'ROIs1158_spring_s1/s1_15', 'ROIs1158_spring_s1/s1_45', 
                               'ROIs1158_spring_s1/s1_97', 'ROIs1158_spring_s1/s1_147', 'ROIs1868_summer_s1/s1_90', 'ROIs1868_summer_s1/s1_87', 'ROIs1868_summer_s1/s1_25', 'ROIs1868_summer_s1/s1_124', 
                               'ROIs1868_summer_s1/s1_114', 'ROIs1868_summer_s1/s1_135', 'ROIs1868_summer_s1/s1_40', 'ROIs1868_summer_s1/s1_101', 'ROIs1868_summer_s1/s1_42', 
                               'ROIs1868_summer_s1/s1_31', 'ROIs1868_summer_s1/s1_36', 'ROIs1868_summer_s1/s1_139', 'ROIs1868_summer_s1/s1_56', 'ROIs1868_summer_s1/s1_133', 'ROIs1868_summer_s1/s1_55', 
                               'ROIs1868_summer_s1/s1_43', 'ROIs1868_summer_s1/s1_113', 'ROIs1868_summer_s1/s1_100', 'ROIs1868_summer_s1/s1_76', 'ROIs1868_summer_s1/s1_123', 'ROIs1868_summer_s1/s1_143', 
                               'ROIs1868_summer_s1/s1_93', 'ROIs1868_summer_s1/s1_125', 'ROIs1868_summer_s1/s1_89', 'ROIs1868_summer_s1/s1_120', 'ROIs1868_summer_s1/s1_126', 'ROIs1868_summer_s1/s1_72', 
                               'ROIs1868_summer_s1/s1_115', 'ROIs1868_summer_s1/s1_121', 'ROIs1868_summer_s1/s1_146', 'ROIs1868_summer_s1/s1_140', 'ROIs1868_summer_s1/s1_95', 
                               'ROIs1868_summer_s1/s1_102', 'ROIs1868_summer_s1/s1_7', 'ROIs1868_summer_s1/s1_11', 'ROIs1868_summer_s1/s1_132', 'ROIs1868_summer_s1/s1_15', 'ROIs1868_summer_s1/s1_137', 
                               'ROIs1868_summer_s1/s1_4', 'ROIs1868_summer_s1/s1_27', 'ROIs1868_summer_s1/s1_147', 'ROIs1868_summer_s1/s1_86', 'ROIs1868_summer_s1/s1_47', 'ROIs2017_winter_s1/s1_68', 
                               'ROIs2017_winter_s1/s1_25', 'ROIs2017_winter_s1/s1_62', 'ROIs2017_winter_s1/s1_135', 'ROIs2017_winter_s1/s1_42', 'ROIs2017_winter_s1/s1_64', 'ROIs2017_winter_s1/s1_21', 
                               'ROIs2017_winter_s1/s1_55', 'ROIs2017_winter_s1/s1_112', 'ROIs2017_winter_s1/s1_116', 'ROIs2017_winter_s1/s1_8', 'ROIs2017_winter_s1/s1_59', 'ROIs2017_winter_s1/s1_49', 
                               'ROIs2017_winter_s1/s1_104',  'ROIs2017_winter_s1/s1_81', 'ROIs2017_winter_s1/s1_146', 'ROIs2017_winter_s1/s1_75', 
                               'ROIs2017_winter_s1/s1_94', 'ROIs2017_winter_s1/s1_102', 'ROIs2017_winter_s1/s1_61', 'ROIs2017_winter_s1/s1_47']
        self.splits['val']  = ['ROIs2017_winter_s1/s1_22', 'ROIs1868_summer_s1/s1_19', 'ROIs1970_fall_s1/s1_65', 'ROIs1158_spring_s1/s1_17', 'ROIs2017_winter_s1/s1_107', 
                               'ROIs1868_summer_s1/s1_80', 'ROIs1868_summer_s1/s1_127', 'ROIs2017_winter_s1/s1_130', 'ROIs1868_summer_s1/s1_17', 'ROIs2017_winter_s1/s1_84'] 
        self.splits['test'] = ['ROIs1158_spring_s1/s1_106', 'ROIs1158_spring_s1/s1_123', 'ROIs1158_spring_s1/s1_140', 'ROIs1158_spring_s1/s1_31', 'ROIs1158_spring_s1/s1_44', 
                               'ROIs1868_summer_s1/s1_119', 'ROIs1868_summer_s1/s1_73', 'ROIs1970_fall_s1/s1_139', 'ROIs2017_winter_s1/s1_108', 'ROIs2017_winter_s1/s1_63']

        self.splits["all"]  = self.splits["train"] + self.splits["test"] + self.splits["val"]
        self.split = split
        
        assert split in ['all', 'train', 'val', 'test'], "Input dataset must be either assigned as all, train, test, or val!"
        assert sample_type in ['pretrain'], "Input data must be pretrain!"
        assert cloud_masks in [None, 'cloud_cloudshadow_mask', 's2cloudless_map', 's2cloudless_mask'], "Unknown cloud mask type!"

        self.modalities     = ["S1", "S2"]
        self.cloud_masks    = cloud_masks   # e.g. 'cloud_cloudshadow_mask', 's2cloudless_map', 's2cloudless_mask'
        self.sample_type    = sample_type   # e.g. 'pretrain'

        self.time_points    = range(1)
        self.n_input_t      = 1             # specifies the number of samples, if only part of the time series is used as an input

        if self.cloud_masks in ['s2cloudless_map', 's2cloudless_mask']:
            self.cloud_detector = S2PixelCloudDetector(threshold=0.4, all_bands=True, average_over=4, dilation_size=2)
        else: self.cloud_detector = None

        self.paths          = self.get_paths()
        self.n_samples      = len(self.paths)

        # raise a warning if no data has been found
        if not self.n_samples: self.throw_warn()

        self.method         = rescale_method

    # indexes all patches contained in the current data split
    def get_paths(self):  # assuming for the same ROI+num, the patch numbers are the same
        print(f'\nProcessing paths for {self.split} split of region {self.region}')

        paths = []
        seeds_S1 = natsorted([s1dir for s1dir in os.listdir(self.root_dir) if "_s1" in s1dir])
        for seed in tqdm(seeds_S1):
            rois_S1 = natsorted(os.listdir(os.path.join(self.root_dir, seed)))
            for roi in rois_S1:
                roi_dir  = os.path.join(self.root_dir, seed, roi)
                paths_S1        = natsorted([os.path.join(roi_dir, s1patch) for s1patch in os.listdir(roi_dir)])
                paths_S2        = [patch.replace('/s1', '/s2').replace('_s1', '_s2') for patch in paths_S1]
                paths_S2_cloudy = [patch.replace('/s1', '/s2_cloudy').replace('_s1', '_s2_cloudy') for patch in paths_S1]

                for pdx, _ in enumerate(paths_S1):
                    # omit patches that are potentially unpaired
                    if not all([os.path.isfile(paths_S1[pdx]), os.path.isfile(paths_S2[pdx]), os.path.isfile(paths_S2_cloudy[pdx])]): continue
                    # don't add patch if not belonging to the selected split
                    if not any([split_roi in paths_S1[pdx] for split_roi in self.splits[self.split]]): continue
                    sample = {"S1":         paths_S1[pdx],
                              "S2":         paths_S2[pdx],
                              "S2_cloudy":  paths_S2_cloudy[pdx]}
                    paths.append(sample)
        return paths

    def __getitem__(self, pdx):  # get the triplet of patch with ID pdx
        s1_tif          = read_tif(os.path.join(self.root_dir, self.paths[pdx]['S1']))
        s2_tif          = read_tif(os.path.join(self.root_dir, self.paths[pdx]['S2']))
        s2_cloudy_tif   = read_tif(os.path.join(self.root_dir, self.paths[pdx]['S2_cloudy']))
        coord           = list(s2_tif.bounds)
        s1              = process_SAR(read_img(s1_tif), self.method)
        s2              = read_img(s2_tif)           # note: pre-processing happens after cloud detection
        s2_cloudy       = read_img(s2_cloudy_tif)    # note: pre-processing happens after cloud detection
        mask            = None if not self.cloud_masks else get_cloud_map(s2_cloudy, self.cloud_masks, self.cloud_detector)

        # sample = {'input': {'S1': s1,
        #                     'S2': process_MS(s2_cloudy, self.method),
        #                     'masks': mask if mask is not None else np.zeros(s2_cloudy.shape[1:]),
        #                     'coverage': np.mean(mask) if mask is not None else 0,
        #                     'S1 path': os.path.join(self.root_dir, self.paths[pdx]['S1']),
        #                     'S2 path': os.path.join(self.root_dir, self.paths[pdx]['S2_cloudy']),
        #                     'coord': coord,
        #                     },
        #             'target': {'S2': process_MS(s2, self.method),
        #                        'S2 path': os.path.join(self.root_dir, self.paths[pdx]['S2']),
        #                        'coord': coord,
        #                         },
        #             }
        sample = {
            "A": process_MS(s2, self.method)[[3, 2, 1], :, :] * 2 - 1,  # → RGB, project to [-1, +1]
            "B": s1 * 2 - 1,  # project to [-1, +1]
            "A_paths": os.path.join(self.root_dir, self.paths[pdx]['S2']),
            "B_paths": os.path.join(self.root_dir, self.paths[pdx]['S1']),
        }

        return sample

    def throw_warn(self):
        warnings.warn("""No data samples found! Please use the following directory structure:

        path/to/your/SEN12MSCR/directory:
            ├───ROIs1158_spring_s1
            |   ├─s1_1
            |   |   |...
            |   |   ├─ROIs1158_spring_s1_1_p407.tif
            |   |   |...
            |    ...
            ├───ROIs1158_spring_s2
            |   ├─s2_1
            |   |   |...
            |   |   ├─ROIs1158_spring_s2_1_p407.tif
            |   |   |...
            |    ...
            ├───ROIs1158_spring_s2_cloudy
            |   ├─s2_cloudy_1
            |   |   |...
            |   |   ├─ROIs1158_spring_s2_cloudy_1_p407.tif
            |   |   |...
            |    ...
            ...

        Note: Please arrange the dataset in a format as e.g. provided by the script dl_data.sh.
        """)

    def __len__(self):
        # length of generated list
        return self.n_samples


import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from natsort import natsorted
from tqdm import tqdm

class SEN12MSCR_CycleGAN_Unpaired(Dataset):
    """
    CycleGAN-compatible unpaired dataset
    Domain A: Sentinel-2 (clean optical)
    Domain B: Sentinel-1 (SAR)
    """

    def __init__(self, root, split="train", rescale_method="default"):
        self.root_dir = root
        self.split = split
        self.method = rescale_method

        assert split in ["train", "val", "test", "all"]

        # --- split definitions (UNCHANGED) ---
        self.splits         = {}
        self.splits['train']= ['ROIs1970_fall_s1/s1_3', 'ROIs1970_fall_s1/s1_22', 'ROIs1970_fall_s1/s1_148', 'ROIs1970_fall_s1/s1_107', 'ROIs1970_fall_s1/s1_1', 'ROIs1970_fall_s1/s1_114', 
                               'ROIs1970_fall_s1/s1_135', 'ROIs1970_fall_s1/s1_40', 'ROIs1970_fall_s1/s1_42', 'ROIs1970_fall_s1/s1_31', 'ROIs1970_fall_s1/s1_149', 'ROIs1970_fall_s1/s1_64', 
                               'ROIs1970_fall_s1/s1_28', 'ROIs1970_fall_s1/s1_144', 'ROIs1970_fall_s1/s1_57', 'ROIs1970_fall_s1/s1_35', 'ROIs1970_fall_s1/s1_133', 'ROIs1970_fall_s1/s1_30', 
                               'ROIs1970_fall_s1/s1_134', 'ROIs1970_fall_s1/s1_141', 'ROIs1970_fall_s1/s1_112', 'ROIs1970_fall_s1/s1_116', 'ROIs1970_fall_s1/s1_37', 'ROIs1970_fall_s1/s1_26', 
                               'ROIs1970_fall_s1/s1_77', 'ROIs1970_fall_s1/s1_100', 'ROIs1970_fall_s1/s1_83', 'ROIs1970_fall_s1/s1_71', 'ROIs1970_fall_s1/s1_93', 'ROIs1970_fall_s1/s1_119', 
                               'ROIs1970_fall_s1/s1_104', 'ROIs1970_fall_s1/s1_136', 'ROIs1970_fall_s1/s1_6', 'ROIs1970_fall_s1/s1_41', 'ROIs1970_fall_s1/s1_125', 'ROIs1970_fall_s1/s1_91', 
                               'ROIs1970_fall_s1/s1_131', 'ROIs1970_fall_s1/s1_120', 'ROIs1970_fall_s1/s1_110', 'ROIs1970_fall_s1/s1_19', 'ROIs1970_fall_s1/s1_14', 'ROIs1970_fall_s1/s1_81', 
                               'ROIs1970_fall_s1/s1_39', 'ROIs1970_fall_s1/s1_109', 'ROIs1970_fall_s1/s1_33', 'ROIs1970_fall_s1/s1_88', 'ROIs1970_fall_s1/s1_11', 'ROIs1970_fall_s1/s1_128', 
                               'ROIs1970_fall_s1/s1_142', 'ROIs1970_fall_s1/s1_122', 'ROIs1970_fall_s1/s1_4', 'ROIs1970_fall_s1/s1_27', 'ROIs1970_fall_s1/s1_147', 'ROIs1970_fall_s1/s1_85', 
                               'ROIs1970_fall_s1/s1_82', 'ROIs1970_fall_s1/s1_105', 'ROIs1158_spring_s1/s1_9', 'ROIs1158_spring_s1/s1_1', 'ROIs1158_spring_s1/s1_124', 'ROIs1158_spring_s1/s1_40', 
                               'ROIs1158_spring_s1/s1_101', 'ROIs1158_spring_s1/s1_21', 'ROIs1158_spring_s1/s1_134', 'ROIs1158_spring_s1/s1_145', 'ROIs1158_spring_s1/s1_141', 'ROIs1158_spring_s1/s1_66', 
                               'ROIs1158_spring_s1/s1_8', 'ROIs1158_spring_s1/s1_26', 'ROIs1158_spring_s1/s1_77', 'ROIs1158_spring_s1/s1_113', 'ROIs1158_spring_s1/s1_100', 
                               'ROIs1158_spring_s1/s1_117', 'ROIs1158_spring_s1/s1_119', 'ROIs1158_spring_s1/s1_6', 'ROIs1158_spring_s1/s1_58', 'ROIs1158_spring_s1/s1_120', 'ROIs1158_spring_s1/s1_110', 
                               'ROIs1158_spring_s1/s1_126', 'ROIs1158_spring_s1/s1_115', 'ROIs1158_spring_s1/s1_121', 'ROIs1158_spring_s1/s1_39', 'ROIs1158_spring_s1/s1_109', 'ROIs1158_spring_s1/s1_63', 
                               'ROIs1158_spring_s1/s1_75', 'ROIs1158_spring_s1/s1_132', 'ROIs1158_spring_s1/s1_128', 'ROIs1158_spring_s1/s1_142', 'ROIs1158_spring_s1/s1_15', 'ROIs1158_spring_s1/s1_45', 
                               'ROIs1158_spring_s1/s1_97', 'ROIs1158_spring_s1/s1_147', 'ROIs1868_summer_s1/s1_90', 'ROIs1868_summer_s1/s1_87', 'ROIs1868_summer_s1/s1_25', 'ROIs1868_summer_s1/s1_124', 
                               'ROIs1868_summer_s1/s1_114', 'ROIs1868_summer_s1/s1_135', 'ROIs1868_summer_s1/s1_40', 'ROIs1868_summer_s1/s1_101', 'ROIs1868_summer_s1/s1_42', 
                               'ROIs1868_summer_s1/s1_31', 'ROIs1868_summer_s1/s1_36', 'ROIs1868_summer_s1/s1_139', 'ROIs1868_summer_s1/s1_56', 'ROIs1868_summer_s1/s1_133', 'ROIs1868_summer_s1/s1_55', 
                               'ROIs1868_summer_s1/s1_43', 'ROIs1868_summer_s1/s1_113', 'ROIs1868_summer_s1/s1_100', 'ROIs1868_summer_s1/s1_76', 'ROIs1868_summer_s1/s1_123', 'ROIs1868_summer_s1/s1_143', 
                               'ROIs1868_summer_s1/s1_93', 'ROIs1868_summer_s1/s1_125', 'ROIs1868_summer_s1/s1_89', 'ROIs1868_summer_s1/s1_120', 'ROIs1868_summer_s1/s1_126', 'ROIs1868_summer_s1/s1_72', 
                               'ROIs1868_summer_s1/s1_115', 'ROIs1868_summer_s1/s1_121', 'ROIs1868_summer_s1/s1_146', 'ROIs1868_summer_s1/s1_140', 'ROIs1868_summer_s1/s1_95', 
                               'ROIs1868_summer_s1/s1_102', 'ROIs1868_summer_s1/s1_7', 'ROIs1868_summer_s1/s1_11', 'ROIs1868_summer_s1/s1_132', 'ROIs1868_summer_s1/s1_15', 'ROIs1868_summer_s1/s1_137', 
                               'ROIs1868_summer_s1/s1_4', 'ROIs1868_summer_s1/s1_27', 'ROIs1868_summer_s1/s1_147', 'ROIs1868_summer_s1/s1_86', 'ROIs1868_summer_s1/s1_47', 'ROIs2017_winter_s1/s1_68', 
                               'ROIs2017_winter_s1/s1_25', 'ROIs2017_winter_s1/s1_62', 'ROIs2017_winter_s1/s1_135', 'ROIs2017_winter_s1/s1_42', 'ROIs2017_winter_s1/s1_64', 'ROIs2017_winter_s1/s1_21', 
                               'ROIs2017_winter_s1/s1_55', 'ROIs2017_winter_s1/s1_112', 'ROIs2017_winter_s1/s1_116', 'ROIs2017_winter_s1/s1_8', 'ROIs2017_winter_s1/s1_59', 'ROIs2017_winter_s1/s1_49', 
                               'ROIs2017_winter_s1/s1_104',  'ROIs2017_winter_s1/s1_81', 'ROIs2017_winter_s1/s1_146', 'ROIs2017_winter_s1/s1_75', 
                               'ROIs2017_winter_s1/s1_94', 'ROIs2017_winter_s1/s1_102', 'ROIs2017_winter_s1/s1_61', 'ROIs2017_winter_s1/s1_47']
        self.splits['val']  = ['ROIs2017_winter_s1/s1_22', 'ROIs1868_summer_s1/s1_19', 'ROIs1970_fall_s1/s1_65', 'ROIs1158_spring_s1/s1_17', 'ROIs2017_winter_s1/s1_107', 
                               'ROIs1868_summer_s1/s1_80', 'ROIs1868_summer_s1/s1_127', 'ROIs2017_winter_s1/s1_130', 'ROIs1868_summer_s1/s1_17', 'ROIs2017_winter_s1/s1_84'] 
        self.splits['test'] = ['ROIs1158_spring_s1/s1_106', 'ROIs1158_spring_s1/s1_123', 'ROIs1158_spring_s1/s1_140', 'ROIs1158_spring_s1/s1_31', 'ROIs1158_spring_s1/s1_44', 
                               'ROIs1868_summer_s1/s1_119', 'ROIs1868_summer_s1/s1_73', 'ROIs1970_fall_s1/s1_139', 'ROIs2017_winter_s1/s1_108', 'ROIs2017_winter_s1/s1_63']
        
        self.splits["all"] = (
            self.splits["train"]
            + self.splits["val"]
            + self.splits["test"]
        )

        self.paths_A, self.paths_B = self._collect_unpaired_paths()

        if len(self.paths_A) == 0 or len(self.paths_B) == 0:
            raise RuntimeError("No valid samples found.")

    # ------------------------------------------------
    # Collect UNPAIRED paths
    # ------------------------------------------------
    def _collect_unpaired_paths(self):
        paths_A = []
        paths_B = []

        seeds_S1 = natsorted([d for d in os.listdir(self.root_dir) if "_s1" in d])

        for seed in tqdm(seeds_S1, desc=f"Indexing {self.split}"):
            roi_dirs = natsorted(os.listdir(os.path.join(self.root_dir, seed)))

            for roi in roi_dirs:
                roi_dir = os.path.join(self.root_dir, seed, roi)
                s1_files = natsorted(os.listdir(roi_dir))

                for s1_file in s1_files:
                    s1_path = os.path.join(roi_dir, s1_file)

                    if not any(split_roi in s1_path for split_roi in self.splits[self.split]):
                        continue

                    s2_path = s1_path.replace("/s1", "/s2").replace("_s1", "_s2")

                    if os.path.isfile(s1_path) and os.path.isfile(s2_path):
                        paths_B.append(s1_path)
                        paths_A.append(s2_path)

        return paths_A, paths_B

    # ------------------------------------------------
    def __len__(self):
        return max(len(self.paths_A), len(self.paths_B))

    # ------------------------------------------------
    def __getitem__(self, idx):
        # --- UNPAIRED SAMPLING ---
        A_path = self.paths_A[idx % len(self.paths_A)]
        B_path = random.choice(self.paths_B)

        # --- LOAD DATA ---
        s2_tif = read_tif(A_path)
        s1_tif = read_tif(B_path)

        A = process_MS(read_img(s2_tif), self.method)
        B = process_SAR(read_img(s1_tif), self.method)

        # --- FORMAT ---
        A = torch.from_numpy(A).float()
        B = torch.from_numpy(B).float()

        # SAR → 3 channels
        if B.shape[0] == 1:
            B = B.repeat(3, 1, 1)

        # # --- NORMALIZATION (CycleGAN style) ---
        # A = (A - A.min()) / (A.max() - A.min() + 1e-6)
        # B = (B - B.min()) / (B.max() - B.min() + 1e-6)

        A = A * 2 - 1
        B = B * 2 - 1

        return {
            "A": A[[3, 2, 1], :, :],
            "B": B,
            "A_paths": A_path,
            "B_paths": B_path,
        }
