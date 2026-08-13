import os
import numpy as np
import torch
from torch.utils.data import Dataset

class SEN12MSCR_1(Dataset):
    # The dataset needs to have the following structure:
    # SEN12MS-CR_1/
    #     |- SEN12MSCR_train_input_npy/
    #     |- SEN12MSCR_train_gt_npy/
    #     |- SEN12MSCR_train_sar_npy/
    #     |- SEN12MSCR_train_mask_npy/
    #     |- SEN12MSCR_train_translated_partial_npy/
    #     |- SEN12MSCR_train_input_npy_full/
    #     |- SEN12MSCR_train_gt_npy_full/
    #     |- SEN12MSCR_train_sar_npy_full/
    #     |- SEN12MSCR_train_mask_npy_full/
    #     |- SEN12MSCR_train_translated_full_npy/
    #     ...
    # The dataset is split into train, test, and val sets.
    # To use the dataset, the user needs to specify the root directory and the split type.
    # The dataset will return a dictionary containing the input, target, sar, and trans images.
    # Input images are stored in the 'input' key, target images are stored in the 'target' key,
    # sar images are stored in the 'sar' key.
    # For example, sample = {'input': input_img, 'target': target_img, 'sar': sar_img, 'filename': filename}
    # input_img = sample['input']
    # target_img = sample['target']
    # sar_img = sample['sar']
    
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
                self.mask_dir = os.path.join(self.root_dir, 'SEN12MSCR_train_mask_npy_full')
                self.trans_dir = os.path.join(self.root_dir, 'SEN12MSCR_train_translated_full_npy')
            elif self.data_mode == 'partial':
                self.input_dir = os.path.join(self.root_dir, 'SEN12MSCR_train_input_npy')
                self.target_dir = os.path.join(self.root_dir, 'SEN12MSCR_train_gt_npy')
                self.sar_dir = os.path.join(self.root_dir, 'SEN12MSCR_train_sar_npy')
                self.trans_dir = os.path.join(self.root_dir, 'SEN12MSCR_train_translated_partial_npy')
                self.mask_dir = os.path.join(self.root_dir, 'SEN12MSCR_train_mask_npy')
            else:
                raise ValueError("Invalid data_mode. Choose either 'partial' or 'full'.")
        elif self.split == 'test':
            if self.data_mode == 'full':
                self.input_dir = os.path.join(self.root_dir, 'SEN12MSCR_test_input_npy_full')
                self.target_dir = os.path.join(self.root_dir, 'SEN12MSCR_test_gt_npy_full')
                self.sar_dir = os.path.join(self.root_dir, 'SEN12MSCR_test_sar_npy_full')
                self.trans_dir = os.path.join(self.root_dir, 'SEN12MSCR_test_translated_full_npy')
                self.mask_dir = os.path.join(self.root_dir, 'SEN12MSCR_test_mask_npy_full')
            elif self.data_mode == 'partial':
                self.input_dir = os.path.join(self.root_dir, 'SEN12MSCR_test_input_npy')
                self.target_dir = os.path.join(self.root_dir, 'SEN12MSCR_test_gt_npy')
                self.sar_dir = os.path.join(self.root_dir, 'SEN12MSCR_test_sar_npy')
                self.trans_dir = os.path.join(self.root_dir, 'SEN12MSCR_test_translated_partial_npy')
                self.mask_dir = os.path.join(self.root_dir, 'SEN12MSCR_test_mask_npy')
            else:
                raise ValueError("Invalid data_mode. Choose either 'partial' or 'full'.")
        elif self.split == 'val':
            self.input_dir = os.path.join(self.root_dir, 'SEN12MSCR_val_input_npy')
            self.target_dir = os.path.join(self.root_dir, 'SEN12MSCR_val_gt_npy')
            self.sar_dir = os.path.join(self.root_dir, 'SEN12MSCR_val_sar_npy')
            if self.data_mode == 'full':
                self.trans_dir = os.path.join(self.root_dir, 'SEN12MSCR_val_translated_full_npy')
            elif self.data_mode == 'partial':
                self.trans_dir = os.path.join(self.root_dir, 'SEN12MSCR_val_translated_partial_npy')
            self.mask_dir = os.path.join(self.root_dir, 'SEN12MSCR_val_mask_npy')
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
            trans_path = os.path.join(self.trans_dir, filename)
            trans_img = torch.from_numpy(np.load(trans_path).astype('float32').transpose(2, 0, 1)) / 255.0
            sample = {'input': input_img, 'target': target_img, 'sar': sar_img, 'trans': trans_img, 'filename': filename}
            return sample
        elif self.split == 'test':
            filename = os.listdir(self.input_dir)[idx]
            input_path = os.path.join(self.input_dir, filename)
            target_path = os.path.join(self.target_dir, filename)
            sar_path = os.path.join(self.sar_dir, filename)
            input_img = torch.from_numpy(np.load(input_path).astype('float32'))
            target_img = torch.from_numpy(np.load(target_path).astype('float32'))
            sar_img = torch.from_numpy(np.load(sar_path).astype('float32'))
            trans_path = os.path.join(self.trans_dir, filename)
            trans_img = torch.from_numpy(np.load(trans_path).astype('float32').transpose(2, 0, 1)) / 255.0
            sample = {'input': input_img, 'target': target_img, 'sar': sar_img, 'trans': trans_img, 'filename': filename}            
            return sample
        elif self.split == 'val':
            filename = os.listdir(self.input_dir)[idx]
            input_path = os.path.join(self.input_dir, filename)
            target_path = os.path.join(self.target_dir, filename)
            sar_path = os.path.join(self.sar_dir, filename)
            input_img = torch.from_numpy(np.load(input_path).astype('float32'))
            target_img = torch.from_numpy(np.load(target_path).astype('float32'))
            sar_img = torch.from_numpy(np.load(sar_path).astype('float32'))
            trans_path = os.path.join(self.trans_dir, filename)
            trans_img = torch.from_numpy(np.load(trans_path).astype('float32').transpose(2, 0, 1)) / 255.0
            sample = {'input': input_img, 'target': target_img, 'sar': sar_img, 'trans': trans_img, 'filename': filename}
            return sample
        else:
            raise ValueError("Unknown split type!")

    def load_npy(self, input_path):
        input_img = torch.load(input_path)
        return input_img
  

