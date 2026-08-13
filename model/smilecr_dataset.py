
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional, Dict, Any
import glob
from pathlib import Path
import logging
from skimage.transform import resize
import cv2
import rasterio

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SMILECRNumpyDataset(Dataset):
    """
    SMILE-CR Dataset that loads from NPY files instead of TIFF files.
    Maintains the same interface as SMILECRDataset but loads much faster.
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        use_modis: bool = True,
        use_sentinel1: bool = True,
        use_mask: bool = True,
        transform: Optional[Any] = None,
        # normalize: bool = True,
        return_filename: bool = False
    ):
        """Initialize SMILE-CR NPY Dataset."""
        self.data_root = Path(data_root)
        self.split = split.lower()
        self.use_modis = use_modis
        self.use_sentinel1 = use_sentinel1
        self.use_mask = use_mask
        self.transform = transform
        # self.normalize = normalize
        self.return_filename = return_filename
        
        # Set up paths based on split
        if self.split == 'train':
            self.data_dir = self.data_root / 'TrainData' / 'TrainData'
        elif self.split == 'val':
            self.data_dir = self.data_root / 'ValData' / 'ValData'
        elif self.split == 'test':
            self.data_dir = self.data_root / 'TestData' / 'TestData'
        else:
            raise ValueError(f"Split must be 'train', 'val', or 'test', got {split}")
            
        # Define subdirectories
        self.cloudy_dir = self.data_dir / 'CloudLandsat_2020'
        self.clear_dir = self.data_dir / 'Landsat-8_2020'
        self.modis_dir = self.data_dir / 'MODIS_2020'
        self.mask_dir = self.data_dir / 'Mask'
        self.sentinel1_dir = self.data_dir / 'Sentinel-1_2020-De'
        self.translated_dir = self.data_dir / 'Translated'
        
        # Get file list
        self.file_list = self._get_file_list()
        
        logger.info(f"Initialized SMILE-CR NPY {split} dataset with {len(self.file_list)} samples")
    
    def _get_file_list(self) -> list:
        """Get list of available files."""
        pattern = str(self.cloudy_dir / "*.npy")
        files = glob.glob(pattern)
        file_list = [Path(f).stem for f in files]
        file_list.sort()
        return file_list
    
    def _normalize_data(self, data: np.ndarray, data_type: str = 'optical') -> np.ndarray:
        """Normalize data based on type."""
        if not self.normalize:
            return data
            
        if data_type == 'optical':
            data = np.clip(data / 3000.0, 0, 1)
        elif data_type == 'sar':
            data = np.clip((data + 30) / 50, 0, 1)
        elif data_type == 'mask':
            data = np.clip(data, 0, 1)
        
        return data
    
    def __len__(self) -> int:
        return len(self.file_list)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample from the dataset."""
        filename = self.file_list[idx]
        sample = {}
        
        # Load cloudy image
        cloudy_path = self.cloudy_dir / f"{filename}.npy"
        cloudy_data = np.load(cloudy_path)
        # cloudy_data = self._normalize_data(cloudy_data, 'optical')
        # sample['input'] = torch.from_numpy(cloudy_data)
        # Resize to 256x256 if needed
        if cloudy_data.shape[-2:] != (256, 256):
            cloudy_data = resize(cloudy_data, (*cloudy_data.shape[:-2], 256, 256), order=1, preserve_range=True, anti_aliasing=True).astype(cloudy_data.dtype)
        sample['input'] = torch.from_numpy(cloudy_data)
        
        # Load clear image
        clear_path = self.clear_dir / f"{filename}.npy"
        if clear_path.exists():
            clear_data = np.load(clear_path)
            # clear_data = self._normalize_data(clear_data, 'optical')
            # sample['target'] = torch.from_numpy(clear_data)
            if clear_data.shape[-2:] != (256, 256):
                clear_data = resize(clear_data, (*clear_data.shape[:-2], 256, 256), order=1, preserve_range=True, anti_aliasing=True).astype(clear_data.dtype)
            sample['target'] = torch.from_numpy(clear_data)
        else:
            sample['target'] = torch.zeros_like(sample['input'])
        
        # Load translated image
        trans_path = self.translated_dir / f"{filename}.png"
        trans_path = str(trans_path)
        trans_img = cv2.imread(trans_path, cv2.IMREAD_COLOR)
        if trans_img is None:
            raise FileNotFoundError(f"Image not found at path: {trans_path}")
        trans_img = cv2.cvtColor(trans_img, cv2.COLOR_BGR2RGB).astype('float32') / 255.0
        trans_img = torch.from_numpy(trans_img.transpose(2, 0, 1))
        sample['trans'] = trans_img

        # Load MODIS data
        if self.use_modis:
            modis_path = self.modis_dir / f"{filename}.npy"
            if modis_path.exists():
                modis_data = np.load(modis_path)
                # modis_data = self._normalize_data(modis_data, 'optical')
                # sample['modis'] = torch.from_numpy(modis_data)
                if modis_data.shape[-2:] != (256, 256):
                    modis_data = resize(modis_data, (*modis_data.shape[:-2], 256, 256), order=1, preserve_range=True, anti_aliasing=True).astype(modis_data.dtype)
                sample['modis'] = torch.from_numpy(modis_data)
            else:
                sample['modis'] = torch.zeros_like(sample['input'])
        

        # Load cloud mask
        if self.use_mask:
            mask_path = self.mask_dir / f"{filename}.npy"
            if mask_path.exists():
                mask_data = np.load(mask_path)
                # mask_data = self._normalize_data(mask_data, 'mask')
                if mask_data.shape[-2:] != (256, 256):
                    mask_data = resize(mask_data, (*mask_data.shape[:-2], 256, 256), order=1, preserve_range=True, anti_aliasing=True).astype(mask_data.dtype)
                sample['mask'] = torch.from_numpy(mask_data)
            else:
                sample['mask'] = torch.ones((1,) + sample['input'].shape[1:])
        
        # Load Sentinel-1 data
        if self.use_sentinel1:
            sentinel1_path = self.sentinel1_dir / f"{filename}.npy"
            if sentinel1_path.exists():
                sentinel1_data = np.load(sentinel1_path)
                # sentinel1_data = self._normalize_data(sentinel1_data, 'sar')
                if sentinel1_data.shape[-2:] != (256, 256):
                    sentinel1_data = resize(sentinel1_data, (*sentinel1_data.shape[:-2], 256, 256), order=1, preserve_range=True, anti_aliasing=True).astype(sentinel1_data.dtype)
                sample['sar'] = torch.from_numpy(sentinel1_data)
            else:
                sample['sar'] = torch.zeros((2,) + sample['input'].shape[1:])
        
        # Add filename if requested
        if self.return_filename:
            sample['filename'] = filename
        
        # Apply transforms if provided
        if self.transform:
            sample = self.transform(sample)
        # Skip sample if mask is all zeros
        # if self.split == 'train' and self.use_mask and 'mask' in sample and torch.all(sample['mask'] == 0):
        #     return self.__getitem__((idx + 1) % len(self))
        # Check if sample['input'] contains any NaN values
        if torch.isnan(sample['input']).any():
            # logger.warning(f"Sample '{filename}' contains NaN values in 'input'. Skipping.")
            return self.__getitem__((idx + 1) % len(self))
        return sample


class SMILECRNumpyDataset_1(Dataset):
    """
    SMILE-CR Dataset that loads from NPY files instead of TIFF files.
    Maintains the same interface as SMILECRDataset but loads much faster.
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        use_modis: bool = True,
        use_sentinel1: bool = True,
        use_mask: bool = True,
        transform: Optional[Any] = None,
        # normalize: bool = True,
        return_filename: bool = False
    ):
        """Initialize SMILE-CR NPY Dataset."""
        self.data_root = Path(data_root)
        self.split = split.lower()
        self.use_modis = use_modis
        self.use_sentinel1 = use_sentinel1
        self.use_mask = use_mask
        self.transform = transform
        # self.normalize = normalize
        self.return_filename = return_filename
        
        # Set up paths based on split
        if self.split == 'train':
            self.data_dir = self.data_root / 'TrainData' / 'TrainData'
        elif self.split == 'val':
            self.data_dir = self.data_root / 'ValData' / 'ValData'
        elif self.split == 'test':
            self.data_dir = self.data_root / 'TestData' / 'TestData'
        else:
            raise ValueError(f"Split must be 'train', 'val', or 'test', got {split}")
            
        # Define subdirectories
        self.cloudy_dir = self.data_dir / 'CloudLandsat_2020'
        self.clear_dir = self.data_dir / 'Landsat-8_2020'
        self.modis_dir = self.data_dir / 'MODIS_2020'
        self.mask_dir = self.data_dir / 'Mask'
        self.sentinel1_dir = self.data_dir / 'Sentinel-1_2020-De'
        if self.split == 'train':
            self.translated_dir = self.data_dir / 'SMILECR_train_translated_full_npy'
        elif self.split == 'val':
            self.translated_dir = self.data_dir / 'SMILECR_val_translated_full_npy'
        elif self.split == 'test':
            self.translated_dir = self.data_dir / 'SMILECR_test_translated_full_npy'
        

        # Get file list
        self.file_list = self._get_file_list()
        
        logger.info(f"Initialized SMILE-CR NPY {split} dataset with {len(self.file_list)} samples")
    
    def _get_file_list(self) -> list:
        """Get list of available files."""
        pattern = str(self.cloudy_dir / "*.npy")
        files = glob.glob(pattern)
        file_list = [Path(f).stem for f in files]
        file_list.sort()
        return file_list
    
    def __len__(self) -> int:
        return len(self.file_list)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample from the dataset."""
        filename = self.file_list[idx]
        sample = {}
        
        # Load cloudy image
        cloudy_path = self.cloudy_dir / f"{filename}.npy"
        cloudy_data = np.load(cloudy_path)

        # Resize to 256x256 if needed
        if cloudy_data.shape[-2:] != (256, 256):
            cloudy_data = resize(cloudy_data, (*cloudy_data.shape[:-2], 256, 256), order=1, preserve_range=True, anti_aliasing=True).astype(cloudy_data.dtype)
        sample['input'] = torch.from_numpy(cloudy_data)
        
        # Load clear image
        clear_path = self.clear_dir / f"{filename}.npy"
        if clear_path.exists():
            clear_data = np.load(clear_path)

            if clear_data.shape[-2:] != (256, 256):
                clear_data = resize(clear_data, (*clear_data.shape[:-2], 256, 256), order=1, preserve_range=True, anti_aliasing=True).astype(clear_data.dtype)
            sample['target'] = torch.from_numpy(clear_data)
        else:
            sample['target'] = torch.zeros_like(sample['input'])
        
        # Load translated image
        # trans_path = self.translated_dir / f"{filename}.png"
        trans_path = self.translated_dir / f"{filename}.npy"
        trans_path = str(trans_path)
        trans_img = np.load(trans_path) / 255.0 
        if trans_img is None:
            raise FileNotFoundError(f"Image not found at path: {trans_path}")
        # trans_img = cv2.cvtColor(trans_img, cv2.COLOR_BGR2RGB).astype('float32') / 255.0
        trans_img = torch.from_numpy(trans_img.transpose(2, 0, 1).astype(cloudy_data.dtype))
        sample['trans'] = trans_img

        # Load MODIS data
        if self.use_modis:
            modis_path = self.modis_dir / f"{filename}.npy"
            if modis_path.exists():
                modis_data = np.load(modis_path)
                # modis_data = self._normalize_data(modis_data, 'optical')
                # sample['modis'] = torch.from_numpy(modis_data)
                if modis_data.shape[-2:] != (256, 256):
                    modis_data = resize(modis_data, (*modis_data.shape[:-2], 256, 256), order=1, preserve_range=True, anti_aliasing=True).astype(modis_data.dtype)
                sample['modis'] = torch.from_numpy(modis_data)
            else:
                sample['modis'] = torch.zeros_like(sample['input'])
        

        # Load cloud mask
        if self.use_mask:
            mask_path = self.mask_dir / f"{filename}.npy"
            if mask_path.exists():
                mask_data = np.load(mask_path)
                # mask_data = self._normalize_data(mask_data, 'mask')
                if mask_data.shape[-2:] != (256, 256):
                    mask_data = resize(mask_data, (*mask_data.shape[:-2], 256, 256), order=1, preserve_range=True, anti_aliasing=True).astype(mask_data.dtype)
                sample['mask'] = torch.from_numpy(mask_data)
            else:
                sample['mask'] = torch.ones((1,) + sample['input'].shape[1:])
        
        # Load Sentinel-1 data
        if self.use_sentinel1:
            sentinel1_path = self.sentinel1_dir / f"{filename}.npy"
            if sentinel1_path.exists():
                sentinel1_data = np.load(sentinel1_path)
                # sentinel1_data = self._normalize_data(sentinel1_data, 'sar')
                if sentinel1_data.shape[-2:] != (256, 256):
                    sentinel1_data = resize(sentinel1_data, (*sentinel1_data.shape[:-2], 256, 256), order=1, preserve_range=True, anti_aliasing=True).astype(sentinel1_data.dtype)
                sample['sar'] = torch.from_numpy(sentinel1_data)
            else:
                sample['sar'] = torch.zeros((2,) + sample['input'].shape[1:])
        
        # Add filename if requested
        if self.return_filename:
            sample['filename'] = filename
        
        # Apply transforms if provided
        if self.transform:
            sample = self.transform(sample)
        # Skip sample if mask is all zeros
        # if self.split == 'train' and self.use_mask and 'mask' in sample and torch.all(sample['mask'] == 0):
        #     return self.__getitem__((idx + 1) % len(self))
        # Check if sample['input'] contains any NaN values
        if torch.isnan(sample['input']).any():
            # logger.warning(f"Sample '{filename}' contains NaN values in 'input'. Skipping.")
            return self.__getitem__((idx + 1) % len(self))
        return sample
