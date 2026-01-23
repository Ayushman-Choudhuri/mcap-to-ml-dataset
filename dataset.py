import h5py
import torch
import numpy as np
import logging
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Any, Optional

from pipeline.config import (
    TIMESTAMP_DATASET_PATH,
    LIDAR_DATA_DATASET_PATH,
    LIDAR_OFFSETS_DATASET_PATH,
    LIDAR_COUNTS_DATASET_PATH,
    CAMERA_IMAGES_DATASET_PATH,
    NUM_SAMPLES_ATTRIBUTE
)

class KittiHDF5Dataset(Dataset):
    def __init__(self, h5FilePath: str, transform: Optional[Any] = None):
        self.h5FilePath = h5FilePath
        self.transform = transform
        self.logger = logging.getLogger(__name__)
        
        with h5py.File(self.h5_path, 'r') as f:
            if NUM_SAMPLES_ATTRIBUTE not in f.attrs:
                raise KeyError(f"HDF5 file missing {NUM_SAMPLES_ATTRIBUTE} attribute.")
            self.length = f.attrs[NUM_SAMPLES_ATTRIBUTE]

        self.h5File = None

    def _init_db(self):
        if self.h5File is None:
            self.h5File = h5py.File(self.h5FilePath, 'r', libver='latest', swmr=True)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        self._init_db()
        
        image = self.h5File[CAMERA_IMAGES_DATASET_PATH][index]
        
        offset = self.h5File[LIDAR_OFFSETS_DATASET_PATH][index]
        count = self.h5File[LIDAR_COUNTS_DATASET_PATH][index]
        lidar = self.h5File[LIDAR_DATA_DATASET_PATH][offset : offset + count]
        
        timestamp = self.h5File[TIMESTAMP_DATASET_PATH][index]

        imageTensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0        
        lidarTensor = torch.from_numpy(lidar).float()
        
        if self.transform:
            imageTensor = self.transform(imageTensor)

        return {
            "image": imageTensor,
            "lidar": lidarTensor,
            "timestamp": torch.tensor(timestamp, dtype=torch.float64),
            "index": index
        }

def kitti_collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """
    Custom collate function to handle variable-length LiDAR tensors.
    Standard stacking fails for LiDAR because N points varies per frame.
    """
    images = torch.stack([item['image'] for item in batch])
    timestamps = torch.stack([item['timestamp'] for item in batch])
    indices = [item['index'] for item in batch]
    
    # We keep lidar as a list of tensors. 
    # Alternatively, you could pad them to the max N in the batch.
    lidars = [item['lidar'] for item in batch]
    
    return {
        "images": images,
        "lidars": lidars,
        "timestamps": timestamps,
        "indices": indices
    }

# Example Usage
if __name__ == "__main__":
    dataset = KittiHDF5Dataset("data/processed/chunks.hdf5")
    
    # Senior-level tip: Mention why num_workers > 0 requires the _init_db pattern
    loader = DataLoader(
        dataset, 
        batch_size=8, 
        shuffle=True, 
        num_workers=4, 
        collate_fn=kitti_collate_fn,
        pin_memory=True  # Speed up transfer to GPU
    )
    
    for batch in loader:
        print(f"Batch loaded: Image shape {batch['images'].shape}, "
              f"Lidar frames in batch: {len(batch['lidars'])}")
        break