
"""
Data Preprocessing & Dataset Loader for Flood Detection
Handles Sentinel-1 SAR preprocessing, tiling, and augmentation
"""

import os
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.mask import mask
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
import torch


# ==================== SAR PREPROCESSING ====================

def preprocess_sar(image_path, output_path=None, apply_filter=True):
    """
    Preprocess Sentinel-1 SAR imagery:
    1. Read VV/VH polarizations
    2. Apply speckle filtering (Lee filter)
    3. Convert to dB scale (10*log10)
    4. Normalize to [0, 1]
    """
    with rasterio.open(image_path) as src:
        bands = src.read()
        profile = src.profile
        transform = src.transform
        crs = src.crs
        bounds = src.bounds

    # Sentinel-1 is typically in linear power scale
    # Convert to dB: 10 * log10(power)
    # Handle zeros by adding small epsilon
    bands_db = 10 * np.log10(bands + 1e-10)

    # Clip extreme values (Sentinel-1 typical range: -30 to 0 dB)
    bands_db = np.clip(bands_db, -35, 0)

    # Normalize to [0, 1]
    bands_norm = (bands_db + 35) / 35

    if apply_filter:
        bands_norm = lee_filter(bands_norm)

    if output_path:
        profile.update(dtype=rasterio.float32, count=bands_norm.shape[0])
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(bands_norm.astype(np.float32))

    return bands_norm, transform, crs, bounds


def lee_filter(img, window_size=7):
    """Lee speckle filter for SAR imagery"""
    from scipy.ndimage import uniform_filter

    img_mean = uniform_filter(img, window_size)
    img_sqr_mean = uniform_filter(img**2, window_size)
    img_variance = img_sqr_mean - img_mean**2

    overall_variance = np.var(img)

    img_weights = img_variance / (img_variance + overall_variance + 1e-10)
    img_filtered = img_mean + img_weights * (img - img_mean)

    return img_filtered


def create_patches(image, mask, patch_size=256, stride=128):
    """
    Create overlapping patches from large satellite images
    Returns list of (image_patch, mask_patch) tuples
    """
    h, w = image.shape[-2:]
    patches = []

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            if len(image.shape) == 3:
                img_patch = image[:, y:y+patch_size, x:x+patch_size]
            else:
                img_patch = image[y:y+patch_size, x:x+patch_size]

            if mask is not None:
                mask_patch = mask[y:y+patch_size, x:x+patch_size]
                patches.append((img_patch, mask_patch))
            else:
                patches.append(img_patch)

    return patches


# ==================== DATASET CLASS ====================

class FloodDataset(Dataset):
    """
    PyTorch Dataset for Flood Segmentation

    Args:
        image_dir: Directory containing preprocessed SAR/Optical images (.npy or .tif)
        mask_dir: Directory containing binary flood masks
        transform: Albumentations augmentation pipeline
        mode: 'train', 'val', or 'test'
    """
    def __init__(self, image_dir, mask_dir=None, transform=None, mode='train'):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.mode = mode

        # List all image files
        self.image_files = sorted([
            f for f in os.listdir(image_dir) 
            if f.endswith(('.npy', '.tif', '.tiff', '.png'))
        ])

        # Default augmentation pipeline
        if transform is None:
            if mode == 'train':
                self.transform = A.Compose([
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
                    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
                    A.GaussNoise(var_limit=(0.001, 0.005), p=0.2),
                    A.Normalize(mean=[0.5], std=[0.5]),  # For single-channel SAR
                    ToTensorV2(),
                ])
            else:
                self.transform = A.Compose([
                    A.Normalize(mean=[0.5], std=[0.5]),
                    ToTensorV2(),
                ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # Load image
        if img_path.endswith('.npy'):
            image = np.load(img_path).astype(np.float32)
        else:
            with rasterio.open(img_path) as src:
                image = src.read().astype(np.float32)

        # Ensure channel-first format (C, H, W)
        if image.ndim == 2:
            image = np.expand_dims(image, axis=0)
        elif image.ndim == 3 and image.shape[-1] in [1, 2, 3, 13]:
            image = np.transpose(image, (2, 0, 1))

        # Load mask if available
        if self.mask_dir:
            mask_name = img_name.replace('.tif', '_mask.png').replace('.npy', '_mask.npy')
            mask_path = os.path.join(self.mask_dir, mask_name)

            if os.path.exists(mask_path):
                if mask_path.endswith('.npy'):
                    mask = np.load(mask_path).astype(np.float32)
                else:
                    mask = rasterio.open(mask_path).read(1).astype(np.float32)

                mask = np.clip(mask, 0, 1)

                # Apply same spatial augmentations to image and mask
                if self.transform:
                    transformed = self.transform(image=image.transpose(1, 2, 0), mask=mask)
                    image = transformed['image']
                    mask = transformed['mask'].unsqueeze(0)

                return image, mask
            else:
                # No mask - return dummy mask for inference
                if self.transform:
                    transformed = self.transform(image=image.transpose(1, 2, 0))
                    image = transformed['image']

                dummy_mask = torch.zeros((1, image.shape[1], image.shape[2]))
                return image, dummy_mask
        else:
            # Inference mode - no masks
            if self.transform:
                transformed = self.transform(image=image.transpose(1, 2, 0))
                image = transformed['image']

            return image


def get_dataloaders(train_dir, val_dir, test_dir=None, 
                    batch_size=8, num_workers=4, patch_size=256):
    """Create train/val/test DataLoaders"""

    train_dataset = FloodDataset(
        image_dir=os.path.join(train_dir, 'images'),
        mask_dir=os.path.join(train_dir, 'masks'),
        mode='train'
    )

    val_dataset = FloodDataset(
        image_dir=os.path.join(val_dir, 'images'),
        mask_dir=os.path.join(val_dir, 'masks'),
        mode='val'
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = None
    if test_dir:
        test_dataset = FloodDataset(
            image_dir=os.path.join(test_dir, 'images'),
            mask_dir=os.path.join(test_dir, 'masks') if os.path.exists(os.path.join(test_dir, 'masks')) else None,
            mode='test'
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True
        )

    return train_loader, val_loader, test_loader


# ==================== DATASET DOWNLOAD HELPERS ====================

def download_sen1floods11(data_root="./data/sen1floods11"):
    """
    Download Sen1Floods11 dataset using wget
    Requires: pip install gdown (for Google Drive links)
    """
    import urllib.request
    import zipfile

    os.makedirs(data_root, exist_ok=True)

    # Sen1Floods11 is available via Google Drive
    # Manual download link: https://drive.google.com/drive/folders/...
    # Or use the direct S3 bucket if available

    print("Sen1Floods11 dataset:")
    print("  1. Visit: https://github.com/cloudtostreet/Sen1Floods11")
    print("  2. Download the dataset manually or use gdown")
    print("  3. Extract to:", data_root)
    print("\nAlternative: Use NASA Earthdata API for Sentinel-1 downloads")


def download_sentinel_data(aoi, start_date, end_date, output_dir="./data/raw"):
    """
    Download Sentinel-1/2 data using SentinelHub
    Requires: pip install sentinelhub

    Args:
        aoi: Area of Interest as GeoJSON or BBox tuple (min_lon, min_lat, max_lon, max_lat)
        start_date, end_date: Date strings 'YYYY-MM-DD'
    """
    try:
        from sentinelhub import (
            SentinelHubRequest, DataCollection, MimeType, 
            CRS, BBox, bbox_to_dimensions, SHConfig
        )

        config = SHConfig()
        # Set your credentials
        # config.sh_client_id = 'YOUR_CLIENT_ID'
        # config.sh_client_secret = 'YOUR_CLIENT_SECRET'

        resolution = 10  # meters
        bbox = BBox(bbox=aoi, crs=CRS.WGS84)
        size = bbox_to_dimensions(bbox, resolution=resolution)

        # Sentinel-1 SAR request
        s1_request = SentinelHubRequest(
            evalscript="""
            //VERSION=3
            function setup() {
                return {
                    input: ["VV", "VH"],
                    output: {bands: 2}
                };
            }
            function evaluatePixel(sample) {
                return [10 * Math.log10(sample.VV + 0.001), 
                        10 * Math.log10(sample.VH + 0.001)];
            }
            """,
            input_data=[SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL1_IW,
                time_interval=(start_date, end_date)
            )],
            responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
            bbox=bbox,
            size=size,
            config=config
        )

        s1_data = s1_request.get_data()

        # Save
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"sentinel1_{start_date}_{end_date}.tif")

        with rasterio.open(
            output_path, 'w',
            driver='GTiff',
            height=size[1],
            width=size[0],
            count=2,
            dtype=s1_data[0].dtype,
            crs='EPSG:4326',
            transform=from_bounds(*bbox, size[0], size[1])
        ) as dst:
            dst.write(s1_data[0].transpose(2, 0, 1))

        print(f"Saved Sentinel-1 data to {output_path}")
        return output_path

    except ImportError:
        print("Please install sentinelhub: pip install sentinelhub")
        return None


if __name__ == "__main__":
    # Example: Create dummy dataset structure
    print("Dataset structure should look like:")
    print("""
    data/
    ├── train/
    │   ├── images/     # Preprocessed SAR patches (.npy or .tif)
    │   └── masks/      # Binary flood masks
    ├── val/
    │   ├── images/
    │   └── masks/
    └── test/
        ├── images/
        └── masks/
    """)
