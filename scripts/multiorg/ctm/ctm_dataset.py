"""
CTM Dataset v2: Proper crop matching via cache_key.

Two modes:
1. Phase 2 mode: Load from vlm_mask_results.json (100 entries, cloud VM testing)
2. Full mode: Load from SAM2 results + crop metadata JSON (16198 entries, 冬生's machine)

Crop naming: {cache_key}.png where cache_key = {Class}_{Plate}_{image}_{det_idx}
"""
import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms


class OrganoidCTMDataset(Dataset):
    """
    Dataset for CTM training/eval.
    
    Each item:
        image: [3, 224, 224] tensor (DINOv2 input)
        label: 0 (FP) or 1 (TP)
        confidence: RF-DETR confidence score
        metadata: dict with bbox, image_name, etc.
    """
    def __init__(
        self,
        metadata_path: str,
        crops_dir: str,
        split: str = 'train',
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        img_size: int = 224,
        augment: bool = False,
        balance: bool = True,
        seed: int = 42,
    ):
        self.img_size = img_size
        self.augment = augment
        
        # Load metadata — detect format by data type, not filename
        # List format: flat array of {cache_key, matched, bbox, ...} (from generate_crops or Phase 2 VLM)
        # Dict format: {per_image: [{image, detections: [...]}]} (raw SAM2 results JSON)
        with open(metadata_path, encoding='utf-8') as f:
            data = json.load(f)

        all_dets = []
        if isinstance(data, list):
            # Flat list format (ctm_metadata.json or vlm_mask_results.json)
            for entry in data:
                cache_key = entry.get('cache_key', '')
                crop_file = f"{cache_key}.png"
                crop_path = os.path.join(crops_dir, crop_file)
                if os.path.exists(crop_path):
                    all_dets.append({
                        'crop_path': crop_path,
                        'cache_key': cache_key,
                        'label': 1 if entry.get('matched', False) else 0,
                        'confidence': entry.get('rfdetr_conf',
                                  entry.get('confidence', 0.5)),
                        'bbox': entry.get('bbox', [0, 0, 0, 0]),
                        'image_name': entry.get('image', ''),
                        'det_idx': entry.get('det_idx', 0),
                    })
        elif isinstance(data, dict) and 'per_image' in data:
            # Raw SAM2 results JSON with per_image structure
            for img_info in data['per_image']:
                image_name = img_info['image']
                for det_idx, det in enumerate(img_info['detections']):
                    cache_key = f"{image_name.replace('/', '_')}_{det_idx}"
                    crop_file = f"{cache_key}.png"
                    crop_path = os.path.join(crops_dir, crop_file)
                    if os.path.exists(crop_path):
                        all_dets.append({
                            'crop_path': crop_path,
                            'cache_key': cache_key,
                            'label': 1 if det.get('matched', False) else 0,
                            'confidence': det.get('confidence', 0.5),
                            'bbox': det.get('bbox', [0, 0, 0, 0]),
                            'image_name': image_name,
                            'det_idx': det_idx,
                        })
        else:
            raise ValueError(
                f"Unrecognized metadata format in {metadata_path}. "
                f"Expected list or dict with 'per_image' key, got {type(data).__name__}"
            )
        
        n_total = len(all_dets)
        
        if n_total == 0:
            # Debug info
            available = os.listdir(crops_dir) if os.path.isdir(crops_dir) else []
            raise ValueError(
                f"No crops matched! crops_dir={crops_dir}, "
                f"available={len(available)} crops, "
                f"first_available={available[:3] if available else 'none'}"
            )
        
        # Split train/val/test — BY IMAGE, not by detection
        # (prevents data leakage: crops from same image in different splits)
        unique_images = sorted(set(d['image_name'] for d in all_dets))
        n_images = len(unique_images)
        rng = np.random.RandomState(seed)
        img_perm = rng.permutation(n_images)
        n_train_img = int(n_images * train_ratio)
        n_val_img = int(n_images * val_ratio)

        train_images = set(unique_images[i] for i in img_perm[:n_train_img])
        val_images = set(unique_images[i] for i in img_perm[n_train_img:n_train_img + n_val_img])
        test_images = set(unique_images[i] for i in img_perm[n_train_img + n_val_img:])

        image_to_split = {}
        for img in train_images:
            image_to_split[img] = 'train'
        for img in val_images:
            image_to_split[img] = 'val'
        for img in test_images:
            image_to_split[img] = 'test'

        all_indices = [i for i, d in enumerate(all_dets)
                       if image_to_split.get(d['image_name']) == split]
        self.indices = all_indices
        
        self.dets = all_dets
        
        # Balance classes for training (undersample majority)
        if balance and split == 'train':
            labels = [self.dets[i]['label'] for i in self.indices]
            tp_indices = [i for i, l in zip(self.indices, labels) if l == 1]
            fp_indices = [i for i, l in zip(self.indices, labels) if l == 0]
            n_min = min(len(tp_indices), len(fp_indices))
            if len(tp_indices) > n_min:
                tp_indices = rng.choice(tp_indices, n_min, replace=False).tolist()
            if len(fp_indices) > n_min:
                fp_indices = rng.choice(fp_indices, n_min, replace=False).tolist()
            self.indices = tp_indices + fp_indices
            rng.shuffle(self.indices)
        
        # Print stats
        labels = [self.dets[i]['label'] for i in self.indices]
        n_tp = sum(labels)
        n_fp = len(labels) - n_tp
        print(f"[Dataset] {split}: {len(self.indices)} samples "
              f"({n_tp} TP + {n_fp} FP), "
              f"from {n_total} total available crops")
        
        # Transforms
        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(0.2, 0.2, 0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225]),
            ])
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        det = self.dets[self.indices[idx]]
        
        try:
            img = Image.open(det['crop_path']).convert('RGB')
        except Exception:
            img = Image.new('RGB', (self.img_size, self.img_size), (128, 128, 128))
        
        img_tensor = self.transform(img)
        
        return {
            'image': img_tensor,
            'label': torch.tensor(det['label'], dtype=torch.long),
            'confidence': torch.tensor(det['confidence'], dtype=torch.float32),
            'image_name': det['image_name'],
            'cache_key': det['cache_key'],
        }


def get_dataloaders(
    metadata_path: str,
    crops_dir: str,
    img_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    augment: bool = True,
    balance: bool = True,
):
    """Get train/val/test dataloaders."""
    train_ds = OrganoidCTMDataset(
        metadata_path, crops_dir, 'train',
        img_size=img_size, augment=augment, balance=balance
    )
    val_ds = OrganoidCTMDataset(
        metadata_path, crops_dir, 'val',
        img_size=img_size, augment=False, balance=False
    )
    test_ds = OrganoidCTMDataset(
        metadata_path, crops_dir, 'test',
        img_size=img_size, augment=False, balance=False
    )
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    print("=" * 60)
    print("Dataset Test (Phase 2 mode: 100 crops)")
    print("=" * 60)
    
    metadata_path = '/home/z/my-project/organoid-fl/results/phase2_vlm_100_mask/vlm_mask_results.json'
    crops_dir = '/home/z/my-project/organoid-fl/results/phase2_vlm_100_mask/crops'
    
    train_ds = OrganoidCTMDataset(
        metadata_path, crops_dir, 'train',
        img_size=224, augment=False, balance=True
    )
    val_ds = OrganoidCTMDataset(
        metadata_path, crops_dir, 'val',
        img_size=224, augment=False, balance=False
    )
    test_ds = OrganoidCTMDataset(
        metadata_path, crops_dir, 'test',
        img_size=224, augment=False, balance=False
    )
    
    print(f"\nTrain: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    
    # Get one sample
    sample = train_ds[0]
    print(f"\nSample 0:")
    print(f"  image: {sample['image'].shape}")
    print(f"  label: {sample['label'].item()}")
    print(f"  confidence: {sample['confidence'].item():.3f}")
    print(f"  cache_key: {sample['cache_key']}")
    
    # Test dataloader
    from torch.utils.data import DataLoader
    loader = DataLoader(train_ds, batch_size=4, shuffle=False)
    batch = next(iter(loader))
    print(f"\nBatch: images {batch['image'].shape}, labels {batch['label']}")
    
    print("\n✓ Dataset test passed!")
