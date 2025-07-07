
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
from scipy.ndimage import zoom, rotate


class CTScanDataset(Dataset):
    def __init__(self, data_dir, target_shape=(64, 128, 128)):
        self.samples = []
        self.labels = []
        self.target_shape = target_shape

        for label, folder in enumerate(["CT-0", "CT-23"]):  # 0 = normal, 1 = abnormal
            full_path = os.path.join(data_dir, folder)
            for fname in os.listdir(full_path):
                if fname.endswith(".nii") or fname.endswith(".nii.gz"):
                    self.samples.append(os.path.join(full_path, fname))
                    self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        label = self.labels[idx]
        volume = self._load_nifti(path)
        volume = self._normalize(volume)
        volume = self._resize(volume)
        volume = torch.from_numpy(volume).float().unsqueeze(0)  # [1, D, H, W]
        return volume, torch.tensor(label, dtype=torch.long)

    def _load_nifti(self, path):
        return nib.load(path).get_fdata()

    def _normalize(self, volume):
        volume = np.clip(volume, -1000, 400)
        volume = (volume + 1000) / 1400
        return volume.astype(np.float32)

    def _resize(self, volume):
        # volume shape: [H, W, D] -> [D, H, W]
        volume = rotate(volume, 90, reshape=False)
        current_shape = volume.shape
        factors = [t / s for t, s in zip(self.target_shape, current_shape)]
        return zoom(volume, zoom=factors, order=1)
