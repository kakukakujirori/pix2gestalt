from typing import Optional
import glob, os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm


class ImageDataset(Dataset):
    def __init__(
            self,
            datadir: str,
            resolution: int,
            max_train_samples: Optional[int] = None,
            max_val_samples: Optional[int] = None,
            is_train: bool = True,
        ):
        assert os.path.isdir(datadir), f"Not Found: {datadir=}"
        self.datadir = datadir

        image_paths = sorted(glob.glob(os.path.join(datadir, '*.png')) + glob.glob(os.path.join(datadir, '*.jpg')))
        train_image_num = len(image_paths) * 99 // 100

        # 99% train, 1% val
        if is_train:
            self.image_paths = image_paths[:train_image_num]
        else:
            self.image_paths = image_paths[train_image_num:]

        if is_train and max_train_samples is not None:
            self.image_paths = self.image_paths[:max_train_samples]
        if not is_train and max_val_samples is not None:
            self.image_paths = self.image_paths[:max_val_samples]

        self.transforms = transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        ret = {
            'pixel_values': self.transforms(img),
        }
        return ret