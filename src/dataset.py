from typing import Optional
import glob, os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm


class Pix2GestaltDataset(Dataset):
    def __init__(
            self,
            datadir: str,
            resolution: int,
            center_crop: bool,
            random_flip: bool,
            max_train_samples: Optional[int] = None,
            max_val_samples: Optional[int] = None,
            is_train: bool = True,
        ):
        assert os.path.isdir(datadir), f"Not Found: {datadir=}"
        self.datadir = datadir

        composed_images = glob.glob(os.path.join(datadir, 'occlusion/*.png'))
        image_ids = sorted([os.path.basename(x).replace("_occlusion.png", "") for x in composed_images])
        train_image_num = len(image_ids) * 99 // 100

        # 99% train, 1% val
        if is_train:
            self.image_ids = image_ids[:train_image_num]
        else:
            self.image_ids = image_ids[train_image_num:]
            center_crop = False
            random_flip = False

        if is_train and max_train_samples is not None:
            self.image_ids = self.image_ids[:max_train_samples]
        if not is_train and max_val_samples is not None:
            self.image_ids = self.image_ids[:max_val_samples]

        # sanity check
        for image_id in tqdm(self.image_ids, desc="[Pix2GestaltDataset] Checking files"):
            assert os.path.isfile(os.path.join(datadir, 'visible_object_mask', f'{image_id}_visible_mask.png'))
            assert os.path.isfile(os.path.join(datadir, 'whole', f'{image_id}_whole.png'))
            assert os.path.isfile(os.path.join(datadir, 'whole_mask', f'{image_id}_whole_mask.png'))

        self.transforms = transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution),
            transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        img_id = self.image_ids[idx]
        occlusion = Image.open(os.path.join(self.datadir, 'occlusion', f'{img_id}_occlusion.png')).convert('RGB')
        vis_obj_mask = Image.open(os.path.join(self.datadir, 'visible_object_mask', f'{img_id}_visible_mask.png')).convert('RGB')
        whole = Image.open(os.path.join(self.datadir, 'whole', f'{img_id}_whole.png')).convert('RGB')
        whole_mask = Image.open(os.path.join(self.datadir, 'whole_mask', f'{img_id}_whole_mask.png')).convert('RGB')

        ret = {
            'occlusion': self.transforms(occlusion),
            'visible_object_mask': self.transforms(vis_obj_mask),
            'whole': self.transforms(whole),
            'whole_mask': self.transforms(whole_mask),
            'image_id': int(img_id),
        }
        return ret