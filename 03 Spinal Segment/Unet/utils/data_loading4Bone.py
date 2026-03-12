import logging
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

import cv2



class BasicDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)

        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]

        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale, is_mask):
        # w, h = pil_img.size
        # scale = 1
        # newW, newH = int(scale * w), int(scale * h)
        # assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        # pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img_ndarray = pil_img
        if img_ndarray.ndim == 2 and not is_mask:
            img_ndarray = img_ndarray[np.newaxis, ...]

        elif not is_mask:
            img_ndarray = img_ndarray.transpose((2, 0, 1))

        if is_mask:
            # print('-----------------------')
            # print('---')
            # print(np.unique(img_ndarray))
            # if len(np.unique(img_ndarray))>2:
            #     # print('0000000000000000000000')
            #print(np.unique(img_ndarray))
            img_ndarray[img_ndarray>1]=0
            img_ndarray = img_ndarray*1.0

        else:
            img_ndarray = img_ndarray/1500
            img_ndarray = img_ndarray

        return img_ndarray

    @classmethod
    def load(cls, filename):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            return np.load(filename)
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            a = Image.open(filename)

            return a

    def __getitem__(self, idx):
        name = self.ids[idx]
        # print(name)
        # mask_file = list(self.masks_dir.glob(name.replace('_', '')+'.npy'))
        mask_file = list(self.masks_dir.glob(name+'.npy'))

        img_file = list(self.images_dir.glob(name + '.npy'))
        # print("mask_file",mask_file)
        # print("img_file",img_file)
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        mask = self.load(mask_file[0])
        img = self.load(img_file[0])
        random_num = np.random.rand()
        if random_num < 0.25:
            angle = np.random.randint(0, 360)
            img = np.rot90(img, k=angle // 90)
            mask = np.rot90(mask, k=angle // 90)

        elif random_num < 0.5:
            img = np.fliplr(img)
            mask = np.fliplr(mask)

        else:
            img = np.flipud(img)
            mask = np.flipud(mask)

        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)



        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }
    # def __getitem__(self, idx):
    #     name = self.ids[idx]
    #     mask_file = self.masks_dir+'/'+name+'.npy'
    #
    #     img_file = list(self.images_dir.glob(name + '.npy'))  self.images_dir+'/'+name+'.npy'
    #     assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
    #     assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
    #     mask = self.load(mask_file[0])
    #     img = self.load(img_file[0])
    #     img = self.preprocess(img, self.scale, is_mask=False)
    #     mask = self.preprocess(mask, self.scale, is_mask=True)
    #     return {
    #         'image': torch.as_tensor(img.copy()).float().contiguous(),
    #         'mask': torch.as_tensor(mask.copy()).long().contiguous()
    #     }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, masks_dir, scale=1):
        super().__init__(images_dir, masks_dir, scale, mask_suffix='_mask')



