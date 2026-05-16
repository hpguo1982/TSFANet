import os
import numpy as np
from PIL import Image
import torch.utils.data as data
import random
import cv2
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

ia.seed(1)
seq = iaa.Sequential([
    iaa.Sharpen((0.0, 1.0)),
    iaa.Affine(scale=(1, 2)),
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5),
    iaa.Crop(percent=(0, 0.1))
], random_order=True)

class Data(data.Dataset):
    def __init__(self, base_dir='/home/weimengru/dataset/CrackMap/data', train=True, dataset='Crack500', crop_szie=None, nclass=2):
        super(Data, self).__init__()
        self.dataset_dir = base_dir
        self.train = train
        self.dataset = dataset
        self.nclass = nclass
        self.images = []
        self.labels = []
        self.names = []

        if crop_szie is None:
            crop_szie = [512, 512]
        self.crop_size = crop_szie

        if self.dataset in ['Crack500', 'CrackMap', 'GAPS384', 'TUT']:
            self.image_dir = os.path.join(self.dataset_dir, self.dataset + '/images')
            self.label_dir = os.path.join(self.dataset_dir, self.dataset + '/labels')
            txt = os.path.join(self.dataset_dir, self.dataset + '/annotations/train.txt' if train else self.dataset + '/annotations/test.txt')
            with open(txt, "r") as f:
                self.filename_list = f.readlines()
            for filename in self.filename_list:
                img_name = filename.strip()
                img_path_jpg = os.path.join(self.image_dir, img_name + '.jpg')
                img_path_bmp = os.path.join(self.image_dir, img_name + '.bmp')
                img_path_png = os.path.join(self.image_dir, img_name + '.png')

                if os.path.exists(img_path_jpg):
                    img_path = img_path_jpg
                elif os.path.exists(img_path_bmp):
                    img_path = img_path_bmp
                elif os.path.exists(img_path_png):
                    img_path = img_path_png
                else:
                    raise FileNotFoundError(f"Image not found: {img_path_jpg} or {img_path_bmp}")

                image = np.array(Image.open(img_path).convert('RGB'))

                if self.dataset == 'Crack500':
                    lbl_name = filename.strip()
                    possible_exts = ['.png', '.jpg', '.bmp']
                    lbl_path = None
                    for ext in possible_exts:
                        candidate = os.path.join(self.label_dir, lbl_name + ext)
                        if os.path.exists(candidate):
                            lbl_path = candidate
                            break
                    if lbl_path is None:
                        raise FileNotFoundError(f"Label not found for {lbl_name} with extensions {possible_exts}")

                else:  # ISIC18
                    lbl_path = os.path.join(self.label_dir, filename.strip() + '_segmentation.png')
                if not os.path.exists(lbl_path):
                    raise FileNotFoundError(f"Label not found: {lbl_path}")

                label = np.array(Image.open(lbl_path))
                if label.dtype != np.uint8:
                    label = label.astype(np.uint8)

                if not self.train:
                    image = cv2.resize(image, tuple(self.crop_size), interpolation=cv2.INTER_NEAREST)
                    label = cv2.resize(label, tuple(self.crop_size), interpolation=cv2.INTER_NEAREST)

                self.images.append(image)
                self.labels.append(label)
                self.names.append(filename.strip())
            assert len(self.images) == len(self.labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        sample = {'image': self.images[index], 'label': self.labels[index]}

        if self.train and random.random() > 0.5:
            segmap = SegmentationMapsOnImage(sample['label'], shape=sample['image'].shape)
            sample['image'], sample['label'] = seq(image=sample['image'], segmentation_maps=segmap)
            sample['label'] = sample['label'].get_arr()

        # Resize both image and label
        sample['image'] = cv2.resize(sample['image'], tuple(self.crop_size), interpolation=cv2.INTER_LINEAR)

        if sample['label'].dtype != np.uint8:
            sample['label'] = sample['label'].astype(np.uint8)
        sample['label'] = cv2.resize(sample['label'], tuple(self.crop_size), interpolation=cv2.INTER_NEAREST)

        # Convert image
        if sample['image'].ndim == 2:
            sample['image'] = np.expand_dims(sample['image'], axis=2)
        sample['image'] = sample['image'].astype(np.float32) / 255.0
        sample['image'] = np.transpose(sample['image'], (2, 0, 1))  # HWC → CHW

        # Convert label
        sample['label'] = sample['label'].astype(np.float32)
        sample['label'] = np.clip(sample['label'], 0, self.nclass - 1)
        sample['label'] = np.expand_dims(sample['label'], axis=0)  # [H, W] → [1, H, W]
        sample['label'] = sample['label'].astype(np.int64)

        return sample

    def __str__(self):
        return f'dataset:{self.dataset} train:{self.train}'