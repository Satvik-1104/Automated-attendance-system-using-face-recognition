import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import logging
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from PIL import Image
import random
import shutil
import argparse

# Set random seeds for reproducibility
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# Constants
DATA_DIR = "../uploads/students_preprocessed"
class_names = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
class_names.sort()
AUGMENTED_DIR = "../uploads/students_augmented"
BATCH_SIZE = 16
TEST_BATCH_SIZE = 2
NUM_EPOCHS = 30
EMBEDDING_SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOG_FILE = "face_recognition_training.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create directories
os.makedirs(AUGMENTED_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)

# 1. iResNet Backbone (unchanged)
__all__ = ['iresnet100']
using_ckpt = False

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class IBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1):
        super(IBasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05)
        self.downsample = downsample
        self.stride = stride

    def forward_impl(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out

    def forward(self, x):
        if self.training and using_ckpt:
            from torch.utils.checkpoint import checkpoint
            return checkpoint(self.forward_impl, x)
        else:
            return self.forward_impl(x)

class IResNet(nn.Module):
    fc_scale = 7 * 7

    def __init__(self, block, layers, dropout=0, num_features=512, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, fp16=False):
        super(IResNet, self).__init__()
        self.extra_gflops = 0.0
        self.fp16 = fp16
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None or a 3-element tuple")
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = nn.PReLU(self.inplanes)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=1e-05)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(512 * block.expansion * self.fc_scale, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, IBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        with torch.cuda.amp.autocast(self.fp16):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.prelu(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.bn2(x)
            x = torch.flatten(x, 1)
            x = self.dropout(x)
        x = self.fc(x.float() if self.fp16 else x)
        x = self.features(x)
        return x

def _iresnet(arch, block, layers, pretrained, progress, **kwargs):
    model = IResNet(block, layers, **kwargs)
    if pretrained:
        raise ValueError("Pretrained weights must be loaded separately")
    return model

def iresnet100(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet100', IBasicBlock, [3, 13, 30, 3], pretrained, progress, **kwargs)

# 2. Enhanced Data Augmentation (unchanged)
def augment_dataset():
    # Enhanced base transformations with increased intensity
    base_light = A.Compose([
        A.HorizontalFlip(p=0.6),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.6),
    ])

    base_medium = A.Compose([
        A.HorizontalFlip(p=0.6),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
        A.Affine(scale=(0.95, 1.05), translate_percent=(-0.05, 0.05), rotate=(-15, 15), p=0.5),
    ])

    base_strong = A.Compose([
        A.HorizontalFlip(p=0.6),
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.8),
        A.Affine(scale=(0.93, 1.07), translate_percent=(-0.07, 0.07), rotate=(-20, 20), p=0.6),
    ])

    # Enhanced augmentation pipelines with increased intensity
    augmentations = [
        A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.18, contrast_limit=0.18, p=0.8),
            A.GaussianBlur(blur_limit=(1, 4), p=0.4),
            A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=15, val_shift_limit=15, p=0.5),
            A.HorizontalFlip(p=0.6),
        ]),
        A.Compose([
            A.Affine(scale=(0.95, 1.05), translate_percent=(-0.05, 0.05), rotate=(-12, 12), p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.18, contrast_limit=0.18, p=0.6),
            A.GaussNoise(var_limit=(10.0, 40.0), p=0.4),
            A.HorizontalFlip(p=0.6),
        ]),
        A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
            A.Rotate(limit=15, p=0.6),
            A.HorizontalFlip(p=0.6),
        ]),
        A.Compose([
            A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=2, shadow_dimension=7, p=0.4),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.6),
            A.ISONoise(intensity=(0.1, 0.2), color_shift=(0.02, 0.03), p=0.4),
            A.HorizontalFlip(p=0.6),
        ]),
        A.Compose([
            A.Affine(scale=(0.93, 1.07), translate_percent=(-0.05, 0.05), rotate=(-10, 10), p=0.6),
            A.RandomBrightnessContrast(brightness_limit=0.18, contrast_limit=0.18, p=0.7),
            A.MotionBlur(blur_limit=3, p=0.3),
            A.HorizontalFlip(p=0.6),
        ]),
        A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
            A.RGBShift(r_shift_limit=12, g_shift_limit=12, b_shift_limit=12, p=0.5),
            A.Affine(scale=(0.94, 1.06), translate_percent=(-0.06, 0.06), rotate=(-18, 18), p=0.6),
            A.HorizontalFlip(p=0.6),
        ]),
        A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.18, contrast_limit=0.18, p=0.7),
            A.GaussianBlur(blur_limit=(1, 4), p=0.4),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=15, p=0.5),
            A.HorizontalFlip(p=0.6),
        ]),
        A.Compose([
            A.Affine(scale=(0.94, 1.06), translate_percent=(-0.05, 0.05), rotate=(-12, 12), p=0.6),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.6),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
            A.HorizontalFlip(p=0.6),
        ]),
        A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.4),
            A.Rotate(limit=15, p=0.6),
            A.HorizontalFlip(p=0.6),
        ]),
        A.Compose([
            A.RandomShadow(shadow_roi=(0, 0, 1, 0.5), num_shadows_lower=1, num_shadows_upper=2, shadow_dimension=5, p=0.4),
            A.RandomBrightnessContrast(brightness_limit=0.18, contrast_limit=0.18, p=0.6),
            A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=12, val_shift_limit=12, p=0.5),
            A.HorizontalFlip(p=0.6),
        ]),
        A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.7),
            A.MotionBlur(blur_limit=3, p=0.3),
            A.Affine(scale=(0.95, 1.05), translate_percent=(-0.05, 0.05), rotate=(-15, 15), p=0.5),
            A.HorizontalFlip(p=0.6),
        ]),
        A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.18, contrast_limit=0.18, p=0.7),
            A.ISONoise(intensity=(0.1, 0.2), color_shift=(0.02, 0.03), p=0.4),
            A.Rotate(limit=15, p=0.6),
            A.HorizontalFlip(p=0.6),
        ]),
        A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.6),
            A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.07, p=0.5),
            A.HorizontalFlip(p=0.6),
        ]),
        A.Compose([
            A.Affine(scale=(0.96, 1.04), translate_percent=(-0.04, 0.04), rotate=(-8, 8), p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.12, contrast_limit=0.12, p=0.5),
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.7, 1.0), p=0.4),
            A.HorizontalFlip(p=0.6),
        ]),
        A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.18, contrast_limit=0.18, p=0.6),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=10, p=0.4),
            A.GaussNoise(var_limit=(5.0, 30.0), p=0.3),
            A.HorizontalFlip(p=0.6),
        ]),
        A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.6),
            A.Affine(scale=(0.96, 1.04), translate_percent=(-0.04, 0.04), rotate=(-10, 10), p=0.5),
            A.RandomShadow(shadow_roi=(0, 0.3, 1, 1), num_shadows_lower=1, num_shadows_upper=1, shadow_dimension=5, p=0.3),
            A.HorizontalFlip(p=0.6),
        ]),
        A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.12, contrast_limit=0.12, p=0.5),
            A.GaussianBlur(blur_limit=(1, 3), p=0.3),
            A.RGBShift(r_shift_limit=8, g_shift_limit=8, b_shift_limit=8, p=0.4),
            A.HorizontalFlip(p=0.6),
        ]),
        A.Compose([
            A.Affine(scale=(0.95, 1.05), translate_percent=(-0.05, 0.05), rotate=(-12, 12), p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.14, contrast_limit=0.14, p=0.5),
            A.MotionBlur(blur_limit=2, p=0.3),
            A.HorizontalFlip(p=0.6),
        ]),
        A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.6),
            A.RGBShift(r_shift_limit=8, g_shift_limit=8, b_shift_limit=8, p=0.4),
            A.CLAHE(clip_limit=1.8, tile_grid_size=(8, 8), p=0.4),
            A.HorizontalFlip(p=0.6),
        ]),
        A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.18, contrast_limit=0.18, p=0.6),
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.7, 1.0), p=0.4),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=10, p=0.4),
            A.HorizontalFlip(p=0.6),
        ]),
        A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.6),
            A.RandomShadow(shadow_roi=(0, 0, 1, 0.3), num_shadows_lower=1, num_shadows_upper=2, shadow_dimension=5, p=0.3),
            A.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.12, hue=0.05, p=0.4),
            A.HorizontalFlip(p=0.6),
        ]),
    ]

    to_tensor = A.Compose([ToTensorV2()])

    # Enhanced mixup function with stronger alpha parameter
    def apply_mixup(img1, img2, alpha=0.2):  # Increased from 0.1 to 0.2 for stronger mixing
        lam = np.random.beta(alpha, alpha)
        mixed = lam * img1 + (1 - lam) * img2
        return mixed

    person_list = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    for person in person_list:
        person_dir = os.path.join(DATA_DIR, person)
        augmented_person_dir = os.path.join(AUGMENTED_DIR, person)

        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(augmented_person_dir, split), exist_ok=True)

        images = [f for f in os.listdir(person_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        if len(images) < 3:
            print(f"Skipping {person} with insufficient images")
            logger.info(f"Skipping {person} with insufficient images")
            continue

        random.shuffle(images)
        train_idx = int(len(images) * 0.7)
        val_idx = int(len(images) * 0.85)

        splits = {
            'train': images[:train_idx],
            'val': images[train_idx:val_idx],
            'test': images[val_idx:]
        }

        for split_name, split_images in splits.items():
            split_dir = os.path.join(augmented_person_dir, split_name)
            for img_file in split_images:
                src = os.path.join(person_dir, img_file)
                dst = os.path.join(split_dir, f"orig_{img_file}")
                shutil.copy(src, dst)

        train_dir = os.path.join(augmented_person_dir, 'train')
        train_images = []

        for img_file in splits['train']:
            img_path = os.path.join(person_dir, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_np = np.array(img)
            train_images.append(img_np)

            for aug_idx, transform in enumerate(augmentations):
                try:
                    augmented = transform(image=img_np)
                    aug_img = augmented['image']
                    aug_img = aug_img.astype(np.uint8)
                    aug_img_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
                    aug_path = os.path.join(train_dir, f"aug{aug_idx}_{img_file}")
                    cv2.imwrite(aug_path, aug_img_bgr)
                except Exception as e:
                    print(f"Augmentation error for {img_file} with aug{aug_idx}: {e}")
                    logger.info(f"Augmentation error for {img_file} with aug{aug_idx}: {e}")

        # Enhanced mixup with more mixups per image if available
        if len(train_images) >= 2:
            # Increased from min(len(train_images) * 5, 50) to min(len(train_images) * 7, 70)
            num_mixups = min(len(train_images) * 7, 70)  
            for i in range(num_mixups):
                try:
                    idx1, idx2 = random.sample(range(len(train_images)), 2)
                    img1, img2 = train_images[idx1], train_images[idx2]
                    if img1.shape != img2.shape:
                        h, w = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
                        img1 = cv2.resize(img1, (w, h))
                        img2 = cv2.resize(img2, (w, h))
                    
                    mixed = apply_mixup(img1, img2, alpha=0.2)  # Increased alpha
                    mixed = mixed.astype(np.uint8)
                    
                    # Apply stronger base augmentation to mixup result
                    base_aug = random.choice([base_medium, base_strong])  # Prefer stronger augmentations
                    if random.random() < 0.7:  # 70% chance to use stronger augmentation
                        base_aug = base_strong
                    
                    augmented = base_aug(image=mixed)
                    aug_img = augmented['image']
                    aug_img = aug_img.astype(np.uint8)
                    aug_img_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
                    aug_path = os.path.join(train_dir, f"mixup{i}_{splits['train'][idx1]}")
                    cv2.imwrite(aug_path, aug_img_bgr)
                except Exception as e:
                    print(f"Mixup augmentation error: {e}")
                    logger.info(f"Mixup augmentation error: {e}")

        # Enhanced validation set augmentation
        val_dir = os.path.join(augmented_person_dir, 'val')
        val_augmentations = [base_light, base_medium, base_strong]  # Added base_strong for validation
        for img_file in splits['val']:
            img_path = os.path.join(person_dir, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_np = np.array(img)
            for aug_idx, transform in enumerate(val_augmentations):
                try:
                    augmented = transform(image=img_np)
                    aug_img = augmented['image']
                    aug_img = aug_img.astype(np.uint8)
                    aug_img_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
                    aug_path = os.path.join(val_dir, f"aug{aug_idx}_{img_file}")
                    cv2.imwrite(aug_path, aug_img_bgr)
                except Exception as e:
                    print(f"Validation augmentation error: {e}")
                    logger.info(f"Validation augmentation error: {e}")

# 3. Dataset Class (unchanged)
class FaceDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.classes = []
        self.class_to_idx = {}
        self.samples = []

        for person_idx, person_name in enumerate(sorted(os.listdir(root_dir))):
            person_dir = os.path.join(root_dir, person_name)
            if os.path.isdir(person_dir):
                split_dir = os.path.join(person_dir, split)
                if os.path.exists(split_dir):
                    self.classes.append(person_name)
                    self.class_to_idx[person_name] = person_idx
                    for img_name in os.listdir(split_dir):
                        if img_name.endswith(('.jpg', '.jpeg', '.png')):
                            img_path = os.path.join(split_dir, img_name)
                            self.samples.append((img_path, person_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# 4. ArcFace Loss (unchanged)
class ArcFaceLoss(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super(ArcFaceLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(embeddings, weight)
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
        one_hot = F.one_hot(labels, num_classes=self.out_features).float().to(DEVICE)
        target_cos = torch.cos(theta + self.m)
        output = self.s * (one_hot * target_cos + (1.0 - one_hot) * cosine)
        return F.cross_entropy(output, labels)

# 5. Modified Model with Gradual Unfreezing Support
class FaceClassifier(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        # Initially freeze all backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_layer(self, layer_name):
        """Unfreeze a specific layer of the backbone."""
        layer = getattr(self.backbone, layer_name)
        for param in layer.parameters():
            param.requires_grad = True
        print(f"Unfroze {layer_name}")
        logger.info(f"Unfroze {layer_name}")

    def forward(self, x):
        return self.backbone(x)

# 6. Training and Evaluation Functions
def train_epoch(model, criterion, optimizer, train_loader, val_loader, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [TRAIN]")
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        embeddings = model(inputs)
        loss = criterion(embeddings, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        with torch.no_grad():
            logits = F.linear(F.normalize(embeddings), F.normalize(criterion.weight))
            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
        pbar.set_postfix(loss=loss.item(), acc=correct / total)

    train_loss = running_loss / len(train_loader.dataset)
    train_acc = correct / total

    val_metrics = evaluate(model, criterion, val_loader, device, epoch, phase="VAL")
    return train_loss, train_acc, val_metrics

def evaluate(model, criterion, loader, device, epoch, phase="VAL"):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [{phase}]")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            embeddings = model(inputs)
            loss = criterion(embeddings, labels)
            running_loss += loss.item() * inputs.size(0)

            logits = F.linear(F.normalize(embeddings), F.normalize(criterion.weight))
            _, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            pbar.set_postfix(loss=loss.item(), acc=correct / total)

    metrics = {
        'loss': running_loss / len(loader.dataset),
        'acc': correct / total,
        'precision': precision_score(all_labels, all_preds, average='weighted', zero_division=0),
        'recall': recall_score(all_labels, all_preds, average='weighted', zero_division=0),
        'f1': f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    }
    return metrics

def test_evaluate(model, criterion, loader, device, class_names, phase="TEST"):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"[{phase}] Evaluation")
        for batch_idx, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(device), labels.to(device)
            embeddings = model(inputs)
            loss = criterion(embeddings, labels)
            running_loss += loss.item() * inputs.size(0)

            logits = F.linear(F.normalize(embeddings), F.normalize(criterion.weight))
            logits = logits * criterion.s
            probabilities = F.softmax(logits, dim=1)
            confidence, predicted = logits.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            batch_start_idx = batch_idx * loader.batch_size
            batch_samples = loader.dataset.samples[batch_start_idx:batch_start_idx + inputs.size(0)]

            for i, (sample_path, true_label) in enumerate(batch_samples):
                sample_name = os.path.basename(sample_path)
                pred_idx = predicted[i].item()
                true_idx = labels[i].item()
                conf_score = probabilities[i, pred_idx].item()
                prob_vector = probabilities[i].cpu().numpy()
                prob_str = ', '.join([f'{p:.4f}' for p in prob_vector])

                pred_name = class_names[pred_idx]
                true_name = class_names[true_idx]

                print(f"Sample: {sample_name} | True: {true_name} | Predicted: {pred_name} | Confidence: {conf_score:.4f}")
                print(f" → Probabilities: [{prob_str}]")
                logger.info(f"Sample: {sample_name} | True: {true_name} | Predicted: {pred_name} | Confidence: {conf_score:.4f}")

            pbar.set_postfix(loss=loss.item(), acc=correct / total)

    metrics = {
        'loss': running_loss / len(loader.dataset),
        'acc': correct / total,
        'precision': precision_score(all_labels, all_preds, average='weighted', zero_division=0),
        'recall': recall_score(all_labels, all_preds, average='weighted', zero_division=0),
        'f1': f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    }
    return metrics

# 7. Training Pipeline with Gradual Unfreezing
def train():
    train_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])

    train_set = FaceDataset(AUGMENTED_DIR, 'train', train_transform)
    val_set = FaceDataset(AUGMENTED_DIR, 'val', val_transform)
    test_set = FaceDataset(AUGMENTED_DIR, 'test', val_transform)

    print(f"Number of training samples: {len(train_set)}")
    logger.info(f"Number of training samples: {len(train_set)}")
    print(f"Number of validation samples: {len(val_set)}")
    logger.info(f"Number of validation samples: {len(val_set)}")
    print(f"Number of test samples: {len(test_set)}")
    logger.info(f"Number of test samples: {len(test_set)}")

    backbone = iresnet100(pretrained=False, num_features=EMBEDDING_SIZE)
    try:
        backbone.load_state_dict(torch.load("models/arcfaceresnet100torch.pth", map_location=DEVICE))
        print("Loaded pretrained backbone weights")
        logger.info("Loaded pretrained backbone weights")
    except:
        print("Could not load pretrained backbone")
        logger.info("Could not load pretrained backbone")

    model = FaceClassifier(backbone).to(DEVICE)
    criterion = ArcFaceLoss(EMBEDDING_SIZE, len(train_set.classes), s=30.0, m=0.50).to(DEVICE)

    # Define optimizer with all parameters (some will have requires_grad=False initially)
    optimizer = optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': 1e-4},
        {'params': criterion.parameters(), 'lr': 1e-3}
    ], weight_decay=5e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_set, TEST_BATCH_SIZE, shuffle=False, num_workers=4, drop_last=True)
    test_loader = DataLoader(test_set, TEST_BATCH_SIZE, shuffle=False, num_workers=4, drop_last=True)

    # Gradual unfreezing schedule
    unfreeze_schedule = {
        5: 'layer4',   # Unfreeze layer4 after 5 epochs
        10: 'layer3',  # Unfreeze layer3 after 10 epochs
        15: 'layer2',  # Unfreeze layer2 after 15 epochs
        20: 'layer1',  # Unfreeze layer1 after 20 epochs
    }

    best_f1 = 0
    for epoch in range(NUM_EPOCHS):
        # Check if we need to unfreeze any layers this epoch
        if epoch in unfreeze_schedule:
            model.unfreeze_layer(unfreeze_schedule[epoch])
            # Rebuild optimizer to include newly unfrozen parameters
            optimizer = optim.AdamW([
                {'params': model.backbone.parameters(), 'lr': 1e-4},
                {'params': criterion.parameters(), 'lr': 1e-3}
            ], weight_decay=5e-4)

        train_loss, train_acc, val_metrics = train_epoch(model, criterion, optimizer, train_loader, val_loader, DEVICE, epoch)
        scheduler.step()

        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS} Results:")
        logger.info(f"\nEpoch {epoch + 1}/{NUM_EPOCHS} Results:")
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        logger.info(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['acc']:.4f}")
        logger.info(f"Val Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['acc']:.4f}")
        print(f"Val Precision: {val_metrics['precision']:.4f} | Val Recall: {val_metrics['recall']:.4f} | Val F1: {val_metrics['f1']:.4f}")
        logger.info(f"Val Precision: {val_metrics['precision']:.4f} | Val Recall: {val_metrics['recall']:.4f} | Val F1: {val_metrics['f1']:.4f}")

        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            torch.save({'model_state_dict': model.state_dict(), 'criterion_state_dict': criterion.state_dict()}, "models/fasmodel.pt")
            print("Saved best model!")
            logger.info("Saved best model!")

    test_metrics = test_evaluate(model, criterion, test_loader, DEVICE, train_set.classes)
    print("\nFinal Test Results:")
    logger.info("\nFinal Test Results:")
    print(f"Loss: {test_metrics['loss']:.4f} | Acc: {test_metrics['acc']:.4f}")
    logger.info(f"Loss: {test_metrics['loss']:.4f} | Acc: {test_metrics['acc']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f} | Recall: {test_metrics['recall']:.4f} | F1: {test_metrics['f1']:.4f}")
    logger.info(f"Precision: {test_metrics['precision']:.4f} | Recall: {test_metrics['recall']:.4f} | F1: {test_metrics['f1']:.4f}")

# 8. Testing Function (unchanged)
def test():
    class_names = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    class_names.sort()
    num_classes = len(class_names)

    backbone = iresnet100(pretrained=False, num_features=EMBEDDING_SIZE)
    model = FaceClassifier(backbone).to(DEVICE)
    criterion = ArcFaceLoss(EMBEDDING_SIZE, num_classes, s=30.0, m=0.50).to(DEVICE)

    checkpoint = torch.load("models/fasmodel.pt", map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    criterion.load_state_dict(checkpoint['criterion_state_dict'])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])
    test_set = FaceDataset(AUGMENTED_DIR, 'test', transform)
    test_loader = DataLoader(test_set, BATCH_SIZE, shuffle=False)

    metrics = test_evaluate(model, criterion, test_loader, DEVICE, class_names)

    print("\nTest Results:")
    logger.info("\nTest Results:")
    print(f"Loss: {metrics['loss']:.4f} | Acc: {metrics['acc']:.4f}")
    logger.info(f"Loss: {metrics['loss']:.4f} | Acc: {metrics['acc']:.4f}")
    print(f"Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f}")

# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--test', action='store_true', help='Test the model')
    args = parser.parse_args()

    if args.train:
        print("Augmenting dataset...")
        logger.info("Augmenting dataset...")
        augment_dataset()
        print("Starting training...")
        logger.info("Starting training...")
        train()

    if args.test:
        print("Starting testing...")
        logger.info("Starting testing...")
        test()