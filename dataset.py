import os
import random
import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageEnhance
import torchvision.transforms as T


class CrowdCountingDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, sigma=3.0,
                 downsample=8, resize=None, random_crop=False, crop_size=(512, 384),
                 use_flip=False, use_adaptive_sigma=True, color_jitter=0.2):
        self.root_dir = root_dir
        self.split = split
        self.sigma = sigma
        self.downsample = downsample
        self.resize = resize
        self.random_crop = random_crop and (split == 'train')
        self.use_flip = use_flip and (split == 'train')
        self.use_adaptive_sigma = use_adaptive_sigma
        self.color_jitter = color_jitter if split == 'train' else 0.0
        self.crop_size = crop_size

        img_dir = os.path.join(root_dir, split + '_data', 'images')
        gt_dir = os.path.join(root_dir, split + '_data', 'ground_truth')

        self.img_paths = sorted(
            [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.jpg')],
            key=lambda x: int(x.split('_')[-1].split('.')[0])
        )
        self.gt_paths = sorted(
            [os.path.join(gt_dir, f) for f in os.listdir(gt_dir) if f.endswith('.mat')],
            key=lambda x: int(x.split('_')[-1].split('.')[0])
        )

        assert len(self.img_paths) == len(self.gt_paths), \
            f"Mismatch: {len(self.img_paths)} images vs {len(self.gt_paths)} GTs"

        self.transform = transform or T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.img_paths)

    def _random_crop(self, img, points):
        w, h = img.size
        crop_w, crop_h = self.crop_size

        scale = random.uniform(0.5, 1.0)
        scaled_w = int(w * scale)
        scaled_h = int(h * scale)

        max_x = w - scaled_w
        max_y = h - scaled_h
        x0 = random.randint(0, max(0, max_x)) if max_x >= 0 else 0
        y0 = random.randint(0, max(0, max_y)) if max_y >= 0 else 0

        img = img.crop((x0, y0, x0 + scaled_w, y0 + scaled_h))
        img = img.resize((crop_w, crop_h), Image.BILINEAR)

        if len(points) > 0:
            points = points - np.array([x0, y0])
            points = points * np.array([crop_w / scaled_w, crop_h / scaled_h])
            mask = (points[:, 0] >= 0) & (points[:, 0] < crop_w) & \
                   (points[:, 1] >= 0) & (points[:, 1] < crop_h)
            points = points[mask]

        return img, points

    def _apply_color_jitter(self, img):
        """Random brightness, contrast, saturation jitter."""
        if random.random() < 0.5:
            factor = 1.0 + random.uniform(-self.color_jitter, self.color_jitter)
            img = ImageEnhance.Brightness(img).enhance(factor)
        if random.random() < 0.5:
            factor = 1.0 + random.uniform(-self.color_jitter, self.color_jitter)
            img = ImageEnhance.Contrast(img).enhance(factor)
        if random.random() < 0.5:
            factor = 1.0 + random.uniform(-self.color_jitter, self.color_jitter)
            img = ImageEnhance.Color(img).enhance(factor)
        return img

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        w, h = img.size

        gt_data = scipy.io.loadmat(self.gt_paths[idx])
        points = gt_data['image_info'][0, 0]['location'][0, 0]

        if self.resize is not None:
            target_w, target_h = self.resize
            scale_x = target_w / w
            scale_y = target_h / h
            img = img.resize((target_w, target_h), Image.BILINEAR)
            points = points * np.array([scale_x, scale_y])
            w, h = target_w, target_h

        # Random crop (train only)
        if self.random_crop:
            img, points = self._random_crop(img, points)
            w, h = img.size

        # Ensure dimensions divisible by downsample
        new_h = (h // self.downsample) * self.downsample
        new_w = (w // self.downsample) * self.downsample
        if new_h != h or new_w != w:
            img = img.resize((new_w, new_h), Image.BILINEAR)
            points = points * np.array([new_w / w, new_h / h])

        # Generate density map (adaptive or fixed sigma)
        if self.use_adaptive_sigma and len(points) > 1:
            density = self._gen_density_map_adaptive(points, new_h, new_w)
        else:
            density = self._gen_density_map_fixed(points, new_h, new_w)

        # Horizontal flip (train only)
        if self.use_flip and random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            density = np.fliplr(density).copy()

        # Color jitter (train only, applied on PIL image before ToTensor)
        if self.color_jitter > 0:
            img = self._apply_color_jitter(img)

        if self.transform:
            img = self.transform(img)

        return img, torch.from_numpy(density).float().unsqueeze(0), torch.tensor(points.shape[0]).float()

    def _gen_density_map_fixed(self, points, img_h, img_w):
        """Fixed-sigma Gaussian density map."""
        density_h = img_h // self.downsample
        density_w = img_w // self.downsample
        density = np.zeros((density_h, density_w), dtype=np.float32)
        points_scaled = points / self.downsample

        if len(points_scaled) == 0:
            return density

        for x, y in points_scaled:
            self._add_gaussian(density, x, y, self.sigma, density_h, density_w)

        return density

    def _gen_density_map_adaptive(self, points, img_h, img_w):
        """Geometry-adaptive sigma: per-person sigma based on k-NN distance.

        Dense crowds → small sigma (sharp peaks).
        Sparse crowds → large sigma (smooth blobs).
        """
        density_h = img_h // self.downsample
        density_w = img_w // self.downsample
        density = np.zeros((density_h, density_w), dtype=np.float32)
        points_scaled = points / self.downsample

        if len(points_scaled) == 0:
            return density
        if len(points_scaled) == 1:
            return self._gen_density_map_fixed(points, img_h, img_w)

        # Compute average distance to k nearest neighbors
        from scipy.spatial import KDTree
        k = min(3, len(points_scaled) - 1)
        tree = KDTree(points_scaled)
        distances, _ = tree.query(points_scaled, k=k + 1)
        avg_d = distances[:, 1:].mean(axis=1)  # exclude self (distance=0)

        beta = 0.3
        sigmas = beta * avg_d
        sigmas = np.clip(sigmas, 0.5, 12.0)

        for i, (x, y) in enumerate(points_scaled):
            self._add_gaussian(density, x, y, sigmas[i], density_h, density_w)

        return density

    def _add_gaussian(self, density, x, y, sigma, density_h, density_w):
        """Add a normalized 2D Gaussian centered at (x, y) with given sigma."""
        radius = max(1, int(sigma * 3))
        x_int = int(round(x))
        y_int = int(round(y))

        x_min = max(0, x_int - radius)
        x_max = min(density_w, x_int + radius + 1)
        y_min = max(0, y_int - radius)
        y_max = min(density_h, y_int + radius + 1)

        if x_min >= x_max or y_min >= y_max:
            return

        yy, xx = np.mgrid[y_min:y_max, x_min:x_max]
        g = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        density[y_min:y_max, x_min:x_max] += g
