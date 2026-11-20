import os
import glob
import random
import cv2
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class GOT10kDataset(Dataset):
    def __init__(
        self,
        root_dir,
        subset="train",
        template_size=128,
        search_size=256,
        max_frame_gap=20,
        samples_per_epoch=60000,
        template_context=0.5,
        search_context=2.0,
        translation_jitter=0.6,
        scale_jitter=0.20,
        color_aug=True,
        gray_prob=0.05,
        blur_prob=0.15,
        flip_prob=0.5,
    ):
        self.root_dir = os.path.join(root_dir, subset)
        self.template_size = template_size
        self.search_size = search_size
        self.max_frame_gap = max_frame_gap
        self.samples_per_epoch = samples_per_epoch

        self.template_context = template_context
        self.search_context = search_context
        self.translation_jitter = translation_jitter
        self.scale_jitter = scale_jitter
        self.blur_prob = blur_prob
        self.flip_prob = flip_prob

        if color_aug:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ColorJitter(
                    brightness=0.3,
                    contrast=0.3,
                    saturation=0.3,
                    hue=0.05,
                ),
                transforms.RandomGrayscale(p=gray_prob),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

        self.seq_dirs = sorted(glob.glob(os.path.join(self.root_dir, "GOT-10k_Train_*")))

        if len(self.seq_dirs) == 0:
            self.seq_dirs = sorted([
                p for p in glob.glob(os.path.join(self.root_dir, "*"))
                if os.path.isdir(p)
            ])

        self.sequences = []

        for seq_dir in self.seq_dirs:
            img_paths = sorted(glob.glob(os.path.join(seq_dir, "*.jpg")))
            gt_path = os.path.join(seq_dir, "groundtruth.txt")

            if not os.path.exists(gt_path) or len(img_paths) < 2:
                continue

            bboxes = self._read_bbox(gt_path)

            if len(bboxes) != len(img_paths):
                n = min(len(bboxes), len(img_paths))
                img_paths = img_paths[:n]
                bboxes = bboxes[:n]

            valid = []
            for i, box in enumerate(bboxes):
                x, y, w, h = box
                if np.isfinite(box).all() and w > 1 and h > 1:
                    valid.append(i)

            if len(valid) >= 2:
                self.sequences.append({
                    "seq_dir": seq_dir,
                    "img_paths": img_paths,
                    "bboxes": bboxes.astype(np.float32),
                    "valid": valid,
                })

        if len(self.sequences) == 0:
            raise RuntimeError(f"No valid GOT-10K sequences found in: {self.root_dir}")

    def __len__(self):
        return self.samples_per_epoch

    def _read_bbox(self, file_path):
        bboxes = np.loadtxt(file_path, delimiter=",", dtype=np.float32)
        if bboxes.ndim == 1:
            bboxes = bboxes[None, :]
        return bboxes

    def _read_image(self, path):
        image = cv2.imread(path)
        if image is None:
            raise RuntimeError(f"Failed to read image: {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _is_bbox_inside_crop(self, bbox):
        cx, cy, w, h = bbox

        return (
            np.isfinite(bbox).all()
            and w > 0.0
            and h > 0.0
            and cx - w / 2 >= 0.0
            and cx + w / 2 <= 1.0
            and cy - h / 2 >= 0.0
            and cy + h / 2 <= 1.0
        )

    def _crop_and_resize(
        self,
        image,
        target_bbox,
        output_size,
        context_amount,
        jitter=False,
        require_inside=False,
    ):
        img_h, img_w = image.shape[:2]
        x, y, bw, bh = target_bbox.astype(np.float32)

        obj_cx = x + bw * 0.5
        obj_cy = y + bh * 0.5

        wc = bw + context_amount * (bw + bh)
        hc = bh + context_amount * (bw + bh)
        crop_size = np.sqrt(wc * hc)

        crop_cx = obj_cx
        crop_cy = obj_cy

        if jitter:
            crop_cx += np.random.randn() * bw * self.translation_jitter
            crop_cy += np.random.randn() * bh * self.translation_jitter
            crop_size *= np.exp(np.random.randn() * self.scale_jitter)

        crop_size = max(10.0, crop_size)

        x1 = int(np.floor(crop_cx - crop_size * 0.5))
        y1 = int(np.floor(crop_cy - crop_size * 0.5))
        x2 = int(np.ceil(crop_cx + crop_size * 0.5))
        y2 = int(np.ceil(crop_cy + crop_size * 0.5))

        pad_left = max(0, -x1)
        pad_top = max(0, -y1)
        pad_right = max(0, x2 - img_w)
        pad_bottom = max(0, y2 - img_h)

        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            image = cv2.copyMakeBorder(
                image,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                cv2.BORDER_CONSTANT,
                value=(127, 127, 127),
            )

        x1_pad = x1 + pad_left
        y1_pad = y1 + pad_top
        x2_pad = x2 + pad_left
        y2_pad = y2 + pad_top

        crop = image[y1_pad:y2_pad, x1_pad:x2_pad]

        if crop.size == 0:
            empty = np.full((output_size, output_size, 3), 127, dtype=np.uint8)
            bbox = np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32)
            return empty, bbox, False

        crop = cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_LINEAR)

        actual_crop_w = max(1.0, float(x2 - x1))
        actual_crop_h = max(1.0, float(y2 - y1))

        new_cx = (obj_cx - x1) / actual_crop_w
        new_cy = (obj_cy - y1) / actual_crop_h
        new_w = bw / actual_crop_w
        new_h = bh / actual_crop_h

        bbox = np.array([new_cx, new_cy, new_w, new_h], dtype=np.float32)
        is_valid = self._is_bbox_inside_crop(bbox)

        if require_inside and not is_valid:
            return crop, bbox, False

        return crop, bbox, is_valid

    def _maybe_blur(self, image):
        if random.random() < self.blur_prob:
            k = random.choice([3, 5])
            image = cv2.GaussianBlur(image, (k, k), sigmaX=0)
        return image

    def _maybe_horizontal_flip(self, template, search, bbox):
        if random.random() < self.flip_prob:
            template = np.ascontiguousarray(template[:, ::-1])
            search = np.ascontiguousarray(search[:, ::-1])
            bbox = bbox.copy()
            bbox[0] = 1.0 - bbox[0]

        return template, search, bbox

    def _sample_pair_indices(self, valid_indices):
        template_idx = random.choice(valid_indices)

        min_idx = max(valid_indices[0], template_idx - self.max_frame_gap)
        max_idx = min(valid_indices[-1], template_idx + self.max_frame_gap)

        candidates = [
            i for i in valid_indices
            if min_idx <= i <= max_idx and i != template_idx
        ]

        if len(candidates) == 0:
            candidates = [i for i in valid_indices if i != template_idx]

        search_idx = random.choice(candidates)

        return template_idx, search_idx

    def __getitem__(self, idx):
        for _ in range(50):
            seq = random.choice(self.sequences)
            img_paths = seq["img_paths"]
            bboxes = seq["bboxes"]
            valid = seq["valid"]

            template_idx, search_idx = self._sample_pair_indices(valid)

            try:
                img_template = self._read_image(img_paths[template_idx])
                img_search = self._read_image(img_paths[search_idx])
            except RuntimeError:
                continue

            bbox_template = bboxes[template_idx]
            bbox_search = bboxes[search_idx]

            crop_template, _, template_valid = self._crop_and_resize(
                img_template,
                bbox_template,
                output_size=self.template_size,
                context_amount=self.template_context,
                jitter=False,
                require_inside=True,
            )

            if not template_valid:
                continue

            crop_search, bbox_search_crop, search_valid = self._crop_and_resize(
                img_search,
                bbox_search,
                output_size=self.search_size,
                context_amount=self.search_context,
                jitter=True,
                require_inside=True,
            )

            if not search_valid:
                continue

            crop_template = self._maybe_blur(crop_template)
            crop_search = self._maybe_blur(crop_search)

            crop_template, crop_search, bbox_search_crop = self._maybe_horizontal_flip(
                crop_template,
                crop_search,
                bbox_search_crop,
            )

            tensor_template = self.transform(crop_template)
            tensor_search = self.transform(crop_search)
            tensor_bbox = torch.tensor(bbox_search_crop, dtype=torch.float32)

            return tensor_template, tensor_search, tensor_bbox

        raise RuntimeError("Failed to sample a valid GOT-10K pair after multiple attempts.")


def get_dataloader(
    root_dir,
    batch_size=16,
    num_workers=8,
    template_size=128,
    search_size=256,
    max_frame_gap=20,
    samples_per_epoch=60000,
    translation_jitter=0.6,
    scale_jitter=0.20,
):
    dataset = GOT10kDataset(
        root_dir=root_dir,
        subset="train",
        template_size=template_size,
        search_size=search_size,
        max_frame_gap=max_frame_gap,
        samples_per_epoch=samples_per_epoch,
        translation_jitter=translation_jitter,
        scale_jitter=scale_jitter,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )