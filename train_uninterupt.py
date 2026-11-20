import copy
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from utils.spectreTrack import SpectreTrack
from utils.dataset import GOT10kDataset

def get_freest_gpu():
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        return "cpu"
    
    # Get the ID of the GPU with the maximum free memory
    best_gpu = max(
        range(torch.cuda.device_count()), 
        key=lambda i: torch.cuda.mem_get_info(i)[0]
    )
    print(f"INFO:GPU {best_gpu} SELECTED")
    return f"cuda:{best_gpu}"

# Replace your original line with:
device = "cuda:2"


def apply_adversarial_mask(
    search_clean,
    importance,
    patch_size=16,
    mask_ratio=0.20,
    mask_value="mean",
):
    """
    Mask the most salient search-image patches based on teacher search-token importance.

    Args:
        search_clean: [B, C, H, W], normalized search images.
        importance: [B, N], teacher search-token L2 norms.
        patch_size: ViT patch size.
        mask_ratio: fraction of search patches to mask.
        mask_value: "mean" uses 0 in normalized space, "black" uses true black after normalization.

    Returns:
        search_masked: [B, C, H, W]
    """
    B, C, H, W = search_clean.shape

    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    num_search_patches = num_patches_h * num_patches_w

    search_importance = importance[:, -num_search_patches:]

    num_mask = max(1, int(mask_ratio * num_search_patches))
    _, topk_indices = torch.topk(search_importance, num_mask, dim=-1)

    mask = torch.ones(
        (B, 1, num_patches_h, num_patches_w),
        device=search_clean.device,
        dtype=search_clean.dtype,
    )

    y_coords = topk_indices // num_patches_w
    x_coords = topk_indices % num_patches_w

    batch_indices = torch.arange(B, device=search_clean.device).view(B, 1)
    mask[batch_indices, 0, y_coords, x_coords] = 0.0

    mask_pixel = F.interpolate(
        mask,
        size=(H, W),
        mode="nearest",
    )

    if mask_value == "black":
        fill = torch.tensor(
            [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            device=search_clean.device,
            dtype=search_clean.dtype,
        ).view(1, 3, 1, 1)
    else:
        fill = torch.zeros(
            (1, C, 1, 1),
            device=search_clean.device,
            dtype=search_clean.dtype,
        )

    search_masked = search_clean * mask_pixel + fill * (1.0 - mask_pixel)

    return search_masked


@torch.no_grad()
def update_teacher_ema(student, teacher, decay=0.996):
    for teacher_param, student_param in zip(teacher.parameters(), student.parameters()):
        teacher_param.data.mul_(decay).add_(student_param.data, alpha=1.0 - decay)


spectre_net = SpectreTrack(
    model_name="vit_base_patch16_224.mae",
    pretrained=True,
).to(device)

teacher_net = copy.deepcopy(spectre_net).to(device)

for param in teacher_net.parameters():
    param.requires_grad = False

teacher_net.eval()

backbone_params = []
head_params = []

for name, param in spectre_net.named_parameters():
    if not param.requires_grad:
        continue

    if "backbone" in name:
        backbone_params.append(param)
    else:
        head_params.append(param)

optimizer = AdamW(
    [
        {"params": backbone_params, "lr": 1e-5},
        {"params": head_params, "lr": 3e-4},
    ],
    weight_decay=1e-4,
)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=40,
    gamma=0.1,
)

lambda_rec = 10.0

import torch
import importlib
import utils.dataset

importlib.reload(utils.dataset)

from utils.dataset import get_dataloader

data_path = '/root/victor/GOT-10K_FULL'

dataloader = get_dataloader(
    root_dir=data_path,
    batch_size=16,
    num_workers=8,
    template_size=128,
    search_size=256,
    max_frame_gap=20,
    samples_per_epoch=60000,
    translation_jitter=0.6,
    scale_jitter=0.20,
)

for template, search, bbox in dataloader:
    print(f"Template shape: {template.shape}")  # [16, 3, 128, 128]
    print(f"Search shape: {search.shape}")      # [16, 3, 256, 256]
    print(f"BBox shape: {bbox.shape}")          # [16, 4]
    print(f"BBox min/max: {bbox.min().item():.4f}, {bbox.max().item():.4f}")

    assert template.shape[-2:] == (128, 128)
    assert search.shape[-2:] == (256, 256)
    assert bbox.shape[-1] == 4
    assert torch.isfinite(template).all()
    assert torch.isfinite(search).all()
    assert torch.isfinite(bbox).all()
    assert bbox.min() >= 0.0 and bbox.max() <= 1.0

    break

import torch
import torch.nn.functional as F
from torchvision.ops import generalized_box_iou


def cxcywh_to_xyxy(bbox):
    cx, cy, w, h = bbox.unbind(-1)

    w = w.clamp(min=1e-6)
    h = h.clamp(min=1e-6)

    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    return torch.stack([x1, y1, x2, y2], dim=-1).clamp(0.0, 1.0)


def calculate_giou_loss(pred_boxes, gt_boxes):
    pred_boxes_xyxy = cxcywh_to_xyxy(pred_boxes)
    gt_boxes_xyxy = cxcywh_to_xyxy(gt_boxes)

    giou_matrix = generalized_box_iou(pred_boxes_xyxy, gt_boxes_xyxy)
    giou = torch.diag(giou_matrix)

    return (1.0 - giou).mean()


@torch.no_grad()
def update_teacher_ema(student, teacher, momentum=0.999):
    for student_param, teacher_param in zip(student.parameters(), teacher.parameters()):
        teacher_param.data.mul_(momentum).add_(student_param.data, alpha=1.0 - momentum)


def train_epoch(
    dataloader,
    spectre_net,
    teacher_net,
    optimizer,
    device,
    epoch,
    lambda_rec=10.0,
    mask_ratio=0.20,
):
    spectre_net.train()
    teacher_net.eval()

    total_loss_epoch = 0.0

    for batch_idx, (template, search_clean, bbox_gt) in enumerate(dataloader):
        template = template.to(device, non_blocking=True)
        search_clean = search_clean.to(device, non_blocking=True)
        bbox_gt = bbox_gt.to(device, non_blocking=True)

        with torch.no_grad():
            features_teacher = teacher_net.extract_clean_features(template, search_clean)
            importance = torch.norm(features_teacher, dim=-1)

        search_masked = apply_adversarial_mask(
            search_clean,
            importance,
            patch_size=16,
            mask_ratio=mask_ratio,
        )

        optimizer.zero_grad(set_to_none=True)

        bbox_pred, features_student, confidence = spectre_net(
            template,
            search_masked,
            history_buffer=None,
        )

        loss_giou = calculate_giou_loss(bbox_pred, bbox_gt)
        loss_l1 = F.smooth_l1_loss(bbox_pred, bbox_gt)

        # Optional: all current samples are valid positives.
        confidence_target = torch.ones_like(confidence)
        loss_score = F.binary_cross_entropy(confidence, confidence_target)

        norm_student = F.normalize(features_student, p=2, dim=-1)
        norm_teacher = F.normalize(features_teacher.detach(), p=2, dim=-1)
        loss_rec = F.mse_loss(norm_student, norm_teacher)

        loss_loc = loss_giou + 2.0 * loss_l1 + 0.1 * loss_score
        total_loss = loss_loc + lambda_rec * loss_rec

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(spectre_net.parameters(), max_norm=1.0)
        optimizer.step()

        update_teacher_ema(spectre_net, teacher_net, momentum=0.999)

        total_loss_epoch += total_loss.item()

        if batch_idx % 50 == 0:
            print(
                f"Epoch [{epoch}] Batch [{batch_idx}] | "
                f"GIoU: {loss_giou.item():.4f} | "
                f"L1: {loss_l1.item():.4f} | "
                f"Score: {loss_score.item():.4f} | "
                f"Rec: {loss_rec.item():.4f} | "
                f"Total: {total_loss.item():.4f}"
            )

    return total_loss_epoch / len(dataloader)


num_epochs = 100

print("Starting SpectreTrack GOT-10K fine-tuning...")
print("Using clean bbox labels, adversarial search masking, and EMA teacher.")

for epoch in range(num_epochs):
    avg_loss = train_epoch(
        dataloader=dataloader,
        spectre_net=spectre_net,
        teacher_net=teacher_net,
        optimizer=optimizer,
        device=device,
        epoch=epoch,
        lambda_rec=1.0,
        mask_ratio=0.20,
    )

    scheduler.step()

    print(f"--- Epoch {epoch} Completed | Average Loss: {avg_loss:.4f} ---")

    if epoch % 10 == 0 or epoch == num_epochs - 1:
        torch.save(
            spectre_net.state_dict(),
            f"spectre_got10k_mae_epoch_{epoch}.pth",
        )