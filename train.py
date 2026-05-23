import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torch.amp import GradScaler, autocast

from dataset import CrowdCountingDataset
from model import CSRNet


def calc_mae_mse(pred_density, gt_density, gt_count):
    pred_count = pred_density.sum(dim=(1, 2, 3))
    mae = torch.abs(pred_count - gt_count).mean().item()
    mse = ((pred_count - gt_count) ** 2).mean().item()
    return mae, mse


def validate(model, dataloader, device):
    model.eval()
    total_mae = 0.0
    total_mse = 0.0
    with torch.no_grad():
        for imgs, densities, counts in dataloader:
            imgs = imgs.to(device)
            densities = densities.to(device)
            counts = counts.to(device)
            pred = model(imgs)
            mae, mse = calc_mae_mse(pred, densities, counts)
            total_mae += mae * imgs.size(0)
            total_mse += mse * imgs.size(0)
    n = len(dataloader.dataset)
    return total_mae / n, (total_mse / n) ** 0.5


def freeze_vgg_stages(model, stages_to_freeze):
    """Freeze specific VGG stages by name: 'conv1','conv2','conv3','conv4'."""
    stage_map = {
        'conv1': model.vgg_conv1,
        'conv2': model.vgg_conv2,
        'conv3': model.vgg_conv3,
        'conv4': model.vgg_conv4,
    }
    for name, stage in stage_map.items():
        requires_grad = name not in stages_to_freeze
        for p in stage.parameters():
            p.requires_grad = requires_grad


def compute_loss(pred, densities, counts, criterion, count_loss_weight):
    """Multi-resolution density loss + count loss.

    Full-res (70%) + half-res (30%) for multi-scale density supervision,
    plus count MAE for global count calibration.
    """
    # Full-resolution density loss
    loss_density_full = criterion(pred, densities)

    # Half-resolution density loss (avg-pooled by 2x)
    pred_half = F.avg_pool2d(pred, kernel_size=2, stride=2)
    gt_half = F.avg_pool2d(densities, kernel_size=2, stride=2)
    loss_density_half = criterion(pred_half, gt_half)

    loss_density = 0.7 * loss_density_full + 0.3 * loss_density_half

    # Count loss
    loss_count = torch.abs(pred.sum(dim=(1, 2, 3)) - counts).mean()

    return loss_density + count_loss_weight * loss_count


def main():
    # ===================== Config =====================
    data_dir = r'Data'
    sigma = 3.0
    batch_size = 4
    epochs = 150
    target_size = (640, 480)
    weight_decay = 1e-4
    count_loss_weight = 1.2
    grad_clip = 1.0
    use_amp = True
    random_crop = True
    use_flip = True
    crop_size = (512, 384)
    use_adaptive_sigma = True
    color_jitter = 0.2
    early_stop_patience = 30  # epochs without improvement before stopping

    # Staged unfreezing: (stage_end_epoch, stages_frozen, lr)
    # Stages to freeze are the VGG blocks NOT being trained yet
    stages = [
        (30,  ['conv1', 'conv2', 'conv3', 'conv4'], 5e-4),   # freeze all VGG
        (60,  ['conv1', 'conv2', 'conv3'],          1e-4),   # unfreeze conv4
        (90,  ['conv1', 'conv2'],                    5e-5),   # unfreeze conv3+4
        (150, [],                                     2e-5),   # unfreeze all VGG
    ]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Device: {device}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Config: batch={batch_size}, epochs={epochs}, AMP={use_amp}, '
          f'crop={random_crop}, flip={use_flip}, grad_clip={grad_clip}')
    print(f'Adaptive sigma={use_adaptive_sigma}, color_jitter={color_jitter}, '
          f'count_loss_weight={count_loss_weight}')
    print(f'Stages: {" -> ".join([f"ep{end}(frz={frz},lr={lr:.0e})" for end,frz,lr in stages])}')
    print(f'Loss: Multi-res SmoothL1 (70% full + 30% half) + {count_loss_weight}*CountMAE')

    # ===================== Dataset =====================
    train_dataset = CrowdCountingDataset(
        data_dir, split='train', sigma=sigma, resize=target_size,
        random_crop=random_crop, crop_size=crop_size, use_flip=use_flip,
        use_adaptive_sigma=use_adaptive_sigma, color_jitter=color_jitter,
    )
    test_dataset = CrowdCountingDataset(
        data_dir, split='test', sigma=sigma, resize=target_size,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)

    print(f'Train: {len(train_dataset)}, Test: {len(test_dataset)}')

    # ===================== Model =====================
    model = CSRNet(pretrained=True).to(device)
    print(f'Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M')

    criterion = nn.SmoothL1Loss()
    writer = SummaryWriter(log_dir='runs/crowd_counting_v3')

    best_mae = float('inf')
    best_path = 'best_model.pth'
    epochs_no_improve = 0

    optimizer = None
    scheduler = None
    scaler = GradScaler('cuda') if use_amp else None

    # ===================== Training Loop =====================
    stage_idx = 0
    for epoch in range(1, epochs + 1):
        # Stage transition
        if stage_idx < len(stages) and epoch == (stages[stage_idx - 1][0] + 1 if stage_idx > 0 else 1):
            pass  # continuing stage

        if stage_idx < len(stages) and epoch == stages[stage_idx][0] + 1:
            stage_idx += 1

        # Check if we need to enter a new stage
        current_stage_end, frozen_stages, stage_lr = stages[stage_idx]
        if epoch == 1 or (stage_idx > 0 and epoch == stages[stage_idx - 1][0] + 1):
            freeze_vgg_stages(model, frozen_stages)
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            optimizer = AdamW(model.parameters(), lr=stage_lr, weight_decay=weight_decay)
            scheduler = CosineAnnealingLR(optimizer, T_max=current_stage_end - epoch + 1)
            print(f'--- Stage {stage_idx+1}: freeze={frozen_stages}, lr={stage_lr:.0e}, '
                  f'trainable={trainable/1e6:.1f}M/{total/1e6:.1f}M ---')

        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for imgs, densities, counts in train_loader:
            imgs = imgs.to(device)
            densities = densities.to(device)
            counts = counts.to(device)

            if use_amp:
                with autocast('cuda'):
                    pred = model(imgs)
                    loss = compute_loss(pred, densities, counts, criterion, count_loss_weight)
            else:
                pred = model(imgs)
                loss = compute_loss(pred, densities, counts, criterion, count_loss_weight)

            optimizer.zero_grad()
            if use_amp:
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            epoch_loss += loss.item() * imgs.size(0)

        scheduler.step()
        avg_loss = epoch_loss / len(train_dataset)

        if epoch % 5 == 0 or epoch == epochs:
            val_mae, val_rmse = validate(model, test_loader, device)
            writer.add_scalar('Val/MAE', val_mae, epoch)
            writer.add_scalar('Val/RMSE', val_rmse, epoch)
            writer.add_scalar('Train/Loss', avg_loss, epoch)

            print(f'Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.4f} | '
                  f'Val MAE: {val_mae:.2f} | Val RMSE: {val_rmse:.2f} | '
                  f'Time: {time.time() - t0:.1f}s | lr: {scheduler.get_last_lr()[0]:.1e}')

            if val_mae < best_mae:
                best_mae = val_mae
                epochs_no_improve = 0
                torch.save(model.state_dict(), best_path)
                print(f'  -> Best model saved (MAE={best_mae:.2f})')
            else:
                epochs_no_improve += 5  # validated every 5 epochs

            if epochs_no_improve >= early_stop_patience:
                print(f'Early stopping: no improvement for {epochs_no_improve} epochs')
                break
        elif epoch % 10 == 0:
            print(f'Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.4f} | '
                  f'Time: {time.time() - t0:.1f}s | lr: {scheduler.get_last_lr()[0]:.1e}')

    writer.close()
    print(f'\nTraining done. Best MAE: {best_mae:.2f}')


if __name__ == '__main__':
    main()
