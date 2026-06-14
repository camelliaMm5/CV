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
from model import SwinCount


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


def compute_loss(pred, densities, counts, criterion, phase: int = 1):
    """Staged loss for crowd counting.

    Density: 1.0 * full_res + 0.1 * half_res (user feedback #7)
    Count:
      Phase 1: only MAE (stabilize basic counting first)
      Phase 2: MAE + MSE (penalize large errors after counting matures)
    """
    # Density loss — full-res dominates, half-res for coarse guidance
    loss_full = criterion(pred, densities)
    pred_half = F.avg_pool2d(pred, kernel_size=2, stride=2)
    gt_half = F.avg_pool2d(densities, kernel_size=2, stride=2)
    loss_half = criterion(pred_half, gt_half)
    loss_density = 1.0 * loss_full + 0.1 * loss_half

    # Count loss — staged by phase
    pred_counts = pred.sum(dim=(1, 2, 3))
    loss_count_mae = torch.abs(pred_counts - counts).mean()

    if phase == 1:
        return loss_density + 2.0 * loss_count_mae
    else:
        loss_count_mse = ((pred_counts - counts) ** 2).mean()
        return loss_density + 2.0 * loss_count_mae + 0.5 * loss_count_mse


def main():
    # ===================== Config =====================
    data_dir = r'Data'
    sigma = 3.0
    batch_size = 4
    epochs = 120
    target_size = (640, 480)
    weight_decay = 1e-4
    grad_clip = 1.0
    use_amp = True
    random_crop = True
    use_flip = True
    crop_size = (512, 384)
    use_adaptive_sigma = True
    color_jitter = 0.2
    early_stop_patience = 30

    # LoRA config
    lora_r = 16
    lora_alpha = 16
    max_lr = 2e-4

    # Phase boundary
    phase_switch_epoch = 50

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'Device: {device}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Config: batch={batch_size}, epochs={epochs}, AMP={use_amp}')
    print(f'Freeze: shallow(frozen+LoRA), deep(unfrozen+LoRA)')
    print(f'LoRA: r={lora_r}, alpha={lora_alpha} '
          f'(qkv+proj+fc1+fc2 in all blocks)')
    print(f'Head: LightCountingHead (no BN, no SE, no dilated)')
    print(f'Density Loss: 1.0*full + 0.1*half')
    print(f'Count Loss: Phase1(1-{phase_switch_epoch}) MAE only, '
          f'Phase2({phase_switch_epoch+1}-{epochs}) MAE+MSE')
    print(f'Adaptive sigma={use_adaptive_sigma}, color_jitter={color_jitter}')

    # ===================== Dataset =====================
    train_dataset = CrowdCountingDataset(
        data_dir, split='train', sigma=sigma, resize=target_size,
        random_crop=random_crop, crop_size=crop_size, use_flip=use_flip,
        use_adaptive_sigma=use_adaptive_sigma, color_jitter=color_jitter,
    )
    test_dataset = CrowdCountingDataset(
        data_dir, split='test', sigma=sigma, resize=target_size,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=0, pin_memory=True)

    print(f'Train: {len(train_dataset)}, Test: {len(test_dataset)}')

    # ===================== Model =====================
    model = SwinCount(pretrained=True, lora_r=lora_r,
                      lora_alpha=lora_alpha).to(device)
    trainable, total = model.parameter_stats()
    print(f'Params: {total:.1f}M total, {trainable:.1f}M trainable')

    criterion = nn.SmoothL1Loss()
    writer = SummaryWriter(log_dir='runs/swin_count_v4')

    optimizer = AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_mae = float('inf')
    best_path = 'best_model_swin.pth'
    epochs_no_improve = 0
    scaler = GradScaler('cuda') if use_amp else None

    # ===================== Training Loop =====================
    for epoch in range(1, epochs + 1):
        # Phase switching
        phase = 1 if epoch <= phase_switch_epoch else 2
        if epoch == 1:
            print(f'Phase 1: MAE count loss only (epoch 1-{phase_switch_epoch})')
        elif epoch == phase_switch_epoch + 1:
            print(f'Phase 2: MAE + MSE count loss '
                  f'(epoch {phase_switch_epoch+1}-{epochs})')

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
                    loss = compute_loss(pred, densities, counts, criterion,
                                        phase)
            else:
                pred = model(imgs)
                loss = compute_loss(pred, densities, counts, criterion,
                                    phase)

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
                  f'Time: {time.time() - t0:.1f}s | '
                  f'lr: {scheduler.get_last_lr()[0]:.1e} | phase={phase}')

            if val_mae < best_mae:
                best_mae = val_mae
                epochs_no_improve = 0
                torch.save(model.state_dict(), best_path)
                print(f'  -> Best model saved (MAE={best_mae:.2f})')
            else:
                epochs_no_improve += 5

            if epochs_no_improve >= early_stop_patience:
                print(f'Early stopping at epoch {epoch} '
                      f'(no improvement for {epochs_no_improve} epochs)')
                break
        elif epoch % 10 == 0:
            print(f'Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.4f} | '
                  f'Time: {time.time() - t0:.1f}s | '
                  f'lr: {scheduler.get_last_lr()[0]:.1e} | phase={phase}')

    writer.close()
    print(f'\nTraining done. Best MAE: {best_mae:.2f}')

    # ===================== Final Eval =====================
    print('\n=== Final Evaluation ===')
    model.load_state_dict(torch.load(best_path, map_location=device,
                                     weights_only=True))
    model.eval()
    all_preds = []
    all_gts = []
    with torch.no_grad():
        for imgs, densities, counts in test_loader:
            imgs = imgs.to(device)
            densities = densities.to(device)
            counts = counts.to(device)
            pred = model(imgs)
            pred_counts = pred.sum(dim=(1, 2, 3))
            all_preds.extend(pred_counts.cpu().tolist())
            all_gts.extend(counts.cpu().tolist())

    import numpy as np
    all_preds = np.array(all_preds)
    all_gts = np.array(all_gts)
    mae = np.abs(all_preds - all_gts).mean()
    rmse = np.sqrt(((all_preds - all_gts) ** 2).mean())
    r2 = 1 - np.sum((all_gts - all_preds) ** 2) / np.sum(
        (all_gts - all_gts.mean()) ** 2)

    print(f'Final Test MAE: {mae:.2f}')
    print(f'Final Test RMSE: {rmse:.2f}')
    print(f'Final Test R^2: {r2:.4f}')


if __name__ == '__main__':
    main()
