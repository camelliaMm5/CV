"""
Train traditional CV crowd counter v2 with weighted ensemble.
Compares GBR, RF, weighted ensemble vs CSRNet v4.
"""

import time, joblib
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from traditional_model import TraditionalCrowdCounter, extract_all_features
from dataset import CrowdCountingDataset


def load_data(data_dir='Data'):
    train_ds = CrowdCountingDataset(data_dir, split='train', resize=(640, 480))
    test_ds = CrowdCountingDataset(data_dir, split='test', resize=(640, 480))
    train_imgs, train_counts = [], []
    test_imgs, test_counts = [], []
    for ds, imgs_l, counts_l in [(train_ds, train_imgs, train_counts),
                                   (test_ds, test_imgs, test_counts)]:
        for i in range(len(ds)):
            img = Image.open(ds.img_paths[i]).convert('RGB')
            img = img.resize((640, 480), Image.BILINEAR)
            _, _, count = ds[i]
            imgs_l.append(img)
            counts_l.append(count.item())
    print(f'Train: {len(train_imgs)}, Test: {len(test_imgs)}')
    print(f'Train count: [{min(train_counts):.0f}, {max(train_counts):.0f}]')
    print(f'Test  count: [{min(test_counts):.0f}, {max(test_counts):.0f}]')
    return train_imgs, train_counts, test_imgs, test_counts


def main():
    t_total = time.time()
    print('=' * 65)
    print('Traditional CV Crowd Counting v2 — Weighted Ensemble')
    print('=' * 65)

    train_imgs, train_counts, test_imgs, test_counts = load_data()
    y_test = np.array(test_counts, dtype=np.float32)

    # ---- 1. Train single GBR ----
    print('\n--- 1. GBR (single) ---')
    t0 = time.time()
    gbr = TraditionalCrowdCounter(model_type='gbr')
    gbr.fit(train_imgs, train_counts)
    gbr_eval = gbr.evaluate(test_imgs, test_counts)
    print(f'GBR  Test MAE: {gbr_eval["mae"]:.2f}, RMSE: {gbr_eval["rmse"]:.2f}, '
          f'R2: {gbr_eval["r2"]:.4f} ({time.time()-t0:.0f}s)')

    # ---- 2. Train single RF ----
    print('\n--- 2. RF (single) ---')
    t0 = time.time()
    rf = TraditionalCrowdCounter(model_type='rf')
    rf.fit(train_imgs, train_counts)
    rf_eval = rf.evaluate(test_imgs, test_counts)
    print(f'RF   Test MAE: {rf_eval["mae"]:.2f}, RMSE: {rf_eval["rmse"]:.2f}, '
          f'R2: {rf_eval["r2"]:.4f} ({time.time()-t0:.0f}s)')

    # ---- 3. Weighted ensemble ----
    print('\n--- 3. Weighted Ensemble (GBR+RF) ---')
    t0 = time.time()
    ensemble = TraditionalCrowdCounter(model_type='ensemble')
    ensemble.fit_ensemble(train_imgs, train_counts)
    ens_eval = ensemble.evaluate(test_imgs, test_counts)
    print(f'Ens  Test MAE: {ens_eval["mae"]:.2f}, RMSE: {ens_eval["rmse"]:.2f}, '
          f'R2: {ens_eval["r2"]:.4f} ({time.time()-t0:.0f}s)')

    # ---- 4. Simple average ensemble (for comparison) ----
    avg_preds = (gbr_eval['preds'] + rf_eval['preds']) / 2
    avg_mae = np.abs(avg_preds - y_test).mean()
    avg_rmse = np.sqrt(((avg_preds - y_test) ** 2).mean())
    avg_r2 = np.corrcoef(y_test, avg_preds)[0, 1] ** 2

    # ---- Comparison ----
    print('\n' + '=' * 65)
    print('Comparison: Traditional v2 vs Deep Learning')
    print('=' * 65)
    from numpy.polynomial.polynomial import polyfit
    ens_coef = polyfit(y_test, ens_eval['preds'], 1)

    print(f'{"Method":<35} {"MAE":>8} {"RMSE":>8} {"R2":>8} {"Slope":>8}')
    print(f'{"CSRNet v4 (Deep)":<35} {"11.69":>8} {"19.23":>8} {"0.960":>8} {"0.962":>8}')
    print(f'{"GBR v2 (Traditional)":<35} {gbr_eval["mae"]:>8.2f} {gbr_eval["rmse"]:>8.2f} '
          f'{gbr_eval["r2"]:>8.4f} {"-":>8}')
    print(f'{"RF v2 (Traditional)":<35} {rf_eval["mae"]:>8.2f} {rf_eval["rmse"]:>8.2f} '
          f'{rf_eval["r2"]:>8.4f} {"-":>8}')
    print(f'{"Avg Ensemble":<35} {avg_mae:>8.2f} {avg_rmse:>8.2f} {avg_r2:>8.4f} {"-":>8}')
    print(f'{"Weighted Ensemble":<35} {ens_eval["mae"]:>8.2f} {ens_eval["rmse"]:>8.2f} '
          f'{ens_eval["r2"]:>8.4f} {ens_coef[1]:>8.4f}')

    # Per-group analysis
    print('\n--- Per-Group (Weighted Ensemble) ---')
    for lo, hi, label in [(0, 20, '0-20'), (20, 50, '20-50'),
                           (50, 100, '50-100'), (100, 999, '100+')]:
        mask = (y_test >= lo) & (y_test < hi)
        if mask.sum() > 0:
            mae_g = np.abs(ens_eval['preds'][mask] - y_test[mask]).mean()
            bias_g = ens_eval['preds'][mask].mean() - y_test[mask].mean()
            print(f'  GT {label}: n={mask.sum()}, MAE={mae_g:.2f}, bias={bias_g:.2f}')

    # Scatter comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, preds, title in [
        (axes[0], gbr_eval['preds'], 'GBR v2'),
        (axes[1], rf_eval['preds'], 'RF v2'),
        (axes[2], ens_eval['preds'], 'Weighted Ensemble'),
    ]:
        ax.scatter(y_test, preds, alpha=0.4, s=10, c='steelblue', edgecolors='none')
        ax.plot([0, 600], [0, 600], 'r--', linewidth=1)
        coef = polyfit(y_test, preds, 1)
        xl = np.array([0, 600])
        ax.plot(xl, coef[0] + coef[1] * xl, 'orange', linewidth=1.5,
                label=f'slope={coef[1]:.3f}')
        ax.set_xlim(0, 600); ax.set_ylim(0, 600)
        mae = np.abs(preds - y_test).mean()
        r2 = np.corrcoef(y_test, preds)[0, 1] ** 2
        ax.set_xlabel('Ground Truth'); ax.set_ylabel('Predicted')
        ax.set_title(f'{title}: MAE={mae:.1f}, R2={r2:.3f}')
        ax.legend(); ax.grid(True, alpha=0.3); ax.set_aspect('equal')

    fig.suptitle('Traditional CV Crowd Counting v2', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig('traditional_results_v2.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('\nSaved traditional_results_v2.png')

    # Save best model
    joblib.dump(ensemble, 'traditional_model.pkl')
    print(f'Saved traditional_model.pkl (weighted ensemble)')
    print(f'Total: {time.time()-t_total:.0f}s')


if __name__ == '__main__':
    main()
