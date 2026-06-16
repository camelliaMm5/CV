"""
Train traditional CV crowd counter v3 — reduced features, stronger regularization, Stacking.
Changes:
  - GBR: added early_stopping, reduced depth/estimators, larger min_samples_leaf
  - Features: removed ORB, SIFT response_mean, CLAHE_HOG
  - Ensemble: StackingRegressor (GBR+RF+Ridge base, Ridge meta) with 5-fold CV
  - Added Ridge single-model baseline
  - Added 5-fold CV evaluation
"""

import time, joblib
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

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


def cross_val_eval(model_type, train_imgs, train_counts, test_imgs, test_counts, n_folds=5):
    """5-fold CV: train on folds, eval on test set. Returns (mean_mae, std_mae, mean_rmse, mean_r2, model_fit_on_all)."""
    y_test = np.array(test_counts, dtype=np.float32)
    y_train_raw = np.array(train_counts, dtype=np.float32)
    n = len(train_imgs)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_mae = []
    fold_rmse = []
    fold_r2 = []

    for fold, (tr_idx, _) in enumerate(kf.split(range(n))):
        imgs_tr = [train_imgs[i] for i in tr_idx]
        counts_tr = y_train_raw[tr_idx]
        model = TraditionalCrowdCounter(model_type=model_type)
        model.fit(imgs_tr, counts_tr, feature_selection=True)
        preds = np.array([model.predict(img) for img in test_imgs])
        fold_mae.append(np.abs(preds - y_test).mean())
        fold_rmse.append(np.sqrt(((preds - y_test) ** 2).mean()))
        fold_r2.append(r2_score(y_test, preds))

    # Train final model on all training data
    final_model = TraditionalCrowdCounter(model_type=model_type)
    final_model.fit(train_imgs, train_counts, feature_selection=True)

    print(f'{model_type.upper():>6} CV{len(fold_mae)}f MAE: {np.mean(fold_mae):.2f} '
          f'±{np.std(fold_mae):.2f}  RMSE: {np.mean(fold_rmse):.2f}  '
          f'R2: {np.mean(fold_r2):.4f}')
    return np.mean(fold_mae), np.std(fold_mae), np.mean(fold_rmse), np.mean(fold_r2), final_model


def main():
    t_total = time.time()
    print('=' * 65)
    print('Traditional CV Crowd Counting v3 — Stacking + CV')
    print('=' * 65)

    train_imgs, train_counts, test_imgs, test_counts = load_data()
    y_test = np.array(test_counts, dtype=np.float32)

    results = {}

    # ---- 1. Ridge baseline ----
    print('\n--- 1. Ridge (linear baseline) ---')
    t0 = time.time()
    ridge_cv = cross_val_eval('ridge', train_imgs, train_counts, test_imgs, test_counts)
    ridge_eval = ridge_cv[4].evaluate(test_imgs, test_counts)
    results['Ridge'] = ridge_eval
    print(f'Ridge Test MAE: {ridge_eval["mae"]:.2f}, RMSE: {ridge_eval["rmse"]:.2f}, '
          f'R2: {ridge_eval["r2"]:.4f} ({time.time()-t0:.0f}s)')

    # ---- 2. GBR ----
    print('\n--- 2. GBR (with early stopping) ---')
    t0 = time.time()
    gbr_cv = cross_val_eval('gbr', train_imgs, train_counts, test_imgs, test_counts)
    gbr_eval = gbr_cv[4].evaluate(test_imgs, test_counts)
    results['GBR'] = gbr_eval
    print(f'GBR  Test MAE: {gbr_eval["mae"]:.2f}, RMSE: {gbr_eval["rmse"]:.2f}, '
          f'R2: {gbr_eval["r2"]:.4f} ({time.time()-t0:.0f}s)')

    # ---- 3. RF ----
    print('\n--- 3. RF ---')
    t0 = time.time()
    rf_cv = cross_val_eval('rf', train_imgs, train_counts, test_imgs, test_counts)
    rf_eval = rf_cv[4].evaluate(test_imgs, test_counts)
    results['RF'] = rf_eval
    print(f'RF   Test MAE: {rf_eval["mae"]:.2f}, RMSE: {rf_eval["rmse"]:.2f}, '
          f'R2: {rf_eval["r2"]:.4f} ({time.time()-t0:.0f}s)')

    # ---- 4. Stacking (GBR + RF + Ridge -> Ridge meta, 5-fold CV) ----
    print('\n--- 4. Stacking (GBR+RF+Ridge -> Ridge, cv=5) ---')
    t0 = time.time()
    stk = TraditionalCrowdCounter(model_type='stacking')
    stk.fit_stacking(train_imgs, train_counts, feature_selection=True)
    stk_eval = stk.evaluate(test_imgs, test_counts)
    results['Stacking'] = stk_eval
    print(f'Stk  Test MAE: {stk_eval["mae"]:.2f}, RMSE: {stk_eval["rmse"]:.2f}, '
          f'R2: {stk_eval["r2"]:.4f} ({time.time()-t0:.0f}s)')

    # ---- Comparison Table ----
    print('\n' + '=' * 65)
    print(f'{"Method":<25} {"MAE":>8} {"RMSE":>8} {"R2":>8}')
    print('-' * 49)
    for name in ['Ridge', 'GBR', 'RF', 'Stacking']:
        e = results[name]
        print(f'{name:<25} {e["mae"]:>8.2f} {e["rmse"]:>8.2f} {e["r2"]:>8.4f}')

    # ---- Per-Group Analysis (Stacking) ----
    print('\n--- Per-Group (Stacking) ---')
    for lo, hi, label in [(0, 20, '0-20'), (20, 50, '20-50'),
                           (50, 100, '50-100'), (100, 999, '100+')]:
        mask = (y_test >= lo) & (y_test < hi)
        if mask.sum() > 0:
            mae_g = np.abs(stk_eval['preds'][mask] - y_test[mask]).mean()
            bias_g = stk_eval['preds'][mask].mean() - y_test[mask].mean()
            print(f'  GT {label:>5}: n={mask.sum():>3}, MAE={mae_g:.2f}, bias={bias_g:+.2f}')

    # ---- Scatter Plots ----
    from numpy.polynomial.polynomial import polyfit

    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    axes = axes.flatten()
    for ax, (name, e) in zip(axes, results.items()):
        ax.scatter(y_test, e['preds'], alpha=0.4, s=10, c='steelblue', edgecolors='none')
        ax.plot([0, 600], [0, 600], 'r--', linewidth=1)
        coef = polyfit(y_test, e['preds'], 1)
        xl = np.array([0, 600])
        ax.plot(xl, coef[0] + coef[1] * xl, 'orange', linewidth=1.5,
                label=f'slope={coef[1]:.3f}')
        ax.set_xlim(0, 600); ax.set_ylim(0, 600)
        ax.set_xlabel('Ground Truth'); ax.set_ylabel('Predicted')
        ax.set_title(f'{name}: MAE={e["mae"]:.1f}, RMSE={e["rmse"]:.1f}, R2={e["r2"]:.3f}')
        ax.legend(); ax.grid(True, alpha=0.3); ax.set_aspect('equal')

    fig.suptitle('Traditional CV Crowd Counting v3 (Stacking)', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig('traditional_results_v3.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('\nSaved traditional_results_v3.png')

    # Save best model (Stacking)
    joblib.dump(stk, 'traditional_model_v3.pkl')
    print(f'Saved traditional_model_v3.pkl')
    print(f'Total: {time.time()-t_total:.0f}s')


if __name__ == '__main__':
    main()
