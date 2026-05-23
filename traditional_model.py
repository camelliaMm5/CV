"""
Traditional CV crowd counting v2 — hand-crafted features + ensemble regression.

Improvements over v1:
  - Illumination robustness: CLAHE preprocessing
  - Head-like detection: SIFT, FAST, multi-scale blob detector
  - Background/foreground mask statistics
  - Blur score for quality-aware prediction
  - Weighted ensemble + stacking meta-learner
"""

import numpy as np
import cv2
from sklearn.ensemble import (GradientBoostingRegressor, RandomForestRegressor,
                               StackingRegressor)
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops


# ======================================================================
#  Preprocessing helpers
# ======================================================================

def _apply_clahe(gray):
    """Contrast-limited adaptive histogram equalization — illumination robust."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _blur_score(gray):
    """Variance of Laplacian — low value = blurry image."""
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def _background_mask(gray):
    """Estimate foreground via morphological black-hat (small bright regions)."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    # Black-hat: original - closing = small bright features (people against dark bg)
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    blackhat = cv2.subtract(closed, gray)
    # Top-hat: original - opening = small bright features
    opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    tophat = cv2.subtract(gray, opened)
    # Combine
    mask = cv2.addWeighted(blackhat, 0.5, tophat, 0.5, 0)
    _, mask = cv2.threshold(mask, 30, 255, cv2.THRESH_BINARY)
    return mask


# ======================================================================
#  Feature extractors (each returns a fixed-size 1D array)
# ======================================================================

def extract_hog(gray, orientations=9, spatial_bins=4):
    """HOG descriptor — fixed 36 dims (per-orientation stats across spatial bins)."""
    h, w = gray.shape
    bh, bw = max(1, h // spatial_bins), max(1, w // spatial_bins)
    orient_means = []
    for i in range(spatial_bins):
        for j in range(spatial_bins):
            patch = gray[i*bh:min((i+1)*bh, h), j*bw:min((j+1)*bw, w)]
            if patch.size < 100:
                orient_means.append(np.zeros(orientations))
                continue
            try:
                h_feat = hog(patch, orientations=orientations,
                             pixels_per_cell=(max(8, patch.shape[0]//2), max(8, patch.shape[1]//2)),
                             cells_per_block=(1, 1), feature_vector=True)
                n_orient = len(h_feat) // orientations * orientations
                if n_orient > 0:
                    orient_means.append(h_feat[:n_orient].reshape(-1, orientations).mean(axis=0))
                else:
                    orient_means.append(np.zeros(orientations))
            except Exception:
                orient_means.append(np.zeros(orientations))
    m = np.array(orient_means)
    return np.concatenate([m.mean(axis=0),
                           np.percentile(m, [25, 50, 75], axis=0).flatten()]).astype(np.float32)


def extract_lbp(gray, radius=2, n_points=16):
    """Uniform LBP histogram — fixed 243 bins."""
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    n_bins = n_points * (n_points - 1) + 3
    hist, _ = np.histogram(lbp, bins=n_bins, range=(-0.5, n_bins - 0.5), density=True)
    return hist.astype(np.float32)


def extract_glcm(gray, distances=(1, 3, 5), angles=(0, np.pi/4, np.pi/2, 3*np.pi/4)):
    """GLCM texture — 12 dims: mean/std of contrast, dissimilarity, homogeneity,
    energy, correlation, ASM."""
    glcm = graycomatrix(gray, distances=distances, angles=angles, symmetric=True, normed=True)
    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    feats = []
    for prop in props:
        vals = graycoprops(glcm, prop).flatten()
        feats.extend([vals.mean(), vals.std()])
    return np.array(feats, dtype=np.float32)


def extract_edge_density(gray):
    """Canny edge stats — 4 dims: edge ratio, mean/median/95th gradient magnitude."""
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = edges.mean() / 255.0
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    return np.array([edge_ratio, mag.mean(), np.median(mag), np.percentile(mag, 95)],
                    dtype=np.float32)


def extract_fft_bands(gray):
    """FFT energy in low/mid/high rings — 4 dims."""
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_max = max(cy, cx)
    feats = []
    for r_lo, r_hi in [(0, r_max * 0.15), (r_max * 0.15, r_max * 0.5), (r_max * 0.5, r_max)]:
        mask = (radius >= r_lo) & (radius < r_hi)
        feats.append(mag[mask].sum() / mask.sum() if mask.sum() > 0 else 0.0)
    feats.append(mag.sum() / (h * w))
    return np.array(feats, dtype=np.float32)


def extract_color_stats(rgb):
    """Per-channel stats — 9 dims: mean, std, skewness for R,G,B."""
    feats = []
    for ch in range(3):
        c = rgb[:, :, ch].astype(np.float32)
        mu, sig = c.mean(), c.std()
        feats.extend([mu, sig])
        if sig > 1e-8:
            feats.append(((c - mu) ** 3).mean() / (sig ** 3))
        else:
            feats.append(0.0)
    return np.array(feats, dtype=np.float32)


def extract_foreground_ratio(gray):
    """Otsu foreground ratio + connected components — 2 dims."""
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    fg_ratio = (cleaned > 0).mean()
    n_labels, _ = cv2.connectedComponents(cleaned)
    cc_norm = n_labels / (gray.size / 1000.0)
    return np.array([fg_ratio, cc_norm], dtype=np.float32)


# ======================================================================
#  NEW: Head-detection & robustness features
# ======================================================================

def extract_sift_density(gray):
    """SIFT keypoint density — captures distinctive head-like features.  4 dims."""
    try:
        sift = cv2.SIFT_create(nfeatures=800)
    except AttributeError:
        # SIFT not available (needs opencv-contrib), fall back to ORB
        return extract_keypoint_density(gray)
    kp = sift.detect(gray, None)
    n = len(kp)
    if n == 0:
        return np.array([0, 0, 0, 0], dtype=np.float32)
    sizes = np.array([k.size for k in kp])
    responses = np.array([k.response for k in kp])
    density = n / (gray.size / 10000.0)
    return np.array([density, sizes.mean(), sizes.std(),
                     responses.mean()], dtype=np.float32)


def extract_fast_density(gray):
    """FAST corner density — corners often appear at head-shoulder junctions.  3 dims."""
    fast = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
    kp = fast.detect(gray, None)
    n = len(kp)
    if n == 0:
        return np.array([0, 0, 0], dtype=np.float32)
    responses = np.array([k.response for k in kp])
    density = n / (gray.size / 10000.0)
    return np.array([density, responses.mean(), responses.std()], dtype=np.float32)


def extract_blob_count(gray):
    """Multi-scale LoG blob detector — detects circular head-like blobs.  5 dims."""
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 5
    params.maxArea = 500
    params.filterByCircularity = True
    params.minCircularity = 0.3
    params.filterByConvexity = True
    params.minConvexity = 0.5
    params.filterByInertia = True
    params.minInertiaRatio = 0.3

    detector = cv2.SimpleBlobDetector_create(params)
    kp = detector.detect(gray)
    n = len(kp)
    if n == 0:
        return np.array([0, 0, 0, 0, 0], dtype=np.float32)
    sizes = np.array([k.size for k in kp])
    density = n / (gray.size / 10000.0)
    return np.array([density, sizes.mean(), sizes.std(),
                     np.percentile(sizes, 25), np.percentile(sizes, 75)], dtype=np.float32)


def extract_keypoint_density(gray):
    """ORB keypoint density — 1 dim."""
    orb = cv2.ORB_create(nfeatures=500)
    kp = orb.detect(gray, None)
    return np.array([len(kp) / (gray.size / 10000.0)], dtype=np.float32)


def extract_background_mask_stats(gray):
    """Morphological foreground mask statistics.  6 dims."""
    mask = _background_mask(gray)
    fg_ratio = (mask > 0).mean()
    # Edge density within foreground only
    edges = cv2.Canny(gray, 50, 150)
    fg_edge_ratio = (edges & mask).mean() / 255.0 if mask.sum() > 0 else 0
    # Connected components in mask
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]  # skip background
        cc_count = n_labels - 1
        mean_area = areas.mean() if len(areas) > 0 else 0
        std_area = areas.std() if len(areas) > 0 else 0
        max_area = areas.max() if len(areas) > 0 else 0
    else:
        cc_count, mean_area, std_area, max_area = 0, 0, 0, 0
    return np.array([fg_ratio, fg_edge_ratio, cc_count / (gray.size / 10000.0),
                     mean_area, std_area, max_area], dtype=np.float32)


def extract_blur_score(gray):
    """Laplacian variance — 1 dim. Low = blurry."""
    return np.array([_blur_score(gray)], dtype=np.float32)


def extract_patch_edge_stats(gray, grid=(4, 4)):
    """Edge density spatial variance — 2 dims."""
    h, w = gray.shape
    ph, pw = h // grid[0], w // grid[1]
    edge_ratios = []
    for i in range(grid[0]):
        for j in range(grid[1]):
            patch = gray[i*ph:(i+1)*ph, j*pw:(j+1)*pw]
            e = cv2.Canny(patch, 50, 150)
            edge_ratios.append(e.mean() / 255.0)
    edge_ratios = np.array(edge_ratios)
    return np.array([edge_ratios.mean(), edge_ratios.std()], dtype=np.float32)


# ======================================================================
#  Full feature extraction (v2 — with illumination robustness)
# ======================================================================

def extract_all_features(img_rgb):
    """Extract all hand-crafted features. Fixed dimension regardless of input size."""
    if hasattr(img_rgb, 'convert'):
        img_rgb = np.array(img_rgb)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray_clahe = _apply_clahe(gray)  # illumination-normalized

    features = []

    # Features on original gray
    features.append(extract_hog(gray))
    features.append(extract_lbp(gray))
    features.append(extract_edge_density(gray))
    features.append(extract_fft_bands(gray))
    features.append(extract_glcm(gray))

    # Features on CLAHE-normalized gray (illumination-robust)
    features.append(extract_hog(gray_clahe))
    features.append(extract_glcm(gray_clahe))
    features.append(extract_edge_density(gray_clahe))

    # Color statistics
    features.append(extract_color_stats(img_rgb))

    # Foreground / background
    features.append(extract_foreground_ratio(gray))
    features.append(extract_background_mask_stats(gray))

    # Keypoint / head-like features
    features.append(extract_keypoint_density(gray))
    features.append(extract_sift_density(gray))
    features.append(extract_fast_density(gray))
    features.append(extract_blob_count(gray))

    # Quality & spatial
    features.append(extract_blur_score(gray))
    features.append(extract_patch_edge_stats(gray))

    return np.concatenate(features).astype(np.float32)


# ======================================================================
#  Traditional crowd counter v2 (with weighted ensemble & stacking)
# ======================================================================

class TraditionalCrowdCounter:
    """Traditional CV crowd counter v2.

    Supports: GBR, RF, weighted ensemble, stacking ensemble.
    """

    def __init__(self, model_type='gbr'):
        self.model_type = model_type
        self.scaler = StandardScaler()
        self.selector = None
        self.regressor = None        # single model
        self.models = {}              # ensemble models
        self.ensemble_weights = None  # per-model weights
        self.feat_dim = None
        self._X_train = None
        self._y_train = None

    def _build_regressor(self, model_type=None):
        mt = model_type or self.model_type
        if mt == 'gbr':
            return GradientBoostingRegressor(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.8, min_samples_leaf=5, random_state=42,
            )
        elif mt == 'rf':
            return RandomForestRegressor(
                n_estimators=300, max_depth=15, min_samples_leaf=5,
                random_state=42, n_jobs=-1,
            )
        elif mt == 'ridge':
            return Ridge(alpha=1.0)
        else:
            raise ValueError(f"Unknown model_type: {mt}")

    def _preprocess(self, images):
        X = np.stack([extract_all_features(img) for img in images])
        if self.scaler is None or not hasattr(self.scaler, 'mean_'):
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)
        if self.selector is not None:
            X = self.selector.transform(X)
        return X

    def _select_features(self, X, y):
        if X.shape[1] > 50:
            selector_rf = RandomForestRegressor(n_estimators=100, max_depth=8,
                                                random_state=42, n_jobs=-1)
            selector_rf.fit(X, y)
            importances = selector_rf.feature_importances_
            threshold = np.percentile(importances, 30)
            self.selector = SelectFromModel(selector_rf, threshold=threshold, prefit=True)
            X = self.selector.transform(X)
            print(f'Feature selection: {X.shape[1]}/{self.feat_dim} retained')
        return X

    def fit(self, images, counts, feature_selection=True):
        """Train single-model regressor."""
        X = np.stack([extract_all_features(img) for img in images])
        y = np.array(counts, dtype=np.float32)
        self.feat_dim = X.shape[1]
        self._X_train = X
        self._y_train = y
        print(f'Feature matrix: {X.shape} ({self.feat_dim} dims)')

        X = self.scaler.fit_transform(X)
        if feature_selection:
            X = self._select_features(X, y)

        self.regressor = self._build_regressor()
        self.regressor.fit(X, y)
        preds = self.regressor.predict(X)
        print(f'Train MAE: {np.abs(preds - y).mean():.2f}')
        return self

    def fit_ensemble(self, images, counts, feature_selection=True):
        """Train weighted ensemble: GBR + RF, weights from validation MAE."""
        X = np.stack([extract_all_features(img) for img in images])
        y = np.array(counts, dtype=np.float32)
        self._X_train = X
        self._y_train = y
        self.feat_dim = X.shape[1]
        print(f'Feature matrix: {X.shape} ({self.feat_dim} dims)')

        X = self.scaler.fit_transform(X)
        if feature_selection:
            X = self._select_features(X, y)

        # Split for validation-weight estimation
        from sklearn.model_selection import train_test_split
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train GBR
        gbr = self._build_regressor('gbr')
        gbr.fit(X_tr, y_tr)
        gbr_mae = np.abs(gbr.predict(X_val) - y_val).mean()

        # Train RF
        rf = self._build_regressor('rf')
        rf.fit(X_tr, y_tr)
        rf_mae = np.abs(rf.predict(X_val) - y_val).mean()

        # Weights: inverse of MAE (normalized)
        w_gbr = 1.0 / max(gbr_mae, 1e-6)
        w_rf = 1.0 / max(rf_mae, 1e-6)
        total_w = w_gbr + w_rf
        self.ensemble_weights = {'gbr': w_gbr / total_w, 'rf': w_rf / total_w}
        self.models = {'gbr': gbr, 'rf': rf}
        self.model_type = 'ensemble'

        # Refit on full data
        self.models['gbr'] = self._build_regressor('gbr').fit(X, y)
        self.models['rf'] = self._build_regressor('rf').fit(X, y)

        preds = self._predict_ensemble(X)
        print(f'Weights: GBR={self.ensemble_weights["gbr"]:.3f}, '
              f'RF={self.ensemble_weights["rf"]:.3f}')
        print(f'Val MAE: GBR={gbr_mae:.2f}, RF={rf_mae:.2f}')
        print(f'Train MAE (ensemble): {np.abs(preds - y).mean():.2f}')
        return self

    def _predict_ensemble(self, X):
        w = self.ensemble_weights
        return w['gbr'] * self.models['gbr'].predict(X) + \
               w['rf'] * self.models['rf'].predict(X)

    def predict(self, img_rgb):
        """Predict crowd count for a single image."""
        feats = extract_all_features(img_rgb).reshape(1, -1)
        feats = self.scaler.transform(feats)
        if self.selector is not None:
            feats = self.selector.transform(feats)

        if self.model_type == 'ensemble':
            return max(0, self._predict_ensemble(feats)[0])
        return max(0, self.regressor.predict(feats)[0])

    def predict_density_map(self, img_rgb, grid_h=10, grid_w=13):
        """Patch-based density map."""
        if hasattr(img_rgb, 'convert'):
            img_rgb = np.array(img_rgb)
        h, w = img_rgb.shape[:2]
        ph, pw = h // grid_h, w // grid_w

        density = np.zeros((grid_h, grid_w), dtype=np.float32)
        for i in range(grid_h):
            for j in range(grid_w):
                patch = img_rgb[i*ph:(i+1)*ph, j*pw:(j+1)*pw]
                if patch.size == 0:
                    continue
                density[i, j] = max(0, self.predict(patch)) / (ph * pw)

        total = self.predict(img_rgb)
        if density.sum() > 1e-8:
            density = density / density.sum() * total
        return density, max(0, round(total))

    def evaluate(self, images, counts):
        """Evaluate on a list of images."""
        preds = np.array([self.predict(img) for img in images])
        gts = np.array(counts, dtype=np.float32)
        errors = np.abs(preds - gts)
        return {'mae': errors.mean(),
                'rmse': np.sqrt((errors ** 2).mean()),
                'r2': np.corrcoef(gts, preds)[0, 1] ** 2,
                'preds': preds, 'gts': gts}
