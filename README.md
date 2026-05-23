# 人群计数 (Crowd Counting) —— 深度学习 vs 传统计算机视觉

本项目实现并对比了**深度学习**和**传统计算机视觉**两种方法在人群计数任务上的表现。使用相同的训练/测试数据集（400 张训练图，316 张测试图），分别构建了 CSRNet 深度学习模型和基于手工特征+集成回归的传统 CV 模型，并通过 Gradio 交互界面进行实时推理对比。

## 项目结构

```
图片计数大作业/
├── Data/
│   ├── train_data/
│   │   ├── images/          # 训练图片 (.jpg)
│   │   └── ground_truth/    # 人头位置标注 (.mat，每个文件包含坐标矩阵)
│   └── test_data/
│       ├── images/
│       └── ground_truth/
├── model.py                 # CSRNet 网络结构定义
├── dataset.py               # 数据集加载与密度图生成
├── train.py                 # 深度学习训练脚本
├── predict.py               # 深度学习命令行推理
├── traditional_model.py     # 传统 CV 特征提取 + 回归模型
├── traditional_train.py     # 传统 CV 训练评估脚本
├── app.py                   # Gradio Web 交互界面
├── best_model.pth           # 训练好的 CSRNet 权重
├── traditional_model.pkl    # 训练好的传统 CV 模型
├── deep_vs_traditional.png  # 深度 vs 传统方法散点对比图
├── scatter_v4_final.png     # CSRNet 最终预测散点图
├── training_curves_v2.png   # CSRNet 训练曲线
└── traditional_results_v2.png # 传统 CV 结果图
```

## 方法一：深度学习 —— CSRNet

### 网络结构 (model.py)

CSRNet（Congested Scene Recognition Network）由两部分组成：

- **前端 (VGG-16, 10 层卷积)**：使用在 ImageNet 上预训练的 VGG-16 的前 10 个卷积层作为特征提取器。将其拆分为 4 个命名阶段（`conv1` ~ `conv4`），输出为原图 1/8 分辨率的特征图（512 通道）。因为去掉了后续的池化层，输出保持较大的空间分辨率，有利于生成高质量的密度图。

- **后端 (扩张卷积)**：6 层扩张卷积（dilation=2/4），不使用 BN。扩张卷积在不增加参数量的前提下扩大感受野，使每个像素能感知更大范围的空间上下文。最后通过 1×1 卷积输出单通道密度图。

```
输入 (3, H, W)
  → VGG conv1 (64ch, stride=2)
  → VGG conv2 (128ch, stride=4)
  → VGG conv3 (256ch, stride=8)
  → VGG conv4 (512ch, stride=8)
  → 扩张卷积 × 6 (512→512→256→128→64ch)
  → 1×1 输出 (1, H/8, W/8)
```

### 密度图生成 (dataset.py)

密度图是训练目标（ground truth），将离散的人头坐标点转化为连续密度分布：

- **固定 σ 高斯核**：每个标注点放置一个标准差 σ=3.0 的高斯核，归一化使每个点贡献积分为 1，密度图求和即为总人数。

- **自适应 σ（几何自适应高斯核）**：针对人群密度不一的问题，使用 k-NN（k=3）计算每个点与其邻居的平均距离，σ = β × 平均距离（β=0.3），并限制在 [0.5, 12.0] 范围内。
  - 密集人群 → 小 σ，峰值尖锐
  - 稀疏人群 → 大 σ，平滑分布

- 密度图在原始图像 1/8 分辨率下生成（与网络输出匹配）。

### 数据增强 (dataset.py)

训练时使用多种数据增强策略：

| 策略 | 方法 | 参数 |
|------|------|------|
| 随机裁剪 | 先随机缩放 (0.5~1.0×)，再随机裁剪 | 最终尺寸 512×384 |
| 水平翻转 | 50% 概率翻转图像和密度图 | p=0.5 |
| 颜色抖动 | 随机调整亮度、对比度、饱和度 | 幅度 ±0.2 |
| 尺寸对齐 | 将尺寸调整为 8 的倍数 | - |

### 损失函数 (train.py)

**多分辨率密度损失 + 计数损失**：

$$L = L_{density} + \lambda \cdot L_{count}$$

- **多分辨率密度损失**：70% 全分辨率 SmoothL1 + 30% 半分辨率（2×2 平均池化后）SmoothL1，提供多尺度密度监督
- **计数损失**：预测密度图总和与真实人数的 MAE，权重 λ=1.2
- 使用 SmoothL1Loss 而非 MSE，对异常值更鲁棒

### 训练策略 (train.py)

| 配置项 | 值 |
|--------|-----|
| 优化器 | AdamW (weight_decay=1e-4) |
| 学习率调度 | CosineAnnealingLR（每阶段重置） |
| Batch size | 4 |
| 总 Epochs | 150（实际早停于 135 epoch） |
| 混合精度 | AMP (GradScaler + autocast) |
| 梯度裁剪 | max_norm=1.0 |
| 早停 | 30 epochs 无改善即停止 |
| 输入尺寸 | 640×480 |

**分阶段解冻 (Staged Unfreezing)**：

| 阶段 | Epoch | 冻结层 | 学习率 | 可训练参数 |
|------|-------|--------|--------|-------------|
| 1 | 1-30 | conv1~4（全部 VGG） | 5e-4 | 8.6M/16.3M |
| 2 | 31-60 | conv1~3 | 1e-4 | 14.5M/16.3M |
| 3 | 61-90 | conv1~2 | 5e-5 | 16.0M/16.3M |
| 4 | 91-135 | 无（全部解冻） | 2e-5 | 16.3M/16.3M |

先冻住预训练的 VGG 只训练后端，再逐步解冻更深的 VGG 层，避免初期破坏预训练权重。

### 评估指标

- **MAE (Mean Absolute Error)**：$\frac{1}{N}\sum |pred - gt|$
- **RMSE (Root Mean Squared Error)**：$\sqrt{\frac{1}{N}\sum (pred - gt)^2}$

### 最终结果 (CSRNet v4)

| 指标 | 值 |
|------|-----|
| Test MAE | **11.69** |
| Test RMSE | **19.23** |
| R² | **0.960** |
| 斜率 (拟合线) | **0.962** |

---

## 方法二：传统计算机视觉 —— 手工特征 + 集成回归

### 整体流程 (traditional_model.py)

```
输入图像
  → CLAHE 光照归一化
  → 多类型手工特征提取（384 维）
  → StandardScaler 标准化
  → 集成回归器（GBR + RF 加权融合）
  → 预测人数
```

### 光照鲁棒预处理

- **CLAHE (对比度受限自适应直方图均衡化)**：`clipLimit=2.0, tileGridSize=8×8`，消除不均匀光照对特征的干扰。

### 特征提取（共 384 维）

#### 1. HOG 特征 (Histogram of Oriented Gradients)

方向梯度直方图，将图像分为 4×4 空间网格，每格计算 9 方向 HOG，取均值对每个方向统计 mean/P25/P50/P75。**共 36 维**。

分别在原始灰度图和 CLAHE 增强图上提取，各 36 维。

#### 2. LBP 纹理特征 (Local Binary Pattern)

Uniform LBP（半径=2，16 邻域点），统计直方图。**共 243 维**。

旋转不变的局部纹理描述子，对光照变化不敏感，能刻画人群中衣物纹理密度的变化。

#### 3. GLCM 纹理统计 (Gray-Level Co-occurrence Matrix)

在 3 种距离 (1,3,5) × 4 种角度 (0°, 45°, 90°, 135°) 计算灰度共生矩阵，提取 6 个属性（contrast, dissimilarity, homogeneity, energy, correlation, ASM）的均值和标准差。**共 12 维**。

分别在原始灰度图和 CLAHE 增强图上提取，各 12 维。

#### 4. 边缘密度 (Canny Edge)

Canny 边缘检测（阈值 50/150），统计边缘像素占比 + Sobel 梯度幅值的 mean/median/P95。**共 4 维**。

分别在原始灰度图和 CLAHE 增强图上提取，各 4 维。

#### 5. 频域特征 (FFT)

2D 傅里叶变换后按半径分三个频带（低频 0~15%、中频 15%~50%、高频 50%+）统计能量密度，外加全图平均能量。**共 4 维**。

#### 6. 颜色统计

RGB 三通道分别计算均值、标准差和偏度 (skewness)。**共 9 维**。

#### 7. 前景/背景分割

Otsu 二值化 + 形态学开运算，统计前景占比和连通域数量。**共 2 维**。

#### 8. 形态学前景掩膜统计

使用黑帽(Black-hat) + 顶帽(Top-hat) 形态学运算提取小亮区域（对应人头），统计前景占比、前景内边缘密度、连通域数量/面积/最大面积。**共 6 维**。

#### 9. SIFT 关键点密度

SIFT 检测（最多 800 个关键点），统计密度、关键点尺寸均值/标准差、响应均值。**共 4 维**。

#### 10. FAST 角点密度

FAST 角点检测（阈值 25），角点常出现在头-肩交界处。统计密度、响应均值/标准差。**共 3 维**。

#### 11. 多尺度 Blob 检测

LoG (Laplacian of Gaussian) Blob 检测器，检测圆形头状区域（面积 5~500 px²，圆度 ≥0.3，凸度 ≥0.5）。统计密度、尺寸均值/标准差/P25/P75。**共 5 维**。

#### 12. ORB 关键点密度

ORB 关键点（最多 500 个），统计密度。**共 1 维**。

#### 13. 模糊度评分

Laplacian 方差，评估图像清晰度。**共 1 维**。

#### 14. 分块边缘方差

将图分为 4×4 网格，每格计算 Canny 边缘密度，统计均值和标准差，衡量边缘分布的空间不均匀性。**共 2 维**。

### 特征标准化与选择

- **StandardScaler**：每维特征标准化为均值 0、标准差 1
- **基于特征重要性的筛选**：使用 RandomForest 计算特征重要性，保留前 70%（30% 分位数阈值）

### 回归模型

#### 单模型

| 模型 | 配置 |
|------|------|
| GradientBoostingRegressor | 300 棵树，深度 5，学习率 0.05，子采样 0.8 |
| RandomForestRegressor | 300 棵树，深度 15，最少叶节点样本 5 |

#### 加权集成 (Weighted Ensemble)

1. 划分验证集（20%）分别评估 GBR 和 RF
2. 权重 = 1/MAE（反比于验证误差），归一化
3. 使用最终权重在全量数据上重新训练
4. 预测：`final = w_gbr × pred_gbr + w_rf × pred_rf`

### 结果对比

| 方法 | MAE ↓ | RMSE ↓ | R² ↑ | 训练/推理时间 |
|------|-------|--------|------|--------------|
| **CSRNet v4 (Deep)** | **11.69** | **19.23** | **0.960** | ~1h GPU 训练 |
| GBR v2 (Traditional) | 37.86 | 57.70 | 0.634 | ~5min CPU |
| RF v2 (Traditional) | 40.26 | 62.13 | 0.574 | ~5min CPU |
| Weighted Ensemble | 38.73 | 59.39 | 0.611 | ~5min CPU |

**结论**：深度学习方法在人群计数任务上显著优于传统方法。传统方法在极端密度（>100 人）场景下偏差较大（bias=-17.21），而 CSRNet 的密度图回归范式能端到端地学习空间特征，拟合斜率 0.962 接近理想值 1.0。

---

## 使用方式

### 环境依赖

```bash
pip install torch torchvision gradio numpy scipy scikit-learn scikit-image opencv-python pillow matplotlib joblib tensorboard
```

### 命令行推理（深度学习）

```bash
python predict.py path/to/image.jpg --model best_model.pth --show-density
```

### 启动 Web 交互界面

```bash
python app.py
```

界面左右栏分别展示 CSRNet 和传统 CV 的预测结果，包含预测人数和密度图可视化。

### 训练

```bash
# 深度模型训练
python train.py

# 传统 CV 模型训练
python traditional_train.py
```

### 数据集

使用 ShanghaiTech Part_B 数据集格式。`Data/` 目录下需包含 `train_data` 和 `test_data`，每个子目录有 `images/`（jpg 图片）和 `ground_truth/`（.mat 标注文件，包含 `image_info.location` 坐标矩阵）。

---

## 关键技术点总结

1. **密度图回归范式**：将计数问题转化为密度图回归，每个像素值表示该位置的人群密度，积分即为总人数，同时保留了空间分布信息。

2. **扩张卷积**：在不降低空间分辨率的前提下扩大感受野，使网络能感知不同尺度的人头。

3. **自适应高斯核**：根据局部人群密度动态调整高斯核 σ，密集区域峰值尖锐，稀疏区域平滑。

4. **分阶段解冻微调**：逐步解冻预训练层，先用固定 VGG 训练后端，再逐层解冻，防止过拟合和灾难性遗忘。

5. **多分辨率损失**：在多个尺度上监督密度图，增强模型对不同大小人头（近/远）的感知能力。

6. **多类型手工特征融合**：HOG（形状）+ LBP（纹理）+ GLCM（统计纹理）+ FFT（频域）+ 关键点检测（SIFT/FAST/Blob），从不同角度刻画人群特征。

7. **加权集成**：利用验证集误差倒数作为模型权重，自动为更准确的模型分配更高权重。
