# 人群计数 (Crowd Counting) —— 深度学习 vs 传统计算机视觉

本项目分别基于 **Swin-V2-T + LoRA** 深度学习方法和**多类型手工特征 + Stacking 集成回归**传统 CV 方法构建人群计数系统，采用密度图回归范式实现高密度场景下的人数预测与密度图可视化。

**数据集**：ShanghaiTech Part_B（400 张训练，316 张测试，人数范围 9~578）。

## 项目结构

```
├── Data/                       # 数据集（需自行准备）
├── model.py                    # Swin-V2-T + LoRA + 轻量计数头
├── dataset.py                  # 数据加载、自适应高斯核密度图生成、数据增强
├── train.py                    # Swin-V2 训练脚本
├── predict.py                  # Swin-V2 命令行推理
├── app.py                      # Gradio Web 交互界面（深度 vs 传统）
├── traditional_model.py        # 传统 CV：13 类手工特征提取 + Stacking 集成
├── traditional_train.py        # 传统 CV 训练评估脚本
├── 训练日志/                    # 训练日志与对比散点图（打包存放）
└── 面试准备_SwinV2人群计数.md   # 面试问题与回答
```

## 方法一：深度学习 —— Swin-V2-T + LoRA

### 整体架构 (model.py)

**编码器-解码器设计，三部分：**

**① 骨干网络 Swin-V2-T**：torchvision 预训练权重（ImageNet-1K），4 个 Stage 输出通道 96/192/384/768，下采样率 4/8/16/32。移位窗口自注意力提供全局感受野，线性计算复杂度 O(N)。

**② 分层 LoRA 微调**：对所有 Swin Block 的注意力层（qkv、proj）和 MLP 层（fc1、fc2）注入 LoRA 适配器（r=16, α=16）。浅层 Stage1-2 冻结原始权重仅 LoRA 适配（保留通用纹理特征），深层 Stage3-4 解冻全量微调 + LoRA（学习任务特异性语义特征）。总计 29M 参数，27.8M 可训练。

**③ 轻量计数头 LightCountingHead**：取 Stage2（H/8, 192ch）和 Stage3（H/16, 384ch）经 1×1 卷积投影至 128 通道，Stage3 上采样后相加融合，两层 3×3 卷积 + ReLU 精炼，1×1 卷积输出单通道密度图。设计约束：不用 BN（batch=4 统计不稳定）、不用 SE（通道重标定会抑制空间模式）、不用空洞卷积（自注意力已提供全局感受野）。

```
输入 (3, H, W)
  → Swin-V2-T Stage1 (96ch, H/4, W/4)   [冻结 + LoRA]
  → Swin-V2-T Stage2 (192ch, H/8, W/8)  [冻结 + LoRA] ──┐
  → Swin-V2-T Stage3 (384ch, H/16, W/16) [全量 + LoRA] ──┼──→ 计数头 → 密度图 (1, H/8, W/8)
  → Swin-V2-T Stage4 (768ch, H/32, W/32) [全量 + LoRA]
```

### 密度图生成 (dataset.py)

**几何自适应高斯核**：对每个人头标注，以 k-NN（k=3）平均距离 × β=0.3 计算局部自适应 σ，裁剪至 [0.5, 12.0]。密集区域 σ 小、高斯核尖锐；稀疏区域 σ 大、分布平滑。密度图在 1/8 分辨率下生成，每个高斯核积分为 1。

### 数据增强 (dataset.py)

| 策略 | 方法 | 参数 |
|------|------|------|
| 随机缩放 + 裁剪 | 缩放到 0.5-1.0×，随机裁剪 | 512×384 |
| 水平翻转 | 50% 概率，同步翻转密度图 | p=0.5 |
| 颜色抖动 | 亮度/对比度/饱和度随机调整 | ±0.2 |
| 尺寸对齐 | 调整为 8 的倍数 | - |

### 损失函数 (train.py)

**多分辨率密度损失 + 分阶段计数损失**：

- **密度损失** = 1.0 × 全分辨率 SmoothL1 + 0.1 × 半分辨率（2×2 AvgPool）SmoothL1
- **计数损失**（分两阶段）：
  - Phase 1（1-50 epoch）：仅 MAE，稳定基础计数
  - Phase 2（51-120 epoch）：MAE + MSE（权重 0.5），精细调整大误差样本
- 总损失 = 密度损失 + 2.0 × 计数损失

选用 SmoothL1Loss 而非 MSE：对标注噪声和离群点更鲁棒。

### 训练策略 (train.py)

| 配置项 | 值 |
|--------|-----|
| 优化器 | AdamW (wd=1e-4) |
| 学习率 | CosineAnnealingLR (2e-4 → 1e-6) |
| Batch size | 4 |
| Epochs | 120（早停 patience=30） |
| 混合精度 | AMP (GradScaler + autocast) |
| 梯度裁剪 | max_norm=1.0 |
| 输入尺寸 | 640×480 |

### 最终结果

| 指标 | 值 |
|------|-----|
| Test MAE | **9.64** |
| Test RMSE | **15.79** |
| R² | **0.9725** |

---

## 方法二：传统 CV —— 手工特征 + Stacking 集成回归

### 整体流程 (traditional_model.py)

```
输入图像
  → CLAHE 光照归一化
  → 双路特征（原始灰度 + CLAHE 增强）
  → 13 类手工特征提取（~310 维）
  → StandardScaler 标准化
  → RF 特征重要性筛选（保留 top 70%）
  → StackingRegressor（GBR + RF + Ridge → Ridge 元学习器，5 折 CV）
  → 预测人数
```

### 预处理

- **CLAHE**（clipLimit=2.0, tile=8×8）：消除过曝/欠曝对特征的影响
- **形态学前/背景掩膜**：黑帽 + 顶帽加权融合，定位人头类小尺度亮目标
- **Laplacian 模糊度评分**：感知图像质量

### 特征体系（13 类）

| 类别 | 提取器 | 维度 | 捕获模式 |
|------|--------|------|----------|
| 梯度方向 | HOG（仅原图） | 36 | 4×4 分块 9 方向梯度分布，人头-肩部轮廓 |
| 局部纹理 | Uniform LBP | 243 | radius=2/16 邻域直方图，纹理重复周期 |
| 灰度共生 | GLCM（原图 + CLAHE） | 24 | 3 距离 × 4 角度 × 6 纹理属性统计量 |
| 边缘统计 | Canny + Sobel（原图 + CLAHE） | 8 | 边缘比例 + 梯度幅值分布 |
| 频域能量 | FFT 环带 | 4 | 低/中/高频能量 + 全局均值 |
| 颜色统计 | RGB 逐通道 | 9 | 每通道均值/标准差/偏度 |
| 前景分割 | Otsu + 连通域 | 2 | 前景比例 + 归一化连通域数 |
| 掩膜统计 | 形态学前景掩膜 | 6 | 前景内边缘密度 + 连通域面积分布 |
| 关键点密度 | SIFT | 3 | 密度 + 尺度均值/标准差（尺度不变） |
| 角点密度 | FAST | 3 | 密度 + 响应统计量，头-肩交界高曲率点 |
| 团块检测 | LoG Blob | 5 | 密度 + 尺度分布，圆形头部轮廓 |
| 图像质量 | Laplacian 模糊度 | 1 | 全局锐度评估 |
| 空间分布 | 分块边缘方差 | 2 | 4×4 分块边缘密度的均值与标准差 |

### Stacking 集成回归

| 层 | 模型 | 关键配置 |
|----|------|----------|
| 基学习器 | GBR | n=200, depth=4, lr=0.03, min_samples_leaf=15, early_stop=5 |
| 基学习器 | RF | n=200, depth=12, min_samples_leaf=10 |
| 基学习器 | Ridge | α=1.0（L2 正则化线性基线） |
| 元学习器 | Ridge | α=1.0，5 折 CV 生成 out-of-fold 元特征 |

GBR 内置 validation_fraction=0.1 + n_iter_no_change=5 早停机制抑制过拟合。Stacking 通过 5 折交叉验证生成元特征，元学习器在交叉验证框架内学习最优融合策略，避免单次验证集权重与全量 refit 模型不匹配的问题。

### 结果对比

| 方法 | MAE ↓ | RMSE ↓ | R² ↑ |
|------|-------|--------|------|
| **Swin-V2-T + LoRA** | **9.64** | **15.79** | **0.9725** |
| Stacking 集成（传统 CV） | ~34 | ~53 | ~0.68 |
| GBR 单模型（传统 CV） | ~35 | ~55 | ~0.66 |

---

## 使用方式

### 环境依赖

```bash
pip install torch torchvision gradio numpy scipy scikit-learn scikit-image opencv-python pillow matplotlib joblib tensorboard
```

### 命令行推理（Swin-V2）

```bash
python predict.py path/to/image.jpg --model best_model_swin.pth --show-density
```

### Web 交互界面

```bash
python app.py
```

左右栏分别展示 Swin-V2-T + LoRA 和传统 CV 的预测人数与密度图可视化。

### 训练

```bash
# Swin-V2 训练
python train.py

# 传统 CV 训练
python traditional_train.py
```

### 数据集

使用 ShanghaiTech Part_B 格式。`Data/` 目录下需包含 `train_data` 和 `test_data`，每子目录含 `images/`（jpg）和 `ground_truth/`（.mat 标注文件，坐标矩阵在 `image_info.location`）。

---

## 关键技术点总结

1. **密度图回归范式**：将计数转化为像素级密度回归，密度图积分 = 总人数，同时保留空间分布信息。
2. **移位窗口自注意力**：Swin-V2-T 提供全局感受野与线性复杂度，突破传统 CNN 感受野受限瓶颈。
3. **分层 LoRA 微调**：浅层冻结 + LoRA 保留通用特征，深层全量微调 + LoRA 学习任务特异性特征。
4. **几何自适应高斯核**：根据 k-NN 局部密度动态调整 σ，密集区尖锐、稀疏区平滑。
5. **分阶段计数损失**：先用 MAE 稳定计数，再用 MSE 精细调整，实现课程式训练。
6. **多类型手工特征融合**：HOG + LBP + GLCM + FFT + SIFT + FAST + Blob，从纹理/边缘/频域/关键点多角度编码人群信息。
7. **Stacking 集成**：5 折交叉验证生成元特征，Ridge 元学习器在 CV 框架内学习最优融合权重。
