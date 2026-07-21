import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import random
import numbers
import os
from collections import Counter
from tqdm import tqdm
from torchvision.transforms import functional as F1
from sklearn.metrics import confusion_matrix
from algorithm import *
from dataloader import WidarDataset, TransformSubset, XRF55Dataset, XRF55_TARGET_GESTURES, datatrcsie, datatecsie
from mytransforms import Physical_Mask_Augment, Frequency_Axis_Flip


def compute_flip_sensitivity(dataset_source, args):
    """
    Compute per-class direction sensitivity via Diversity Ratio.

    For each class c, measure how much adding frequency-flipped samples
    increases the intra-class feature diversity:

        r_c = tr(Cov(X_c ∪ X̃_c)) / tr(Cov(X_c))

    where X_c are original pixel features and X̃_c are their freq-axis flips.

    - r_c ≈ 1: flip adds no new mode → direction-agnostic (safe augmentation)
    - r_c >> 1: flip introduces a new mode → direction-sensitive (hard negative)

    This handles both:
    - XRF55 Push (only push samples) → flip adds pull-like mode → r_c high
    - Widar Push&Pull (both directions in one class) → flip adds nothing new → r_c ≈ 1

    Returns:
        direction_weights: dict mapping class_label (int) -> float in [0, 1]
            0.0 = direction-agnostic (flip = positive pair)
            1.0 = maximally direction-sensitive (flip = hard negative)
    """
    from PIL import Image
    from collections import defaultdict

    print('\n[Direction Sensitivity] Computing diversity-ratio based sensitivity...')

    # Get the root dataset to access file paths
    ds = dataset_source
    while hasattr(ds, 'subset'):
        ds = ds.subset
    while hasattr(ds, 'dataset'):
        ds = ds.dataset

    max_samples = 50

    # Support WidarDataset, XRF55Dataset, and datatrcsie
    if hasattr(ds, 'imgs'):
        file_label_pairs = [(path, label) for path, label in ds.imgs]
    elif hasattr(ds, 'img_paths') and hasattr(ds, 'img_labels'):
        file_label_pairs = list(zip(ds.img_paths, ds.img_labels))
    else:
        print('  WARNING: Dataset has no file list attribute, skipping direction detection')
        return {}

    # Group by class
    class_files = defaultdict(list)
    for path, label in file_label_pairs:
        class_files[label].append(path)

    def _img_to_feat(img_arr):
        """Flatten resized grayscale image to feature vector."""
        gray = img_arr.mean(axis=2).astype(np.float64)
        return gray.reshape(-1)

    class_ratios = {}
    for label, files in sorted(class_files.items()):
        sampled = files[:max_samples]
        orig_feats = []
        flip_feats = []
        for fpath in sampled:
            img = np.array(Image.open(fpath).resize((64, 64)))  # downsample for speed
            flipped = img[::-1, :, :].copy()  # freq-axis (vertical) flip
            orig_feats.append(_img_to_feat(img))
            flip_feats.append(_img_to_feat(flipped))

        orig_feats = np.stack(orig_feats)        # (N, D)
        flip_feats = np.stack(flip_feats)         # (N, D)
        combined = np.concatenate([orig_feats, flip_feats], axis=0)  # (2N, D)

        # Diversity = trace of covariance matrix
        div_orig = np.trace(np.cov(orig_feats, rowvar=False)) + 1e-8
        div_combined = np.trace(np.cov(combined, rowvar=False)) + 1e-8
        ratio = div_combined / div_orig

        class_ratios[label] = ratio
        print(f'  Class {label}: diversity_ratio={ratio:.4f} (n={len(sampled)})')

    # Normalize to [0, 1] -> probabilistic weight w_c
    all_ratios = list(class_ratios.values())
    r_min = min(all_ratios)
    r_max = max(all_ratios)
    r_range = r_max - r_min if r_max > r_min else 1.0

    direction_weights = {}
    for label, r in class_ratios.items():
        direction_weights[label] = (r - r_min) / r_range

    print(f'  Diversity ratio range: [{r_min:.4f}, {r_max:.4f}]')
    for label in sorted(direction_weights.keys()):
        w = direction_weights[label]
        tag = 'sensitive' if w > 0.5 else 'agnostic'
        print(f'  Class {label}: w_c={w:.4f} ({tag})')

    return direction_weights


def print_row(row, colwidth=10, latex=False):
    if latex:
        sep = " & "
        end_ = "\\\\"
    else:
        sep = "  "
        end_ = ""

    def format_val(x):
        if np.issubdtype(type(x), np.floating):
            x = "{:.10f}".format(x)
        return str(x).ljust(colwidth)[:colwidth]
    print(sep.join([format_val(x) for x in row]), end_)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def accuracy(network, loader, weights, usedpredict='p'):
    correct = 0
    total = 0
    weights_offset = 0

    network.eval()
    with torch.no_grad():
        for inputs, labels, pdlables, item in loader:
            x = inputs.cuda().float()
            y = labels.cuda().long()
            if usedpredict == 'p':
                p = network.predict(x)
            else:
                p = network.predict1(x)
            if weights is None:
                batch_weights = torch.ones(len(x))
            else:
                batch_weights = weights[weights_offset:
                                        weights_offset + len(x)]
                weights_offset += len(x)
            batch_weights = batch_weights.cuda()
            if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float() *
                            batch_weights.view(-1, 1)).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float() *
                            batch_weights).sum().item()
            total += batch_weights.sum().item()
    network.train()

    return correct / total


def save_confusion_and_perclass(network, loader, log_dir, num_classes):
    """在测试集上收集预测，保存混淆矩阵(.npy/.csv)与逐类精度(.csv)。
    回应 R1(对向手势混淆) / R3-5(per-class 分析)。"""
    import csv
    network.eval()
    all_y, all_p = [], []
    with torch.no_grad():
        for inputs, labels, pdlables, item in loader:
            x = inputs.cuda().float()
            p = network.predict(x).argmax(1).cpu().numpy()
            all_p.extend(p.tolist())
            all_y.extend(labels.numpy().tolist())
    network.train()
    y = np.array(all_y); pr = np.array(all_p)
    cm = confusion_matrix(y, pr, labels=list(range(num_classes)))
    np.save(os.path.join(log_dir, "confusion_matrix.npy"), cm)
    # 行归一化 recall
    with np.errstate(all='ignore'):
        cm_norm = cm / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)
    # 保存 csv
    with open(os.path.join(log_dir, "confusion_matrix.csv"), "w", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["true\\pred"] + list(range(num_classes)))
        for i in range(num_classes):
            w.writerow([i] + cm[i].tolist())
    with open(os.path.join(log_dir, "per_class_acc.csv"), "w", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["class", "n", "recall"])
        for i in range(num_classes):
            n = int(cm[i].sum())
            rec = float(cm_norm[i, i]) if n > 0 else 0.0
            w.writerow([i, n, f"{rec:.4f}"])
    print(f"[ConfMatrix] saved to {log_dir} (cm shape={cm.shape})")
    return cm


# ═══════════════════════════════════════════════════════════════════════════
# Widar 数据集构建器
# ═══════════════════════════════════════════════════════════════════════════

# Widar 统一的 6 类手势映射（所有实验共享，保证训练/测试标签一致）
WIDAR_GESTURE_MAP = {'G01': 0, 'G02': 1, 'G03': 2, 'G04': 3, 'G05': 4, 'G06': 5}
WIDAR_GESTURES = ['G01', 'G02', 'G03', 'G04', 'G05', 'G06']


def build_widar_loaders(args, img_transform, img_transformte):
    """
    根据实验类型构建 Widar 的训练/测试 DataLoader。

    支持的实验类型 (args.experiment):
      - in_domain   : 全量数据 (E1+E2+E3, G01-G06) 随机 80/20 划分
      - cross_user  : E1 内，按用户划分
      - cross_env   : E1+E2 训练，E3 测试
      - cross_loc   : E1 内，按位置划分
      - cross_ori   : E1 内，按方向划分
    """
    data_dir = args.data_path
    gmap = WIDAR_GESTURE_MAP
    rx_filter = [args.rx] if getattr(args, 'rx', None) else None

    if args.experiment == 'in_domain':
        # 全量数据，transform=None（稍后通过 TransformSubset 分别包装）
        full_dataset = WidarDataset(
            data_dir, transform=None,
            allowed_gestures=WIDAR_GESTURES,
            gesture_map=gmap, allowed_rx=rx_filter
        )
        # 80/20 随机划分（固定种子保证可复现）
        total = len(full_dataset)
        train_size = int(0.8 * total)
        test_size = total - train_size
        generator = torch.Generator().manual_seed(42)
        train_subset, test_subset = torch.utils.data.random_split(
            full_dataset, [train_size, test_size], generator=generator
        )
        dataset_source = TransformSubset(train_subset, img_transform)
        dataset_target = TransformSubset(test_subset, img_transformte)

    elif args.experiment == 'cross_user':
        # E1 内，按用户划分
        train_users = ['U05', 'U10', 'U11', 'U12', 'U13', 'U14', 'U15']
        test_users = ['U16', 'U17']
        dataset_source = WidarDataset(
            data_dir, transform=img_transform,
            allowed_envs=['E1'], allowed_users=train_users,
            allowed_gestures=WIDAR_GESTURES, gesture_map=gmap, allowed_rx=rx_filter
        )
        dataset_target = WidarDataset(
            data_dir, transform=img_transformte,
            allowed_envs=['E1'], allowed_users=test_users,
            allowed_gestures=WIDAR_GESTURES, gesture_map=gmap, allowed_rx=rx_filter
        )

    elif args.experiment == 'cross_env':
        # 支持 leave-one-out: --train_envs E1,E2 则测试剩余环境
        if getattr(args, 'train_envs', None):
            train_envs = [e.strip() for e in args.train_envs.split(',')]
            all_envs = ['E1', 'E2', 'E3']
            test_envs = [e for e in all_envs if e not in train_envs]
        else:
            train_envs = ['E1', 'E2']
            test_envs = ['E3']
        dataset_source = WidarDataset(
            data_dir, transform=img_transform,
            allowed_envs=train_envs,
            allowed_gestures=WIDAR_GESTURES, gesture_map=gmap, allowed_rx=rx_filter
        )
        dataset_target = WidarDataset(
            data_dir, transform=img_transformte,
            allowed_envs=test_envs,
            allowed_gestures=WIDAR_GESTURES, gesture_map=gmap, allowed_rx=rx_filter
        )

    elif args.experiment == 'cross_loc':
        # 支持 leave-one-out: --train_envs L1,L2,L3,L4
        if getattr(args, 'train_envs', None):
            train_locs = [e.strip() for e in args.train_envs.split(',')]
            all_locs = ['L1', 'L2', 'L3', 'L4', 'L5']
            test_locs = [e for e in all_locs if e not in train_locs]
        else:
            train_locs = ['L1', 'L2', 'L3', 'L4']
            test_locs = ['L5']
        dataset_source = WidarDataset(
            data_dir, transform=img_transform,
            allowed_envs=['E1'], allowed_locs=train_locs,
            allowed_gestures=WIDAR_GESTURES, gesture_map=gmap, allowed_rx=rx_filter
        )
        dataset_target = WidarDataset(
            data_dir, transform=img_transformte,
            allowed_envs=['E1'], allowed_locs=test_locs,
            allowed_gestures=WIDAR_GESTURES, gesture_map=gmap, allowed_rx=rx_filter
        )

    elif args.experiment == 'cross_ori':
        # 支持 leave-one-out: --train_envs O1,O2,O3,O4
        if getattr(args, 'train_envs', None):
            train_oris = [e.strip() for e in args.train_envs.split(',')]
            all_oris = ['O1', 'O2', 'O3', 'O4', 'O5']
            test_oris = [e for e in all_oris if e not in train_oris]
        else:
            train_oris = ['O1', 'O2', 'O3', 'O4']
            test_oris = ['O5']
        dataset_source = WidarDataset(
            data_dir, transform=img_transform,
            allowed_envs=['E1'], allowed_oris=train_oris,
            allowed_gestures=WIDAR_GESTURES, gesture_map=gmap, allowed_rx=rx_filter
        )
        dataset_target = WidarDataset(
            data_dir, transform=img_transformte,
            allowed_envs=['E1'], allowed_oris=test_oris,
            allowed_gestures=WIDAR_GESTURES, gesture_map=gmap, allowed_rx=rx_filter
        )
    else:
        raise ValueError(f"Unknown experiment type: {args.experiment}")

    return dataset_source, dataset_target


def build_widar_source_eval(args, img_transformte):
    """
    构建源域评估数据集（使用 test transform，无增强，确定性评估）。
    与训练集同分布但不做随机增强。
    """
    data_dir = args.data_path
    gmap = WIDAR_GESTURE_MAP
    rx_filter = [args.rx] if getattr(args, 'rx', None) else None

    if args.experiment == 'in_domain':
        full_dataset = WidarDataset(
            data_dir, transform=None,
            allowed_gestures=WIDAR_GESTURES,
            gesture_map=gmap, allowed_rx=rx_filter
        )
        total = len(full_dataset)
        train_size = int(0.8 * total)
        test_size = total - train_size
        generator = torch.Generator().manual_seed(42)
        train_subset, _ = torch.utils.data.random_split(
            full_dataset, [train_size, test_size], generator=generator
        )
        return TransformSubset(train_subset, img_transformte)

    elif args.experiment == 'cross_user':
        train_users = ['U05', 'U10', 'U11', 'U12', 'U13', 'U14', 'U15']
        return WidarDataset(
            data_dir, transform=img_transformte,
            allowed_envs=['E1'], allowed_users=train_users,
            allowed_gestures=WIDAR_GESTURES, gesture_map=gmap, allowed_rx=rx_filter
        )

    elif args.experiment == 'cross_env':
        if getattr(args, 'train_envs', None):
            train_envs = [e.strip() for e in args.train_envs.split(',')]
        else:
            train_envs = ['E1', 'E2']
        return WidarDataset(
            data_dir, transform=img_transformte,
            allowed_envs=train_envs,
            allowed_gestures=WIDAR_GESTURES, gesture_map=gmap, allowed_rx=rx_filter
        )

    elif args.experiment == 'cross_loc':
        if getattr(args, 'train_envs', None):
            train_locs = [e.strip() for e in args.train_envs.split(',')]
        else:
            train_locs = ['L1', 'L2', 'L3', 'L4']
        return WidarDataset(
            data_dir, transform=img_transformte,
            allowed_envs=['E1'], allowed_locs=train_locs,
            allowed_gestures=WIDAR_GESTURES, gesture_map=gmap, allowed_rx=rx_filter
        )

    elif args.experiment == 'cross_ori':
        if getattr(args, 'train_envs', None):
            train_oris = [e.strip() for e in args.train_envs.split(',')]
        else:
            train_oris = ['O1', 'O2', 'O3', 'O4']
        return WidarDataset(
            data_dir, transform=img_transformte,
            allowed_envs=['E1'], allowed_oris=train_oris,
            allowed_gestures=WIDAR_GESTURES, gesture_map=gmap, allowed_rx=rx_filter
        )
    else:
        raise ValueError(f"Unknown experiment type: {args.experiment}")


def _assign_random_domain_labels(dataset, K):
    """
    M2 消融用：随机分配伪域标签（均匀分布到 0..K-1），替代 K-Means 聚类。
    返回 Counter 统计各域样本数。
    """
    N = len(dataset)
    random_labels = np.random.randint(0, K, size=N)
    indices = np.arange(N)
    dataset.set_labels_by_index(random_labels, indices, 'pdlabel')
    return Counter(random_labels.tolist())


def trainer(trainmodel, img_transform, img_transformte, device, opta, scheduler,
            total_epoch=200, log_dir="log", args=None):
    num_classes = args.num_classes if args else 8
    ones = torch.sparse.torch.eye(num_classes)
    ones = ones.to(device)
    bestac = 0.0

    ablation = args.ablation if (args and hasattr(args, 'ablation')) else 'full'
    print(f'\n[Ablation Mode] {ablation}')

    # ── 构建数据集 ──────────────────────────────────────────────────────────
    if args is not None and args.dataset == 'widar':
        dataset_source, dataset_target = build_widar_loaders(
            args, img_transform, img_transformte
        )
        # 源域评估集（无增强，确定性评估）
        dataset_source_eval = build_widar_source_eval(args, img_transformte)
    else:
        # XRF55 逻辑 — 支持 in_domain, cross_user, cross_env
        data_path = args.data_path if (args and args.data_path) else \
            r'C:\Users\G\Downloads\GesFiCode-main-v2\GesFiCode-main\Processed_Data'
        experiment = args.experiment if args else 'cross_user'

        if experiment == 'cross_env':
            # 留一法: 指定 test_scene, 其余作为训练集
            test_scene = int(args.test_scene) if (args and args.test_scene) else 4
            all_scenes = [1, 2, 3, 4]
            train_scenes = [s for s in all_scenes if s != test_scene]
            base_dir = os.path.dirname(data_path) if 'Scene' in data_path else data_path
            train_dirs = [os.path.join(base_dir, f'Processed_Data_Scene_{s}') for s in train_scenes]
            test_dirs = [os.path.join(base_dir, f'Processed_Data_Scene_{test_scene}')]
            print(f'[XRF55 cross_env] Train scenes: {train_scenes}, Test scene: {test_scene}')
            dataset_source = XRF55Dataset(train_dirs, transform=img_transform)
            dataset_target = XRF55Dataset(test_dirs, transform=img_transformte)
            dataset_source_eval = XRF55Dataset(train_dirs, transform=img_transformte)
        elif experiment == 'in_domain':
            # Scene1 内随机 80/20 划分
            scene1_dir = data_path if 'Scene' in data_path else os.path.join(data_path, 'Processed_Data_Scene_1')
            full_ds = XRF55Dataset([scene1_dir], transform=None)
            total = len(full_ds)
            train_size = int(0.8 * total)
            test_size = total - train_size
            generator = torch.Generator().manual_seed(42)
            train_sub, test_sub = torch.utils.data.random_split(full_ds, [train_size, test_size], generator=generator)
            dataset_source = TransformSubset(train_sub, img_transform)
            dataset_target = TransformSubset(test_sub, img_transformte)
            dataset_source_eval = TransformSubset(train_sub, img_transformte)
        else:
            # cross_user: Scene1
            # 支持 LOUO: --xrf_test_users 5 表示 user5 做测试,其余训练
            scene1_dir = data_path if 'Scene' in data_path else os.path.join(data_path, 'Processed_Data_Scene_1')
            xrf_test_u = getattr(args, 'xrf_test_users', None)
            if xrf_test_u is not None:
                test_users = {int(xrf_test_u)}
                train_users = set(range(1, 31)) - test_users
                print(f'[XRF55 LOUO] test_user={xrf_test_u}, train_users={sorted(train_users)[:3]}...({len(train_users)}人)')
            else:
                train_users = set(range(1, 25))
                test_users = set(range(25, 31))
            dataset_source = XRF55Dataset([scene1_dir], transform=img_transform, allowed_users=train_users)
            dataset_target = XRF55Dataset([scene1_dir], transform=img_transformte, allowed_users=test_users)
            dataset_source_eval = XRF55Dataset([scene1_dir], transform=img_transformte, allowed_users=train_users)

    batch_size = args.batch_size if args else 32

    # ── 数据缩减 ──────────────────────────────────────────────────────────────
    data_frac = getattr(args, 'data_fraction', 1.0) if args else 1.0
    if data_frac < 1.0:
        n_total = len(dataset_source)
        n_keep = max(1, int(n_total * data_frac))
        generator = torch.Generator().manual_seed(42)
        subset, _ = torch.utils.data.random_split(
            dataset_source, [n_keep, n_total - n_keep], generator=generator
        )
        class SubsetProxy(torch.utils.data.Dataset):
            def __init__(self, subset):
                self.subset = subset
            def _root(self):
                ds = self.subset
                while hasattr(ds, 'dataset'):
                    ds = ds.dataset
                return ds
            def set_labels_by_index(self, tlabels=None, tindex=None, label_type='domain_label'):
                self._root().set_labels_by_index(tlabels, tindex, label_type)
            def __getitem__(self, idx):
                return self.subset[idx]
            def __len__(self):
                return len(self.subset)
        dataset_source = SubsetProxy(subset)
        print(f'[Data Fraction] Using {data_frac*100:.0f}% of training data: {n_keep}/{n_total}')

    train_loader = torch.utils.data.DataLoader(
        dataset=dataset_source, batch_size=batch_size,
        shuffle=True, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=dataset_target, batch_size=batch_size,
        shuffle=False, num_workers=2
    )
    source_eval_loader = torch.utils.data.DataLoader(
        dataset=dataset_source_eval, batch_size=batch_size,
        shuffle=False, num_workers=2
    )
    lengthtr = len(train_loader)
    lengthte = len(test_loader)
    print(f'Train set size: {len(dataset_source)} samples, {lengthtr} batches')
    print(f'Test set size:  {len(dataset_target)} samples, {lengthte} batches')
    print(f'Source eval size: {len(dataset_source_eval)} samples')

    # ── Direction sensitivity (manual specification) ────────────────────
    # Widar: 手势为方向合并类别(Push&Pull合一), 方向解耦不适用, 权重全0
    # XRF55: 基于物理方向语义的先验权重
    #   G44-G49 (push/pull/swipeL/R/U/D) = 方向敏感 → w=0.5
    #   G50-G51 (circle/cross) = 方向无关 → w=0.0
    dir_w = getattr(args, 'dir_weight', 0.5) if args else 0.5
    if args.dataset == 'widar':
        direction_sensitive = {i: 0.0 for i in range(6)}
        print(f'\n[Direction Sensitivity] Widar: all agnostic (merged gestures)')
    else:
        direction_sensitive = {i: dir_w for i in range(6)}  # class 0-5: push/pull/swipe → sensitive
        direction_sensitive[6] = 0.0  # class 6: circle → agnostic
        direction_sensitive[7] = 0.0  # class 7: cross → agnostic
        print(f'\n[Direction Sensitivity] Physics-informed: classes 0-5 (directional)={dir_w}, classes 6-7 (agnostic)=0.0')

    # ── 物理增强实例化 ────────────────────────────────────────────────────────
    var_pct = args.variance_percentile if args else 30
    mask_r = args.mask_ratio if args else 0.15
    p_size = args.patch_size if args else 16
    phys_mask_aug = Physical_Mask_Augment(
        variance_percentile=var_pct,
        mask_ratio=mask_r,
        patch_size=p_size
    )
    freq_flip = Frequency_Axis_Flip()

    acc_file = os.path.join(log_dir, "acc.txt")
    bestacc_file = os.path.join(log_dir, "bestacc.txt")

    with open(acc_file, "w") as f:
        for round in range(total_epoch):
            # 更新模型内的 round 计数器（驱动 GRL alpha 自适应衰减）
            trainmodel._round = round

            trainmodel.train()
            print(f'\n======== ROUND {round} ========  [Ablation: {ablation}]')

            # ============================================================
            # M0 (Baseline): 纯 CE 分类，所有 epoch 走 4 个 step
            # ============================================================
            if ablation == 'M0':
                print('==== M0 Baseline: CE Only ====')
                loss_list = ['total', 'cls']
                print_row(['epoch'] + [item + '_loss' for item in loss_list], colwidth=15)
                for step in range(4):
                    for inputs, labels, pdlables, item in train_loader:
                        loss_result_dict = trainmodel.update_baseline(
                            inputs, labels, opta
                        )
                    print_row([step] + [loss_result_dict.get(item, 0) for item in loss_list], colwidth=15)

            else:
                # ══════════════════════════════════════════════════════════
                # 阶段①: Feature Update & Contrastive Pre-training
                # ══════════════════════════════════════════════════════════
                print('==== Stage 1: SupCon Pre-training ====')

                # M1 消融：永远不传 SupCon 视图（包括前 3 个 epoch）
                use_supcon = (ablation not in ('M1',))

                if round <= 2 and use_supcon:
                    loss_list = ['total', 'cls', 'supcon']
                    print_row(['epoch'] + [item + '_loss' for item in loss_list], colwidth=15)
                    for step in range(2):
                        for inputs, labels, pdlables, item in train_loader:
                            # 方差先验掩码增强，构造双视图
                            x_raw = inputs.cuda().float()
                            x_view1, x_view2 = phys_mask_aug(x_raw)
                            loss_result_dict = trainmodel.update_a(
                                inputs, labels, pdlables, opta,
                                x_view1=x_view1, x_view2=x_view2
                            )
                        print_row([step] + [loss_result_dict.get(item, 0) for item in loss_list], colwidth=15)
                else:
                    loss_list = ['total', 'cls']
                    print_row(['epoch'] + [item + '_loss' for item in loss_list], colwidth=15)
                    for step in range(2):
                        for inputs, labels, pdlables, item in train_loader:
                            loss_result_dict = trainmodel.update_a(
                                inputs, labels, pdlables, opta,
                                x_view1=None, x_view2=None
                            )
                        print_row([step] + [loss_result_dict.get(item, 0) for item in loss_list], colwidth=15)

                # ══════════════════════════════════════════════════════════
                # 阶段②: Latent Domain Characterization & PCL
                # ══════════════════════════════════════════════════════════
                # M2 消融：跳过整个 Stage②，随机分配域标签
                if ablation == 'M2':
                    print('==== Stage 2: SKIPPED (M2 ablation) ====')
                    # 每个 epoch 重新随机分配域标签
                    K = args.latent_domain_num
                    Cpd = _assign_random_domain_labels(dataset_source, K)
                    print(f'  Random domain assignment: {Cpd}')
                else:
                    skip_grl_s2 = (ablation == 'M3')
                    print('==== Stage 2: Latent Domain PCL ====')
                    if skip_grl_s2:
                        print('  (GRL disabled for M3 ablation)')
                    loss_list = ['total', 'dis', 'proto', 'ent_pen']
                    print_row(['epoch'] + [item + '_loss' for item in loss_list], colwidth=15)
                    for step in range(1):
                        for inputs, labels, pdlables, item in train_loader:
                            loss_result_dict = trainmodel.update_d(
                                inputs, labels, pdlables, opta,
                                skip_grl=skip_grl_s2
                            )
                        print_row([step] + [loss_result_dict.get(item, 0) for item in loss_list], colwidth=15)

                    Cpd = trainmodel.set_dlabel(train_loader)

                # ══════════════════════════════════════════════════════════
                # 阶段③: Domain-invariant & Hard Negative Contrastive
                # ══════════════════════════════════════════════════════════
                skip_grl_s3 = (ablation == 'M3')
                use_hardnce = (ablation not in ('M4',))

                print('==== Stage 3: Domain-invariant + HardNCE ====')
                if skip_grl_s3:
                    print('  (GRL disabled for M3 ablation)')
                if not use_hardnce:
                    print('  (HardNCE disabled for M4 ablation)')

                loss_list = ['total', 'cls', 'dis', 'hard']
                print_row(['epoch'] + [item + '_loss' for item in loss_list], colwidth=15)
                for step in range(1):
                    for inputs, labels, pdlables, item in train_loader:
                        # 频轴翻转构造困难负样本（M4 消融时不构造）
                        if use_hardnce:
                            x_raw = inputs.cuda().float()
                            x_mirrored = freq_flip(x_raw)
                            # Build per-sample direction mask from pre-computed sensitivity
                            if direction_sensitive:
                                dir_mask = torch.tensor(
                                    [direction_sensitive.get(l.item(), 0.0) for l in labels],
                                    dtype=torch.float32, device=x_raw.device
                                )
                            else:
                                dir_mask = None
                        else:
                            x_mirrored = None
                            dir_mask = None

                        step_vals = trainmodel.update(
                            inputs, labels, pdlables, opta, Cpd,
                            x_mirrored=x_mirrored,
                            skip_grl=skip_grl_s3,
                            direction_mask=dir_mask
                        )
                    print_row([step] + [step_vals.get(item, 0) for item in loss_list], colwidth=15)

            # ══════════════════════════════════════════════════════════════
            # 评估：源域精度 + 目标域精度
            # ══════════════════════════════════════════════════════════════
            src_acc = accuracy(trainmodel, source_eval_loader, None)
            tgt_acc = accuracy(trainmodel, test_loader, None)
            print(f'  Source acc: {src_acc:.6f}  |  Target acc: {tgt_acc:.6f}')

            if tgt_acc > bestac:
                bestac = tgt_acc
            print(f'  Best target acc: {bestac:.6f}')

            f.write(f"Round {round}: src_acc={src_acc:.6f}, tgt_acc={tgt_acc:.6f}, best_tgt={bestac:.6f}\n")
            f.flush()

            # 学习率衰减
            scheduler.step()

    with open(bestacc_file, "w") as f:
        f.write(f"{bestac:.6f}\n")

    # ── 训练结束：保存混淆矩阵 + per-class 精度（最终模型）──────────────────
    try:
        save_confusion_and_perclass(trainmodel, test_loader, log_dir, num_classes)
    except Exception as e:
        print(f"[ConfMatrix] skipped due to error: {e}")

    print(f"\nResults saved to {log_dir}")
    print(f"acc.txt: {acc_file}")
    print(f"bestacc.txt: {bestacc_file}")
