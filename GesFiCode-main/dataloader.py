import torch.utils.data as data
from PIL import Image
import os
import re
import numpy as np
import torchvision.transforms as transforms


# ═══════════════════════════════════════════════════════════════════════════
# Widar 数据集通用类
# 文件名格式: E{env}_U{user}_G{gesture}_L{loc}_O{ori}_R{rep}.png
# 支持通过 allowed_* 参数灵活过滤，动态构建从 0 开始的连续标签映射
# ═══════════════════════════════════════════════════════════════════════════

# Widar 文件名正则：匹配 E1_U05_G01_L1_O1_R01.png 或 E1_U05_G01_L1_O1_R01_3AP.png 格式 (6AP/3AP)
_WIDAR_PATTERN = re.compile(
    r'^(E\d+)_(U\d+)_(G\d+)_(L\d+)_(O\d+)_(R\d+)(?:_3AP)?\.png$'
)
# Widar 1AP 格式：匹配 RX1_E1_U05_G01_L1_O1_R01.png
_WIDAR_1AP_PATTERN = re.compile(
    r'^(RX\d+)_(E\d+)_(U\d+)_(G\d+)_(L\d+)_(O\d+)_(R\d+)\.png$'
)


class WidarDataset(data.Dataset):
    """
    Widar 通用数据集。

    Parameters
    ----------
    data_dir : str
        图片所在目录路径
    transform : torchvision.transforms.Compose or None
        数据增强/预处理流水线
    allowed_envs : list[str] or None
        允许的环境，如 ['E1']；None 表示不限
    allowed_users : list[str] or None
        允许的用户，如 ['U05','U10']；None 表示不限
    allowed_gestures : list[str] or None
        允许的手势，如 ['G01','G02',...,'G06']；None 表示不限
    allowed_locs : list[str] or None
        允许的位置，如 ['L1','L2','L3','L4']；None 表示不限
    allowed_oris : list[str] or None
        允许的方向，如 ['O1','O2','O3','O4']；None 表示不限
    gesture_map : dict or None
        预设的手势→标签映射 (如 {'G01':0,...,'G06':5})。
        若为 None 则自动扫描构建；跨域实验中传入预设映射可保证训练/测试一致。
    """

    def __init__(self, data_dir, transform=None,
                 allowed_envs=None, allowed_users=None,
                 allowed_gestures=None, allowed_locs=None,
                 allowed_oris=None, allowed_rx=None, gesture_map=None):
        self.transform = transform
        self.img_paths = []
        self.img_labels = []
        self.pdlabels = []

        # 将 list 转为 set 加速查找
        env_set = set(allowed_envs) if allowed_envs else None
        user_set = set(allowed_users) if allowed_users else None
        gest_set = set(allowed_gestures) if allowed_gestures else None
        loc_set = set(allowed_locs) if allowed_locs else None
        ori_set = set(allowed_oris) if allowed_oris else None
        rx_set = set(allowed_rx) if allowed_rx else None

        # ── 第一遍扫描：收集有效文件，提取 gesture ID 集合 ─────────────────
        valid_files = []          # [(filepath, gesture_str), ...]
        gesture_ids_found = set()

        all_files = [f for f in os.listdir(data_dir) if f.endswith('.png')]
        for f in all_files:
            # 尝试匹配 6AP 格式
            m = _WIDAR_PATTERN.match(f)
            if m is not None:
                env, user, gesture, loc, ori, rep = m.groups()
                rx = None
            else:
                # 尝试匹配 1AP 格式
                m = _WIDAR_1AP_PATTERN.match(f)
                if m is None:
                    continue
                rx, env, user, gesture, loc, ori, rep = m.groups()

            # 逐维度过滤
            if rx_set and (rx is None or rx not in rx_set):
                continue
            if env_set and env not in env_set:
                continue
            if user_set and user not in user_set:
                continue
            if gest_set and gesture not in gest_set:
                continue
            if loc_set and loc not in loc_set:
                continue
            if ori_set and ori not in ori_set:
                continue

            filepath = os.path.join(data_dir, f)
            valid_files.append((filepath, gesture))
            gesture_ids_found.add(gesture)

        # ── 构建动态映射（排序后从 0 开始连续编号）──────────────────────────
        if gesture_map is not None:
            self.gesture_map = gesture_map
        else:
            gesture_list = sorted(gesture_ids_found)
            self.gesture_map = {g: i for i, g in enumerate(gesture_list)}

        # ── 第二遍：赋标签 ────────────────────────────────────────────────
        for filepath, gesture in valid_files:
            label = self.gesture_map[gesture]
            self.img_paths.append(filepath)
            self.img_labels.append(label)
            self.pdlabels.append(0)   # 伪域标签初始化为 0

        self.n_data = len(self.img_paths)
        print(f'[WidarDataset] loaded {self.n_data} samples, '
              f'gesture_map={self.gesture_map}')

    def set_labels_by_index(self, tlabels=None, tindex=None, label_type='domain_label'):
        """与 Stage-2 伪域标签更新接口兼容"""
        if label_type == 'pdlabel':
            self.pdlabels = np.array(self.pdlabels)
            self.pdlabels[tindex.astype(int)] = tlabels.astype(int)
            self.pdlabels = self.pdlabels.tolist()

    def __getitem__(self, item):
        img_path = self.img_paths[item]
        label = self.img_labels[item]
        pdlabel = self.pdlabels[item]

        img = Image.open(img_path)
        img = img.resize((224, 224))

        if self.transform is not None:
            img = self.transform(img)
            label = int(label)
            pdlabel = int(pdlabel)

        return img, label, pdlabel, item

    def __len__(self):
        return self.n_data


class TransformSubset(data.Dataset):
    """
    为 random_split 产生的 Subset 包装不同的 transform。
    用于 in_domain 实验：底层 WidarDataset 以 transform=None 创建，
    再通过此类分别给训练/测试子集设置各自的 transform。
    同时代理 set_labels_by_index 方法。
    """

    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    @property
    def _root_dataset(self):
        """获取底层的 WidarDataset 实例"""
        ds = self.subset
        while hasattr(ds, 'dataset'):
            ds = ds.dataset
        return ds

    def set_labels_by_index(self, tlabels=None, tindex=None, label_type='domain_label'):
        """代理到底层 WidarDataset"""
        self._root_dataset.set_labels_by_index(tlabels, tindex, label_type)

    def __getitem__(self, idx):
        img_path, label, pdlabel, global_idx = self.subset[idx]
        # subset 返回的 img_path 可能已经是 PIL Image（如果底层 transform=None）
        # 底层 WidarDataset transform=None 时, __getitem__ 返回的 img 是 PIL Image
        if self.transform is not None:
            img_path = self.transform(img_path)
            label = int(label)
            pdlabel = int(pdlabel)
        return img_path, label, pdlabel, global_idx

    def __len__(self):
        return len(self.subset)


# ═══════════════════════════════════════════════════════════════════════════
# 原有 XRF55 数据集类（保持兼容）
# ═══════════════════════════════════════════════════════════════════════════

class datatrcsie(data.Dataset):
    def __init__(self, data_list, transform=False):
        self.transform = transform
        self.img_paths = []
        self.img_labels = []
        self.pdlabels = []
        count = 0

        all_files = [f for f in os.listdir(data_list) if f.endswith('.png')]
        for f in all_files:
            try:
                # Processed_Data format: U{user:02d}_G{gesture}_R{repetition:02d}.png
                parts = f.replace('.png', '').split('_')
                user_id = int(parts[0][1:])   # 'U01' -> 1
                gesture_id = int(parts[1][1:])  # 'G44' -> 44
                # repetition = int(parts[2][1:])  # unused but kept for validation
            except:
                continue
            if 1 <= user_id <= 24 and 44 <= gesture_id <= 51:
                self.img_paths.append(os.path.join(data_list, f))
                self.img_labels.append(gesture_id - 44)
                self.pdlabels.append(gesture_id - gesture_id)
                count += 1
        self.n_data = count
        print(count)

    def set_labels_by_index(self, tlabels=None, tindex=None, label_type='domain_label'):
        if label_type == 'pdlabel':
            self.pdlabels=np.array(self.pdlabels)
            self.pdlabels[tindex.astype(int)] = tlabels.astype(int)
            self.pdlabels.tolist()

    def __getitem__(self, item):
        img_paths, labels, pdlabels = self.img_paths[item], self.img_labels[item] , self.pdlabels[item]
        inputs = Image.open(img_paths)#.convert('L')
        inputs = inputs.resize((224, 224))
        #inputs = inputs.convert('RGB')
        if self.transform is not None:
            inputs = self.transform(inputs)
            labels = int(labels)
            pdlabels = int(pdlabels)

        return inputs, labels, pdlabels, item

    def __len__(self):
        return self.n_data


# ═══════════════════════════════════════════════════════════════════════════
# XRF55 通用数据集构建器 (支持 cross_env 留一法)
# ═══════════════════════════════════════════════════════════════════════════

_XRF55_PATTERN = re.compile(r'^U(\d+)_G(\d+)_R(\d+)\.png$')
XRF55_TARGET_GESTURES = [44, 45, 46, 47, 48, 49, 50, 51]


class XRF55Dataset(data.Dataset):
    """XRF55 通用数据集，支持灵活过滤"""

    def __init__(self, data_dirs, transform=None,
                 allowed_users=None, allowed_gestures=None):
        """
        data_dirs: list of directories to scan
        allowed_users: set of user IDs (int) to include, or None for all
        allowed_gestures: list of gesture IDs (int), default XRF55_TARGET_GESTURES
        """
        self.transform = transform
        self.img_paths = []
        self.img_labels = []
        self.pdlabels = []

        if allowed_gestures is None:
            allowed_gestures = XRF55_TARGET_GESTURES
        gest_set = set(allowed_gestures)
        user_set = set(allowed_users) if allowed_users else None

        for data_dir in data_dirs:
            for f in os.listdir(data_dir):
                m = _XRF55_PATTERN.match(f)
                if m is None:
                    continue
                user_id = int(m.group(1))
                gesture_id = int(m.group(2))
                if gesture_id not in gest_set:
                    continue
                if user_set and user_id not in user_set:
                    continue
                self.img_paths.append(os.path.join(data_dir, f))
                self.img_labels.append(gesture_id - 44)
                self.pdlabels.append(0)

        self.n_data = len(self.img_paths)
        print(f'[XRF55Dataset] loaded {self.n_data} samples from {len(data_dirs)} dirs')

    def set_labels_by_index(self, tlabels=None, tindex=None, label_type='domain_label'):
        if label_type == 'pdlabel':
            self.pdlabels = np.array(self.pdlabels)
            self.pdlabels[tindex.astype(int)] = tlabels.astype(int)
            self.pdlabels = self.pdlabels.tolist()

    def __getitem__(self, item):
        img_path = self.img_paths[item]
        label = int(self.img_labels[item])
        pdlabel = int(self.pdlabels[item])
        img = Image.open(img_path)
        img = img.resize((224, 224))
        if self.transform is not None:
            img = self.transform(img)
        return img, label, pdlabel, item

    def __len__(self):
        return self.n_data

class datatecsie(data.Dataset):
    def __init__(self, data_list, transform=False):
        self.transform = transform
        self.img_paths = []
        self.img_labels = []
        self.pdlabels = []
        count = 0

        # Processed_Data format: U{user:02d}_G{gesture}_R{repetition:02d}.png
        # 动态扫描：测试集 User 25-30，Gesture 44-51
        all_files = [f for f in os.listdir(data_list) if f.endswith('.png')]
        for f in all_files:
            try:
                parts = f.replace('.png', '').split('_')
                user_id = int(parts[0][1:])   # 'U01' -> 1
                gesture_id = int(parts[1][1:])  # 'G44' -> 44
            except:
                continue
            if 25 <= user_id <= 30 and 44 <= gesture_id <= 51:
                self.img_paths.append(os.path.join(data_list, f))
                self.img_labels.append(gesture_id - 44)
                self.pdlabels.append(gesture_id - gesture_id)
                count += 1
        self.n_data = count
        print(count)

    def __getitem__(self, item):
        img_paths, labels, pdlabels = self.img_paths[item], self.img_labels[item] , self.pdlabels[item]
        inputs = Image.open(img_paths)#.convert('L')
        inputs = inputs.resize((224, 224))
        #inputs = inputs.convert('RGB')
        if self.transform is not None:
            inputs = self.transform(inputs)
            labels = int(labels)
            pdlabels = int(pdlabels)

        return inputs, labels, pdlabels, item

    def __len__(self):
        return self.n_data