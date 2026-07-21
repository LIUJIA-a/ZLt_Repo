import torch
import torch.optim as optim
from utils import trainer
from algorithm import *
import os
import argparse
import torchvision.transforms as transforms
import mytransforms
from model import *

def act_param_init(args):
    args.select_position = {'emg': [0]}
    args.select_channel = {'emg': np.arange(8)}
    args.hz_list = {'emg': 1000}
    args.act_people = {'emg': [[i*9+j for j in range(9)]for i in range(4)]}

    # 根据数据集自动设置类别数
    if args.dataset == 'widar':
        args.num_classes = 6   # G01-G06
    else:
        args.num_classes = 8   # XRF55: G44-G51

    return args

def get_args():
    parser = argparse.ArgumentParser(description='DG')
    parser.add_argument('--alpha', type=float,
                        default=0.1, help="DANN dis alpha")
    parser.add_argument('--alpha1', type=float,
                        default=0.1, help="DANN dis alpha")
    parser.add_argument('--batch_size', type=int,
                        default=32, help="batch_size")
    parser.add_argument('--beta1', type=float, default=0.5, help="Adam")
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--classifier', type=str,
                        default="linear", choices=["linear", "wn"])
    parser.add_argument('--dis_hidden', type=int, default=256)
    parser.add_argument('--gpu_id', type=str, nargs='?',
                        default='0', help="device id to run")
    parser.add_argument('--layer', type=str, default="bn",
                        choices=["ori", "bn"])
    parser.add_argument('--lam', type=float, default=0.0)
    parser.add_argument('--latent_domain_num', type=int, default=3)

    # ── 新增超参数 ──────────────────────────────────────────────────────────
    parser.add_argument('--supcon_tau', type=float, default=0.07,
                        help="温度系数 (τ) for SupConLoss")
    parser.add_argument('--proto_tau', type=float, default=0.1,
                        help="温度系数 (τ) for ProtoNCELoss")
    parser.add_argument('--hardnce_tau', type=float, default=0.07,
                        help="温度系数 (τ) for InfoNCE_HardNegative")
    parser.add_argument('--gamma', type=float, default=0.5,
                        help="阶段① SupCon 损失权重: cls_loss + gamma * supcon_loss")
    parser.add_argument('--lam_pcl', type=float, default=0.5,
                        help="阶段② 原型对比损失权重: disc_loss + lam_pcl * proto_loss")
    parser.add_argument('--lam_ent', type=float, default=0.3,
                        help="阶段② 域分配熵惩罚权重，防止隐式域坍塌")
    parser.add_argument('--beta', type=float, default=0.5,
                        help="阶段③ 镜像困难负样本损失权重: cls_loss + disc_loss + beta * hard_contrast_loss")
    parser.add_argument('--variance_percentile', type=float, default=30.0,
                        help="Physical_Mask_Augment: 方差低于此百分位的 patch 被选为掩码候选区")
    parser.add_argument('--mask_ratio', type=float, default=0.15,
                        help="Physical_Mask_Augment: 每个 patch 被掩码的概率")
    parser.add_argument('--patch_size', type=int, default=16,
                        help="Physical_Mask_Augment: 掩码 patch 大小 (像素)")

    # ── 数据集与实验模式 ────────────────────────────────────────────────────
    parser.add_argument('--dataset', type=str, default='xrf55',
                        choices=['xrf55', 'widar'],
                        help="数据集选择: xrf55 或 widar")
    parser.add_argument('--data_path', type=str, default='',
                        help="数据集目录路径")
    parser.add_argument('--experiment', type=str, default='cross_user',
                        choices=['in_domain', 'cross_user', 'cross_env',
                                 'cross_loc', 'cross_ori'],
                        help="实验类型 (仅 widar 生效)")

    # ── 消融实验模式 ────────────────────────────────────────────────────────
    parser.add_argument('--ablation', type=str, default='full',
                        choices=['M0', 'M1', 'M2', 'M3', 'M4', 'full'],
                        help="消融实验模式: M0=Baseline CE, M1=w/o SupCon, "
                             "M2=w/o PCL, M3=w/o GRL, M4=w/o HardNCE, full=完整模型")

    parser.add_argument('--local_epoch', type=int,
                        default=1, help='local iterations')
    parser.add_argument('--lr', type=float, default=0.0002, help="learning rate")
    parser.add_argument('--max_epoch', type=int,
                        default=50, help="max iterations")
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--exp_id', type=str, default='exp_1',
                        help="experiment identifier, used to create unique log directory")
    parser.add_argument('--train_envs', type=str, default=None,
                        help="comma-separated train envs for leave-one-out, e.g. E1,E2")
    parser.add_argument('--rx', type=str, default=None,
                        help="receiver filter, e.g. RX2 for single receiver")
    parser.add_argument('--data_fraction', type=float, default=1.0,
                        help="fraction of training data to use (0-1)")
    parser.add_argument('--direction_k', type=float, default=1.0,
                        help="Direction sensitivity threshold: k in tau = mean - k*std. "
                             "Higher k = fewer classes marked as direction-sensitive. "
                             "Set very high (e.g. 99) to disable direction detection entirely.")
    parser.add_argument('--test_scene', type=str, default='',
                        help="XRF55 cross_env 留一法: 指定测试场景编号如 1,2,3,4")
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                        help="Label smoothing factor for CE loss (0=off, 0.1=typical)")
    parser.add_argument('--xrf_test_users', type=int, default=None,
                        help="XRF55 cross_user LOUO: 指定单个测试用户ID(1-30),其余29人训练")
    parser.add_argument('--seed', type=int, default=None,
                        help="global random seed for reproducibility / paired runs; "
                             "None keeps legacy non-deterministic behavior")
    parser.add_argument('--dir_weight', type=float, default=0.5,
                        help="direction sensitivity weight for directional gestures "
                             "(circle=0; others=this value). For R2-2 sensitivity sweep.")
    args = parser.parse_args()
    args = act_param_init(args)
    return args

args = get_args()

# ── 全局随机种子设定（用于配对实验/复现）──────────────────────────────────
if args.seed is not None:
    import random as _random
    _random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    print(f'[Seed] global random seed set to {args.seed}')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = GeneFi(args).to(device)
params = net.parameters()
optimizer = optim.Adam(net.parameters(), lr=args.lr,weight_decay=args.weight_decay, betas=(args.beta1, 0.9))
milestones = [10,20]
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)
global_train_acc = []
global_test_acc = []
img_transform = transforms.Compose([
        transforms.RandomGrayscale(p=0.2),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomApply([transforms.RandomCrop(224, padding=(32,0),padding_mode='reflect')],p=0.2),
        mytransforms.RandomSpi(p=0.2),
        mytransforms.RandomComPre(p=0.2),
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
    ])
img_transformte = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
    ])

if __name__ == '__main__':
    log_dir = os.path.join("log", args.exp_id)
    os.makedirs(log_dir, exist_ok=True)
    trainer(net, img_transform, img_transformte, device, optimizer, scheduler,
            args.max_epoch, log_dir, args=args)