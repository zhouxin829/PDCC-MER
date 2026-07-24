import argparse
from src.pdcc_utils import *
from torch.utils.data import DataLoader
from src import (
    pdcc_train_label_smoothing_pretrain,
    pdcc_train_pseudo_pretrain,
    pdcc_train_run,
    train_pretrain,
)
from dataloader import MMDataset
import os


parser = argparse.ArgumentParser(description='PDCC-MER: Progressive Denoising and Calibrated Consensus Learning for Multimodal Emotion Recognition')
parser.add_argument('-f', default='', type=str)

# Tasks
parser.add_argument('--dataset', type=str, default='SIMS',
                    help='dataset to use (SIMS / SIMS-v2 / MOSI / MOSEI)')
parser.add_argument('--data_path', type=str, default='/data/Lab105/zhouxin/MSA_Datasets', # 数据集路径
                    help='path for storing the dataset')
parser.add_argument('--model_path', type=str, default='./savemodel', # 模型存放路径
                    help='path for storing the model')

parser.add_argument('--run_dir', type=str, default='',
                    help='One isolated experiment directory; contains pseudo labels, logs and metrics.')

parser.add_argument('--is_pseudo', action='store_true',
                    help='whether to use the pseudo labels (default: false)')
parser.add_argument('--pretrained_model', action='store_true',
                    help='whether to use the pretrained unimodal model (false -> train unimodal model ; true -> train PDCC model)')
parser.add_argument('--pretrain_temperature', type=float, default=0.07,
                    help='temperature in SupCon loss (default: 0.07)')

# Dropouts
parser.add_argument('--out_dropout', type=float, default=0.4,
                    help='output dropout (default: SIMS: 0.4 / SIMS-v2: 0.0)')

# Network
parser.add_argument('--transformer_layers', type=int, default=2,
                    help='transformer layers (default: 2)')
parser.add_argument('--nhead', type=int, default=4,
                    help='num heads (default: 4)')

# Tuning
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size (default: 64)')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate (default: 1e-3)')

parser.add_argument('--init_t_momentum', type=float, default=0.8,
                    help='initial momentum for text pseudo label update (default: 0.8)')
parser.add_argument('--init_a_momentum', type=float, default=0.99,
                    help='initial momentum for audio pseudo label update (default: 0.99)')
parser.add_argument('--init_v_momentum', type=float, default=0.99,
                    help='initial momentum for vision pseudo label update (default: 0.99)')

parser.add_argument('--t_momentum_gamma', type=float, default=0.5,
                    help='gamma for text pseudo label update (default: 0.5)')
parser.add_argument('--a_momentum_gamma', type=float, default=0.5,
                    help='gamma for audio pseudo label update (default: 0.5)')
parser.add_argument('--v_momentum_gamma', type=float, default=0.5,
                    help='gamma for vision pseudo label update (default: 0.5)')

parser.add_argument('--init_ema_t_model_momentum', type=float, default=0.8,
                    help='initial momentum for ema text model update (default: 0.8)')
parser.add_argument('--init_ema_a_model_momentum', type=float, default=0.9,
                    help='initial momentum for ema audio model update (default: 0.9)')
parser.add_argument('--init_ema_v_model_momentum', type=float, default=0.6,
                    help='initial momentum for ema vision model update (default: 0.6)')

parser.add_argument('--ema_t_model_momentum_gamma', type=float, default=2.5,
                    help='gamma for ema text model update (default: 2.5)')
parser.add_argument('--ema_a_model_momentum_gamma', type=float, default=5.0,
                    help='gamma for ema audio model update (default: 5.0)')
parser.add_argument('--ema_v_model_momentum_gamma', type=float, default=2.0,
                    help='gamma for ema vision model update (default: 2.0)')

parser.add_argument('--top_k', type=int, default=3,
                    help='top k experts (default: 3)')
parser.add_argument('--pcrp_temperature', type=float, default=0.07,
                    help='temperature used in PCRP (default: 0.07)')

# ===== PDCC-MER module settings =====
parser.add_argument('--use_tplr', action='store_true',
                    help='enable TPLR (Teacher-guided Progressive Label Refinement)')
parser.add_argument('--use_pcrp', action='store_true',
                    help='enable PCRP (Progressive Cross-modal Relation Propagation)')
parser.add_argument('--use_rccr', action='store_true',
                    help='enable RCCR (Reliability-Calibrated Consensus Routing)')
parser.add_argument('--use_label_smoothing_control', action='store_true',
                    help=('replace TPLR with fixed label-smoothing targets; '
                          'does not use an EMA teacher or pseudo-label history'))

# TPLR
parser.add_argument('--tplr_lambda', type=float, default=1.0,
                    help='refinement strength for TPLR')
parser.add_argument('--tplr_steps', type=int, default=400,
                    help='number of progressive refinement steps in TPLR')
parser.add_argument('--tplr_sigma', type=float, default=0.01,
                    help='noise scale used during TPLR refinement')
parser.add_argument('--label_smoothing_epsilon', type=float, default=0.1,
                    help=('fixed label-smoothing coefficient for the controlled '
                          'baseline; target=(1-epsilon)*one_hot+epsilon/C'))

# PCRP
parser.add_argument('--pcrp_steps', type=int, default=3,
                    help='propagation depth in PCRP')
parser.add_argument('--pcrp_strength', type=float, default=0.5,
                    help='residual propagation strength in PCRP (0~1)')

# RCCR
parser.add_argument('--rccr_tau', type=float, default=1.0,
                    help='temperature for reliability prior estimation in RCCR')
parser.add_argument('--rccr_lambda', type=float, default=0.1,
                    help='interpolation weight between router distribution and reliability prior in RCCR')

parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--num_epochs', type=int, default=100,
                    help='number of epochs (default: 100)')

# Logistics
parser.add_argument('--patience', type=int, default=10,
                    help='patience used for early stop (default: 10)')
parser.add_argument('--seed', type=str, default='random', help='random seed (int) or "random"')
parser.add_argument('--no_cuda', action='store_true',
                    help='do not use cuda (default: false)')
parser.add_argument('--language', type=str, default='cn',
                    help='bert language (default: cn)')
parser.add_argument('--finetune', action='store_true',
                    help='whether to finetune the BERT (default: false)')

parser.add_argument("--use_best", action="store_true",
                    help="override args with tuned best config for the dataset")

# ===== Robustness re-training settings =====
# These options are consumed by dataloader.MMDataset. They do NOT alter --use_best
# model/module hyperparameters; they only define the input availability condition.
parser.add_argument("--robust_mode", type=str, default="clean",
                    choices=["clean", "missing", "noise", "random_missing", "misalign"],
                    help=("input condition: clean, fixed missing modality, noise, "
                          "sample-random missing modality, or cross-sample misalignment"))
parser.add_argument("--robust_modality", type=str, default="none",
                    choices=["none", "text", "audio", "vision"],
                    help="modality altered when --robust_mode is not clean")
parser.add_argument("--robust_level", type=float, default=0.0,
                    help=("noise scale, or affected-sample probability for "
                          "random_missing/misalign; ignored by clean/missing"))
parser.add_argument("--robust_scope", type=str, default="all",
                    choices=["all", "test_only"],
                    help="apply condition to train+valid+test (all) or only test split")
parser.add_argument("--robust_seed", type=int, default=20260707,
                    help="fixed seed defining the deterministic corrupted dataset")
parser.add_argument("--num_workers", type=int, default=14,
                    help="DataLoader worker count (default: 14)")





args = parser.parse_args()

if args.use_tplr and args.use_label_smoothing_control:
    raise ValueError(
        "--use_tplr and --use_label_smoothing_control are mutually exclusive"
    )
if args.use_label_smoothing_control and not args.is_pseudo:
    raise ValueError(
        "--use_label_smoothing_control requires --is_pseudo so the same fixed "
        "targets can be used in both stages"
    )
if not 0.0 <= float(args.label_smoothing_epsilon) < 1.0:
    raise ValueError("--label_smoothing_epsilon must be in [0, 1)")

# ===== Apply best tuned config (optional) =====
if getattr(args, "use_best", False):
    import pdcc_best_config

    cfg = pdcc_best_config.BEST_CONFIGS.get(getattr(args, "dataset", None), None)
    if cfg is None:
        raise ValueError(f"[BEST] No best config for dataset={getattr(args, 'dataset', None)}")

    for k, v in cfg.items():
        setattr(args, k, v)

    print("\n[BEST] Overriding args with tuned best config:")
    for k, v in cfg.items():
        print(f"  {k} = {v}")
    print("")




# === Resolve and set random seed ===
def _resolve_seed(seed_arg):
    try:
        return int(seed_arg)
    except Exception:
        pass
    if isinstance(seed_arg, str) and seed_arg.lower() == "random":
        import os
        return int.from_bytes(os.urandom(8), "little") % (2**32 - 1)
    import os
    return int.from_bytes(os.urandom(8), "little") % (2**32 - 1)

args.seed = _resolve_seed(getattr(args, "seed", "random"))

import json
from pathlib import Path

args.run_dir = str(Path(args.run_dir or os.getcwd()).resolve())
Path(args.run_dir).mkdir(parents=True, exist_ok=True)
args.pseudo_dir = str(Path(args.run_dir) / 'pseudo_labels')
Path(args.pseudo_dir).mkdir(parents=True, exist_ok=True)

# The multi-seed launcher passes --model_path <run_dir>/models.
# Normalize it here so Stage 1 and Stage 2 use exactly the same checkpoint directory.
args.model_path = str(Path(args.model_path).resolve())
Path(args.model_path).mkdir(parents=True, exist_ok=True)

def _json_safe(x):
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    if hasattr(x, 'item'):
        try:
            return x.item()
        except Exception:
            pass
    return x

def _save_stage_result(stage, ans):
    method = 'PDCC' if innovation_enabled(args) else 'BASE'
    payload = {
        'meta': {
            'dataset': args.dataset,
            'method': method,
            'seed': int(args.seed),
            'stage': stage,
            'model_path': str(Path(args.model_path).resolve()),
            'run_dir': args.run_dir,
            'robust_mode': getattr(args, 'robust_mode', 'clean'),
            'robust_modality': getattr(args, 'robust_modality', 'none'),
            'robust_level': float(getattr(args, 'robust_level', 0.0)),
            'robust_scope': getattr(args, 'robust_scope', 'all'),
            'robust_seed': int(getattr(args, 'robust_seed', 20260707)),
            'use_label_smoothing_control': bool(
                getattr(args, 'use_label_smoothing_control', False)
            ),
            'label_smoothing_epsilon': float(
                getattr(args, 'label_smoothing_epsilon', 0.1)
            ),
        },
        'metrics': _json_safe(ans),
    }
    filename = 'metrics.json' if stage == 'stage2' else 'stage1_summary.json'
    with open(Path(args.run_dir) / filename, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)

# Make runs deterministic given the chosen seed
import random as _random
import numpy as _np
import torch as _torch

_random.seed(args.seed)
_np.random.seed(args.seed)
_torch.manual_seed(args.seed)
if _torch.cuda.is_available():
    _torch.cuda.manual_seed_all(args.seed)
_torch.backends.cudnn.deterministic = True
_torch.backends.cudnn.benchmark = False
print(f"[Seed] Using seed: {args.seed}")

seed_everything(args)

use_cuda = False

torch.set_default_dtype(torch.float32)
if torch.cuda.is_available():
    if args.no_cuda:
        print("WARNING: You have a CUDA device, so you should probably not run with --no_cuda")
        device = torch.device('cpu')
    else:
        seed_everything(args)
        use_cuda = True
        device = torch.device('cuda')
else:
    device = torch.device('cpu')

####################################################################
#
# Load the dataset (aligned or non-aligned)
#
####################################################################

print("Start loading the data....")

args.weight_decay_bert = 0.01
args.lr_bert = 2e-5

if args.robust_mode == "clean":
    args.robust_modality = "none"
    args.robust_level = 0.0
elif args.robust_modality == "none":
    raise ValueError("--robust_modality must be text/audio/vision when robust_mode is not clean")
elif args.robust_mode in {"noise", "random_missing", "misalign"} and not (
    0.0 < float(args.robust_level) <= 1.0
):
    raise ValueError(
        "--robust_level must be in (0, 1] for noise, random_missing, and misalign"
    )
print(
    "[Robustness] "
    f"mode={args.robust_mode}, modality={args.robust_modality}, "
    f"level={args.robust_level}, scope={args.robust_scope}, seed={args.robust_seed}"
)
train_data = MMDataset(args, 'train')
valid_data = MMDataset(args, 'valid')
test_data = MMDataset(args, 'test')
if 'SIMS' in args.dataset:
    D_msc = MMDataset(args, 'test', 'D_msc')
    D_msi = MMDataset(args, 'test', 'D_msi')

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# Bind the shuffled train order (and worker base seeds) to the experiment seed.
train_generator = torch.Generator()
train_generator.manual_seed(args.seed)

dataloaders = {
    'train': DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=train_generator,
    ),
    'valid': DataLoader(valid_data, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, worker_init_fn=seed_worker),
    'test': DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, worker_init_fn=seed_worker)
}
if 'SIMS' in args.dataset:
    new_dataloaders = {
        'D_msc': DataLoader(D_msc, batch_size=args.batch_size, shuffle=False,
                                   num_workers=args.num_workers, worker_init_fn=seed_worker),
        'D_msi': DataLoader(D_msi, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, worker_init_fn=seed_worker)
    }
    dataloaders.update(new_dataloaders)

print('Finish loading the data....')
print(f'### Dataset - {args.dataset}')

####################################################################
#
# Hyperparameters
#
####################################################################

hyp_params = args

# Fix feature dimensions and sequence length
hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v = train_data.get_feature_dim()
hyp_params.l_len, hyp_params.a_len, hyp_params.v_len = train_data.get_seq_len()

hyp_params.use_cuda = use_cuda
hyp_params.n_train, hyp_params.n_valid, hyp_params.n_test = len(train_data), len(valid_data), len(test_data)
hyp_params.embed_dim = 32
hyp_params.output_dim = 3
hyp_params.criterion = 'CrossEntropyLoss'
hyp_params.eval_metric = 'Has0_acc_2'
hyp_params.language = 'cn' if 'SIMS' in hyp_params.dataset else 'en'

if __name__ == '__main__':
    if not hyp_params.pretrained_model:
        print('Start Pre-Training....')
        if hyp_params.is_pseudo:
            if hyp_params.use_label_smoothing_control:
                print('Use fixed label-smoothing targets to pre-train....')
                ans = pdcc_train_label_smoothing_pretrain.initiate(
                    hyp_params, dataloaders, device
                )
            else:
                print('Use Pseudo Labels to Pre-training....')
                ans = pdcc_train_pseudo_pretrain.initiate(hyp_params, dataloaders, device)
        else:
            ans = train_pretrain.initiate(hyp_params, dataloaders, device)
        for key, value in ans.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for k2, v2 in value.items():
                    try:
                        print(f"  {k2:20s}: {v2:.4f}")
                    except Exception:
                        print(f"  {k2:20s}: {v2}")
            else:
                print(f"{key}: {value:.4f}")
        _save_stage_result('stage1', ans)
        print(f"[Run] Saved Stage-1 summary: {Path(args.run_dir) / 'stage1_summary.json'}")

    else:
        print('Start Training....')
        prefix = 'Pseudo_' if hyp_params.is_pseudo else ''
        pretrained_t_model = resolve_existing_model_path(
            hyp_params, names={'model_name': f'{prefix}TextModel'}
        )
        pretrained_a_model = resolve_existing_model_path(
            hyp_params, names={'model_name': f'{prefix}AudioModel'}
        )
        pretrained_v_model = resolve_existing_model_path(
            hyp_params, names={'model_name': f'{prefix}VisionModel'}
        )

        # Baseline and PDCC artifacts are both resolved here.
        if not os.path.exists(pretrained_t_model) or not os.path.exists(pretrained_a_model) or not os.path.exists(pretrained_v_model):
            pretrained_models = None
            print('WARNING: There are no pretrained models!')
        else:
            pretrained_models = (pretrained_t_model, pretrained_a_model, pretrained_v_model)
        ans = pdcc_train_run.initiate(hyp_params, dataloaders, pretrained_models, device)
        for key, value in ans.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for k2, v2 in value.items():
                    try:
                        print(f"  {k2:20s}: {v2:.4f}")
                    except Exception:
                        print(f"  {k2:20s}: {v2}")
            else:
                print(f"{key}: {value:.4f}")
        _save_stage_result('stage2', ans)
        print(f"[Run] Saved Stage-2 metrics: {Path(args.run_dir) / 'metrics.json'}")
