# src/utils.py
import torch
import os
import random
import numpy as np






# =========================================================
# innovation switch
# =========================================================
def innovation_enabled(args) -> bool:
    flags = [
        getattr(args, "use_tplr", False),
        getattr(args, "use_pcrp", False),
        getattr(args, "use_rccr", False),
        getattr(args, "use_label_smoothing_control", False),
    ]
    return any(flags)

def save_load_name(args, names):
    model_name = names['model_name']
    tag = "PDCC" if innovation_enabled(args) else "BASE"
    return f"{args.dataset}_{tag}_{model_name}"

def pseudo_tag(args) -> str:
    return "PDCC" if innovation_enabled(args) else "BASE"


def get_pseudo_label_filename(args, split: str) -> str:
    # split: train / valid / test
    tag = pseudo_tag(args)
    return f"{args.dataset}_{tag}_{split}_pseudo_labels.pkl"


def get_pseudo_label_path(args, split: str) -> str:
    pseudo_dir = resolve_pseudo_dir(args)
    return os.path.join(pseudo_dir, get_pseudo_label_filename(args, split))


def resolve_existing_pseudo_label_path(args, split: str) -> str:
    canonical = get_pseudo_label_path(args, split)
    if os.path.exists(canonical) or pseudo_tag(args) == "BASE":
        return canonical
    legacy = os.path.join(
        resolve_pseudo_dir(args),
        f"{args.dataset}_OURS_{split}_pseudo_labels.pkl",
    )
    if os.path.exists(legacy):
        print(f"[Compatibility] Using legacy pseudo labels: {legacy}")
        return legacy
    return canonical


def get_all_pseudo_label_paths(args, existing: bool = False):
    resolver = (
        resolve_existing_pseudo_label_path
        if existing
        else get_pseudo_label_path
    )
    return {
        "train": resolver(args, "train"),
        "valid": resolver(args, "valid"),
        "test": resolver(args, "test"),
    }

def _to_history_dir(path: str, history_basename: str) -> str:
    """
    Convert a base dir (e.g., .../savemodel) into .../savemodel_history
    in a robust way. If basename doesn't match, append '_history'.
    """
    path = os.path.normpath(path)
    parent = os.path.dirname(path)
    base = os.path.basename(path)

    # Common case: user uses ".../savemodel"
    if base == "savemodel" and history_basename == "savemodel_history":
        return os.path.join(parent, "savemodel_history")

    # Fallback: append suffix
    if not base.endswith("_history"):
        base = base + "_history"
    return os.path.join(parent, base)


def resolve_model_dir(args) -> str:
    return args.model_path



def resolve_pseudo_dir(args) -> str:
    """Return the pseudo-label directory for the current isolated run."""
    explicit = getattr(args, "pseudo_dir", "")
    if explicit:
        return os.path.abspath(explicit)

    run_dir = getattr(args, "run_dir", "")
    root = os.path.abspath(run_dir) if run_dir else os.getcwd()
    return os.path.join(root, "pseudo_labels")



# =========================================================
# Save / Load
# =========================================================
def save_model(args, model, names):
    name = save_load_name(args, names)
    model_dir = resolve_model_dir(args)
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model, f'{model_dir}/{name}.pt')
    print(f"Saved model at {model_dir}/{name}.pt!")


def model_path_candidates(args, names):
    """Return the canonical path followed by pre-rename artifact aliases."""
    model_dir = resolve_model_dir(args)
    canonical_name = save_load_name(args, names)
    tags = ["PDCC", "OURS"] if innovation_enabled(args) else ["BASE"]
    model_names = [names["model_name"]]
    if "PDCCModel" in names["model_name"]:
        model_names.append(names["model_name"].replace("PDCCModel", "DCCModel"))

    candidates = [os.path.join(model_dir, f"{canonical_name}.pt")]
    for tag in tags:
        for model_name in model_names:
            candidate = os.path.join(
                model_dir,
                f"{args.dataset}_{tag}_{model_name}.pt",
            )
            if candidate not in candidates:
                candidates.append(candidate)
    return candidates


def resolve_existing_model_path(args, names):
    candidates = model_path_candidates(args, names)
    for candidate in candidates:
        if os.path.exists(candidate):
            if candidate != candidates[0]:
                print(f"[Compatibility] Using legacy checkpoint: {candidate}")
            return candidate
    return candidates[0]


def load_model(args, names):
    path = resolve_existing_model_path(args, names)
    print(f"Loading model at {path}!")
    model = torch.load(path, weights_only=False)
    return model


def seed_everything(args):
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)

    torch.manual_seed(args.seed)
    if not args.no_cuda:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    torch.use_deterministic_algorithms(True)

def transfer_models(new_model, pretrained_models):
    pretrained_t_model, pretrained_a_model, pretrained_v_model = pretrained_models
    new_dict = new_model.state_dict()

    t_model = torch.load(pretrained_t_model, map_location=torch.device('cuda'), weights_only=False)
    pretrain_t_dict = t_model.state_dict()
    t_proj_state_dict = {}
    t_enc_state_dict = {}
    for k, v in pretrain_t_dict.items():
        if k in [
            "proj1.weight",
            "proj1.bias",
            "proj2.weight",
            "proj2.bias",
            "out_layer.weight",
            "out_layer.bias"
        ]:
            k_list = k.split('.')
            k_list[0] = k_list[0] + 's.0'
            new_k = '.'.join(k_list)
            t_proj_state_dict[new_k] = v
        else:
            t_enc_state_dict[k] = v
    new_dict.update(t_proj_state_dict)
    new_dict.update(t_enc_state_dict)

    a_model = torch.load(pretrained_a_model, map_location=torch.device('cuda'), weights_only=False)
    pretrain_a_dict = a_model.state_dict()
    a_proj_state_dict = {}
    a_enc_state_dict = {}
    for k, v in pretrain_a_dict.items():
        if k in [
            "proj1.weight",
            "proj1.bias",
            "proj2.weight",
            "proj2.bias",
            "out_layer.weight",
            "out_layer.bias"
        ]:
            k_list = k.split('.')
            k_list[0] = k_list[0] + 's.1'
            new_k = '.'.join(k_list)
            a_proj_state_dict[new_k] = v
        else:
            a_enc_state_dict[k] = v
    new_dict.update(a_proj_state_dict)
    new_dict.update(a_enc_state_dict)

    v_model = torch.load(pretrained_v_model, map_location=torch.device('cuda'), weights_only=False)
    pretrain_v_dict = v_model.state_dict()
    v_proj_state_dict = {}
    v_enc_state_dict = {}
    for k, v in pretrain_v_dict.items():
        if k in [
            "proj1.weight",
            "proj1.bias",
            "proj2.weight",
            "proj2.bias",
            "out_layer.weight",
            "out_layer.bias"
        ]:
            k_list = k.split('.')
            k_list[0] = k_list[0] + 's.2'
            new_k = '.'.join(k_list)
            v_proj_state_dict[new_k] = v
        else:
            v_enc_state_dict[k] = v
    new_dict.update(v_proj_state_dict)
    new_dict.update(v_enc_state_dict)

    new_model.load_state_dict(new_dict)
    return new_model
