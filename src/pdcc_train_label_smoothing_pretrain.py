"""Stage-1 control using fixed label-smoothing targets instead of TPLR.

This path intentionally has no EMA teacher, progressive perturbation, iterative
refinement, or historical pseudo-label memory.  A static target derived from
the multimodal label supervises all three unimodal encoders:

    q = (1 - epsilon) * one_hot(y) + epsilon / C

The same fixed distributions are exported for Stage 2, where they remain soft
targets for the three auxiliary unimodal heads.
"""
from __future__ import annotations

import os
import pickle
import time

import torch
import torch.nn.functional as F
import torch.optim as optim

from src import pdcc_models
from src.pdcc_utils import get_all_pseudo_label_paths, load_model, save_model
from src.eval_metrics import eval_mosi_classification, eval_sims_classification


def fixed_label_smoothing_targets(labels, epsilon, num_classes=3):
    labels = labels.view(-1).long()
    one_hot = F.one_hot(labels, num_classes=num_classes).float()
    return (1.0 - epsilon) * one_hot + epsilon / float(num_classes)


def soft_target_cross_entropy(logits, targets):
    targets = targets.to(device=logits.device, dtype=logits.dtype)
    return -(targets * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


def _binary_contrastive_labels(labels, dataset):
    if "SIMS" in dataset:
        return (labels > 1).float()
    return (labels >= 1).float()


def _classification_metrics(logits, labels, dataset):
    if "SIMS" in dataset:
        return eval_sims_classification(logits, labels)
    return eval_mosi_classification(logits, labels)


def initiate(hyp_params, dataloaders, device):
    epsilon = float(hyp_params.label_smoothing_epsilon)
    if not 0.0 <= epsilon < 1.0:
        raise ValueError("label_smoothing_epsilon must be in [0, 1)")

    t_model = pdcc_models.TextModel(hyp_params)
    a_model = pdcc_models.AudioModel(hyp_params)
    v_model = pdcc_models.VisionModel(hyp_params)

    if hyp_params.use_cuda:
        t_model = t_model.to(device)
        a_model = a_model.to(device)
        v_model = v_model.to(device)

    bert_no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    bert_params = list(t_model.text_model.named_parameters())
    bert_params_decay = [
        p for n, p in bert_params if not any(nd in n for nd in bert_no_decay)
    ]
    bert_params_no_decay = [
        p for n, p in bert_params if any(nd in n for nd in bert_no_decay)
    ]
    text_other = [
        p for n, p in t_model.named_parameters() if "text_model" not in n
    ]
    t_optimizer = optim.Adam(
        [
            {
                "params": bert_params_decay,
                "weight_decay": hyp_params.weight_decay_bert,
                "lr": hyp_params.lr_bert,
            },
            {
                "params": bert_params_no_decay,
                "weight_decay": 0.0,
                "lr": hyp_params.lr_bert,
            },
            {"params": text_other, "weight_decay": 0.0, "lr": hyp_params.lr},
        ]
    )
    a_optimizer = optim.Adam(a_model.parameters())
    v_optimizer = optim.Adam(v_model.parameters())
    contrastive_criterion = pdcc_models.SupConLoss(
        temperature=hyp_params.pretrain_temperature
    )

    print(
        "[LabelSmoothingControl] "
        f"epsilon={epsilon:g}, target=(1-epsilon)*one_hot+epsilon/3, "
        "EMA_teacher=False, progressive_refinement=False, history_memory=False"
    )

    settings = {
        "models": (t_model, a_model, v_model),
        "optimizers": (t_optimizer, a_optimizer, v_optimizer),
        "contrastive_criterion": contrastive_criterion,
    }
    return train_model(settings, hyp_params, dataloaders, device)


def train_model(settings, hyp_params, dataloaders, device):
    t_model, a_model, v_model = settings["models"]
    t_optimizer, a_optimizer, v_optimizer = settings["optimizers"]
    contrastive_criterion = settings["contrastive_criterion"]
    epsilon = float(hyp_params.label_smoothing_epsilon)

    def run_split(split, training):
        loader = dataloaders[split]
        models = (t_model, a_model, v_model)
        optimizers = (t_optimizer, a_optimizer, v_optimizer)
        for model in models:
            model.train(training)

        totals = {"T": 0.0, "A": 0.0, "V": 0.0, "C": 0.0}
        results = {"T": [], "A": [], "V": []}
        truths = {"T": [], "A": [], "V": []}

        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for batch_data in loader:
                text = batch_data["text"].to(device)
                audio = batch_data["audio"].to(device)
                vision = batch_data["vision"].to(device)
                labels_m = batch_data["labels"]["M"].to(device)
                labels_t = batch_data["labels"]["T"].to(device)
                labels_a = batch_data["labels"]["A"].to(device)
                labels_v = batch_data["labels"]["V"].to(device)
                batch_size = text.size(0)

                if training:
                    for optimizer in optimizers:
                        optimizer.zero_grad()

                targets = fixed_label_smoothing_targets(
                    labels_m, epsilon, hyp_params.output_dim
                ).to(device)
                t_outputs = t_model(text)
                a_outputs = a_model(audio)
                v_outputs = v_model(vision)

                loss_t = soft_target_cross_entropy(t_outputs["pred"], targets)
                loss_a = soft_target_cross_entropy(a_outputs["pred"], targets)
                loss_v = soft_target_cross_entropy(v_outputs["pred"], targets)

                h_tav = torch.cat(
                    (t_outputs["h_l"], a_outputs["h_a"], v_outputs["h_v"]),
                    dim=0,
                )
                hard_shared = labels_m.view(-1, 1).long()
                labels_tav = torch.cat(
                    (hard_shared, hard_shared, hard_shared), dim=0
                )
                labels_tav = _binary_contrastive_labels(
                    labels_tav, hyp_params.dataset
                )
                loss_c = contrastive_criterion(h_tav, labels_tav)

                if training:
                    (loss_t + loss_a + loss_v + loss_c).backward()
                    for optimizer in optimizers:
                        optimizer.step()

                totals["T"] += loss_t.item() * batch_size
                totals["A"] += loss_a.item() * batch_size
                totals["V"] += loss_v.item() * batch_size
                totals["C"] += loss_c.item() * batch_size
                results["T"].append(t_outputs["pred"].detach().cpu())
                results["A"].append(a_outputs["pred"].detach().cpu())
                results["V"].append(v_outputs["pred"].detach().cpu())
                truths["T"].append(labels_t.detach().cpu())
                truths["A"].append(labels_a.detach().cpu())
                truths["V"].append(labels_v.detach().cpu())

        denominator = len(loader.dataset)
        losses = tuple(totals[key] / max(1, denominator) for key in ("T", "A", "V", "C"))
        for modality in ("T", "A", "V"):
            results[modality] = torch.cat(results[modality])
            truths[modality] = torch.cat(truths[modality])
        return losses, results, truths

    text_parameters = sum(param.nelement() for param in t_model.parameters())
    audio_parameters = sum(param.nelement() for param in a_model.parameters())
    vision_parameters = sum(param.nelement() for param in v_model.parameters())
    print(
        "Stage-1 trainable parameters: "
        f"text={text_parameters}, audio={audio_parameters}, "
        f"vision={vision_parameters}, total="
        f"{text_parameters + audio_parameters + vision_parameters}"
    )

    valid_best = {"T": float("inf"), "A": float("inf"), "V": float("inf")}
    curr_patience = hyp_params.patience

    for epoch in range(1, hyp_params.num_epochs + 1):
        start = time.time()
        train_losses, _, _ = run_split("train", training=True)
        valid_losses, _, _ = run_split("valid", training=False)
        elapsed = time.time() - start

        print("-" * 50)
        print(
            f"Epoch {epoch:3d} | Time {elapsed:7.2f} sec | "
            f"Train T/A/V/C={train_losses[0]:.4f}/{train_losses[1]:.4f}/"
            f"{train_losses[2]:.4f}/{train_losses[3]:.4f} | "
            f"Valid T/A/V/C={valid_losses[0]:.4f}/{valid_losses[1]:.4f}/"
            f"{valid_losses[2]:.4f}/{valid_losses[3]:.4f}"
        )

        improved = False
        for modality, model, loss in zip(
            ("T", "A", "V"), (t_model, a_model, v_model), valid_losses[:3]
        ):
            if loss < valid_best[modality]:
                valid_best[modality] = float(loss)
                save_model(
                    hyp_params,
                    model,
                    names={"model_name": f"Pseudo_{'Text' if modality == 'T' else 'Audio' if modality == 'A' else 'Vision'}Model"},
                )
                improved = True

        if improved:
            curr_patience = hyp_params.patience
        else:
            curr_patience -= 1
        if curr_patience <= 0:
            print("Early stopping triggered.")
            break

    best_models = (
        load_model(hyp_params, names={"model_name": "Pseudo_TextModel"}),
        load_model(hyp_params, names={"model_name": "Pseudo_AudioModel"}),
        load_model(hyp_params, names={"model_name": "Pseudo_VisionModel"}),
    )
    t_model, a_model, v_model = best_models
    if hyp_params.use_cuda:
        t_model = t_model.to(device)
        a_model = a_model.to(device)
        v_model = v_model.to(device)

    _, test_results, test_truths = run_split("test", training=False)
    answer = {"label_smoothing_epsilon": epsilon}
    for modality in ("T", "A", "V"):
        metrics = _classification_metrics(
            test_results[modality], test_truths[modality], hyp_params.dataset
        )
        answer[modality] = float(metrics["Has0_acc_2"])

    _save_static_targets(hyp_params, dataloaders, epsilon)
    return answer


def _save_static_targets(hyp_params, dataloaders, epsilon):
    """Export fixed targets once; this is not an epoch-to-epoch memory bank."""
    paths = get_all_pseudo_label_paths(hyp_params)
    for split in ("train", "valid", "test"):
        bank = {}
        for batch_data in dataloaders[split]:
            targets = fixed_label_smoothing_targets(
                batch_data["labels"]["M"], epsilon, hyp_params.output_dim
            ).cpu()
            for index, sample_id in enumerate(batch_data["id"]):
                target = targets[index].clone()
                bank[sample_id] = {
                    "T": target.clone(),
                    "A": target.clone(),
                    "V": target.clone(),
                }

        path = paths[split]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as handle:
            pickle.dump(bank, handle)
        print(f"[LabelSmoothingControl] Saved static {split} targets -> {path}")
