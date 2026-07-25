# src/train_run.py
import os
import pickle
import time
import random
import numpy as np

import torch
from torch import nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from src import pdcc_models
from src.pdcc_utils import *
from src.eval_metrics import *
from src.pdcc_utils import resolve_pseudo_dir  # 文件顶部确保有这个 import
from src.eval_regression_metrics import eval_corr_mae_from_logits3


def soft_target_cross_entropy(logits, targets):
    """Cross entropy for a fixed probability target distribution."""
    targets = targets.to(device=logits.device, dtype=logits.dtype)
    return -(targets * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


####################################################################
# Construct the model
####################################################################

def initiate(hyp_params, dataloaders, pretrained_models, device):
    model = pdcc_models.PDCCModel(hyp_params)
    if pretrained_models is not None:
        model = transfer_models(model, pretrained_models)

    if hyp_params.use_cuda:
        model = model.to(device)

    # BERT参数分组（no_decay）
    bert_no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    bert_params = list(model.text_model.named_parameters())
    bert_params_decay = [p for n, p in bert_params if not any(nd in n for nd in bert_no_decay)]
    bert_params_no_decay = [p for n, p in bert_params if any(nd in n for nd in bert_no_decay)]
    model_params_other = [p for n, p in list(model.named_parameters()) if 'text_model' not in n]

    # 损失（任务损失 与 对比损失）
    task_criterion = getattr(nn, hyp_params.criterion)()
    contrastive_criterion = pdcc_models.SupConLoss(temperature=hyp_params.pretrain_temperature)

    optimizer_grouped_parameters = [
        {'params': bert_params_decay, 'weight_decay': hyp_params.weight_decay_bert, 'lr': hyp_params.lr_bert},
        {'params': bert_params_no_decay, 'weight_decay': 0.0, 'lr': hyp_params.lr_bert},
        {'params': model_params_other, 'weight_decay': 0.0, 'lr': hyp_params.lr}
    ]
    # 使用 AdamW（已在 param groups 指定 weight_decay）
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=hyp_params.lr)

    settings = {
        'model': model,
        'optimizer': optimizer,
        'task_criterion': task_criterion,
        'contrastive_criterion': contrastive_criterion
    }

    return train_model(settings, hyp_params, dataloaders, device)


####################################################################
# Training and evaluation scripts
####################################################################

def train_model(settings, hyp_params, dataloaders, device):
    model = settings['model']
    optimizer = settings['optimizer']
    task_criterion = settings['task_criterion']
    contrastive_criterion = settings['contrastive_criterion']

    # scheduler：按 epoch 调度，T_max 使用 num_epochs
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, hyp_params.num_epochs), eta_min=1e-6)

    grad_clip_value = 1.0

    # 伪标签加载（仅在启用时）
    if hyp_params.is_pseudo:
        pseudo_paths = get_all_pseudo_label_paths(hyp_params, existing=True)

        for split, path in pseudo_paths.items():
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"[PseudoLabel] Missing {split} pseudo label file: {path}\n"
                    f"Please run pseudo pretraining first under the SAME BASE/PDCC setting."
                )

        with open(pseudo_paths["train"], 'rb') as f:
            train_pseudo_labels = pickle.load(f)
        with open(pseudo_paths["valid"], 'rb') as f:
            valid_pseudo_labels = pickle.load(f)
        with open(pseudo_paths["test"], 'rb') as f:
            test_pseudo_labels = pickle.load(f)

        print("[PseudoLabel] Loaded:")
        print(f"  train -> {pseudo_paths['train']}")
        print(f"  valid -> {pseudo_paths['valid']}")
        print(f"  test  -> {pseudo_paths['test']}")
    else:
        train_pseudo_labels = valid_pseudo_labels = test_pseudo_labels = None

    def train_one_epoch(model, optimizer, task_criterion, contrastive_criterion):
        epoch_loss = 0.0
        model.train()

        for batch_data in dataloaders['train']:
            text = batch_data['text'].to(device)
            audio = batch_data['audio'].to(device)
            vision = batch_data['vision'].to(device)
            labels = batch_data['labels']['M'].to(device)

            if not hyp_params.is_pseudo:
                labels_t = batch_data['labels']['T'].to(device)
                labels_a = batch_data['labels']['A'].to(device)
                labels_v = batch_data['labels']['V'].to(device)
            else:
                video_id = batch_data['id']
                pseudo_labels_t, pseudo_labels_a, pseudo_labels_v = [], [], []
                for i in range(len(video_id)):
                    pseudo_labels_t.append(train_pseudo_labels[video_id[i]]['T'].unsqueeze(0))
                    pseudo_labels_a.append(train_pseudo_labels[video_id[i]]['A'].unsqueeze(0))
                    pseudo_labels_v.append(train_pseudo_labels[video_id[i]]['V'].unsqueeze(0))
                pseudo_labels_t = torch.cat(pseudo_labels_t).to(device)
                pseudo_labels_a = torch.cat(pseudo_labels_a).to(device)
                pseudo_labels_v = torch.cat(pseudo_labels_v).to(device)
                labels_t = torch.argmax(pseudo_labels_t, dim=-1, keepdim=True)
                labels_a = torch.argmax(pseudo_labels_a, dim=-1, keepdim=True)
                labels_v = torch.argmax(pseudo_labels_v, dim=-1, keepdim=True)

            batch_size = text.size(0)
            optimizer.zero_grad()

            outputs = model(text, audio, vision)
            loss_m = task_criterion(outputs['pred'], labels.view(-1).long())
            if getattr(hyp_params, 'use_label_smoothing_control', False):
                loss_t = soft_target_cross_entropy(outputs['pred_t'], pseudo_labels_t)
                loss_a = soft_target_cross_entropy(outputs['pred_a'], pseudo_labels_a)
                loss_v = soft_target_cross_entropy(outputs['pred_v'], pseudo_labels_v)
            else:
                loss_t = task_criterion(outputs['pred_t'], labels_t.view(-1).long())
                loss_a = task_criterion(outputs['pred_a'], labels_a.view(-1).long())
                loss_v = task_criterion(outputs['pred_v'], labels_v.view(-1).long())
            loss_task = loss_m + loss_t + loss_a + loss_v

            h_tav = torch.cat((outputs['h_l'], outputs['h_a'], outputs['h_v']), dim=0)
            labels_tav = torch.cat((labels_t, labels_a, labels_v), dim=0)
            if 'SIMS' in hyp_params.dataset:
                labels_tav = (labels_tav > 1).float()
            else:
                labels_tav = (labels_tav >= 1).float()
            loss_c = contrastive_criterion(h_tav, labels_tav)

            loss = loss_task + loss_c
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
            optimizer.step()

            epoch_loss += loss.item() * batch_size

        # scheduler 按 epoch 更新
        scheduler.step()

        epoch_loss /= max(1, hyp_params.n_train)
        return epoch_loss

    def evaluate(model, task_criterion, contrastive_criterion, test=False, test_mode=None):
        model.eval()
        loader = dataloaders['valid'] if not test else dataloaders[test_mode]

        epoch_loss = 0.0
        results = []
        truths = []
        reg_truths = []  # === 新增：回归真值收集 ===

        with torch.no_grad():
            for batch_data in loader:
                text = batch_data['text'].to(device)
                audio = batch_data['audio'].to(device)
                vision = batch_data['vision'].to(device)
                labels = batch_data['labels']['M'].to(device)
                if not hyp_params.is_pseudo:
                    labels_t = batch_data['labels']['T'].to(device)
                    labels_a = batch_data['labels']['A'].to(device)
                    labels_v = batch_data['labels']['V'].to(device)
                else:
                    video_id = batch_data['id']
                    pseudo_src = valid_pseudo_labels if not test else test_pseudo_labels
                    pseudo_labels_t, pseudo_labels_a, pseudo_labels_v = [], [], []
                    for i in range(len(video_id)):
                        pseudo_labels_t.append(pseudo_src[video_id[i]]['T'].unsqueeze(0))
                        pseudo_labels_a.append(pseudo_src[video_id[i]]['A'].unsqueeze(0))
                        pseudo_labels_v.append(pseudo_src[video_id[i]]['V'].unsqueeze(0))
                    pseudo_labels_t = torch.cat(pseudo_labels_t).to(device)
                    pseudo_labels_a = torch.cat(pseudo_labels_a).to(device)
                    pseudo_labels_v = torch.cat(pseudo_labels_v).to(device)
                    labels_t = torch.argmax(pseudo_labels_t, dim=-1, keepdim=True)
                    labels_a = torch.argmax(pseudo_labels_a, dim=-1, keepdim=True)
                    labels_v = torch.argmax(pseudo_labels_v, dim=-1, keepdim=True)

                batch_size = text.size(0)

                outputs = model(text, audio, vision)
                loss_m = task_criterion(outputs['pred'], labels.view(-1).long())
                if getattr(hyp_params, 'use_label_smoothing_control', False):
                    loss_t = soft_target_cross_entropy(outputs['pred_t'], pseudo_labels_t)
                    loss_a = soft_target_cross_entropy(outputs['pred_a'], pseudo_labels_a)
                    loss_v = soft_target_cross_entropy(outputs['pred_v'], pseudo_labels_v)
                else:
                    loss_t = task_criterion(outputs['pred_t'], labels_t.view(-1).long())
                    loss_a = task_criterion(outputs['pred_a'], labels_a.view(-1).long())
                    loss_v = task_criterion(outputs['pred_v'], labels_v.view(-1).long())
                loss_task = loss_m + loss_t + loss_a + loss_v

                h_tav = torch.cat((outputs['h_l'], outputs['h_a'], outputs['h_v']), dim=0)
                labels_tav = torch.cat((labels_t, labels_a, labels_v), dim=0)
                if 'SIMS' in hyp_params.dataset:
                    labels_tav = (labels_tav > 1).float()
                else:
                    labels_tav = (labels_tav >= 1).float()
                loss_c = contrastive_criterion(h_tav, labels_tav)

                loss = loss_task + loss_c

                # detach and move to cpu for accumulation
                results.append(outputs['pred'].detach().cpu())
                truths.append(labels.detach().cpu())

                # === 新增：收集 regression 标签（连续值）===
                if 'regression_m' in batch_data:
                    reg_truths.append(batch_data['regression_m'].detach().cpu())

                epoch_loss += loss.item() * batch_size

        epoch_loss /= (hyp_params.n_valid if not test else hyp_params.n_test)
        results = torch.cat(results) if len(results) > 0 else torch.empty(0)
        truths = torch.cat(truths) if len(truths) > 0 else torch.empty(0)
        reg_truths = torch.cat(reg_truths) if len(reg_truths) > 0 else torch.empty(0)
        return epoch_loss, results, truths, reg_truths

    total_parameters = sum([param.nelement() for param in model.parameters()])
    bert_parameters = sum([param.nelement() for param in model.text_model.parameters()])
    print(f'Total Trainable Parameters: {total_parameters}...')
    print(f'BERT Parameters: {bert_parameters}...')
    print(f'PDCCModel Parameters: {total_parameters - bert_parameters}...')

    valid_best_loss = float('inf')
    best_has0_acc2 = 0.0
    curr_patience = hyp_params.patience

    for epoch in range(1, hyp_params.num_epochs + 1):
        start = time.time()

        # ====== 1) Train ======
        train_loss = train_one_epoch(model, optimizer, task_criterion, contrastive_criterion)

        # ====== 2) Eval VALID ======
        valid_loss, valid_logits, valid_cls, valid_reg = evaluate(
            model, task_criterion, contrastive_criterion, test=False
        )

        # ====== 3) Eval TEST (epoch 内监控用；最终 test 仍以 best ckpt 为准) ======
        _, test_logits, test_cls, test_reg = evaluate(
            model, task_criterion, contrastive_criterion, test=True, test_mode='test'
        )

        end = time.time()
        duration = end - start

        print("-" * 50)
        print(f"Epoch {epoch:3d} | Time {duration:7.4f} sec | "
              f"Train Loss {train_loss:7.4f} | Valid Loss {valid_loss:7.4f}")

        # ====== 4) Classification metrics (VALID for model selection, TEST for monitoring only) ======
        if 'SIMS' in hyp_params.dataset:
            valid_metrics = eval_sims_classification(valid_logits, valid_cls)
            test_metrics = eval_sims_classification(test_logits, test_cls)
        else:
            valid_metrics = eval_mosi_classification(valid_logits, valid_cls)
            test_metrics = eval_mosi_classification(test_logits, test_cls)

        print("VALID metrics summary:")
        for k, v in valid_metrics.items():
            try:
                print(f"  {k:20s}: {v:.4f}")
            except Exception:
                print(f"  {k:20s}: {v}")

        print("TEST metrics summary (monitor only):")
        for k, v in test_metrics.items():
            try:
                print(f"  {k:20s}: {v:.4f}")
            except Exception:
                print(f"  {k:20s}: {v}")

        # ====== 5) Regression metrics (关键：同时打印 VALID & TEST) ======
        # 你现在 eval_regression_metrics.py 里 logits3_to_continuous 对 MOSI/MOSEI 是固定 [-3,3] 映射
        # 所以 MAE/Corr 的尺度应该稳定；如果仍异常，一般就是 test_reg/valid_reg 本身或预测塌缩。
        if valid_reg.numel() > 0:
            reg_valid = eval_corr_mae_from_logits3(valid_logits, valid_reg, dataset=hyp_params.dataset)
            print("Regression-style metrics (VALID, Corr/MAE):")
            for k, v in reg_valid.items():
                print(f"  {k:10s}: {float(v):.4f}")
            # debug: 真值范围
            try:
                print(f"[valid_reg] min={float(valid_reg.min()):.3f}, max={float(valid_reg.max()):.3f}")
            except Exception:
                pass
        else:
            print("[Regression-style metrics][VALID] skipped: regression_m not found in dataloader batch.")

        if test_reg.numel() > 0:
            reg_test = eval_corr_mae_from_logits3(test_logits, test_reg, dataset=hyp_params.dataset)
            print("Regression-style metrics (TEST, Corr/MAE):")
            for k, v in reg_test.items():
                print(f"  {k:10s}: {float(v):.4f}")
            # debug: 真值范围
            try:
                print(f"[test_reg] min={float(test_reg.min()):.3f}, max={float(test_reg.max()):.3f}")
            except Exception:
                pass

            # ===== 额外 debug：看预测是否塌缩 =====
            try:
                from src.eval_regression_metrics import logits3_to_continuous
                y_pred_cont = logits3_to_continuous(test_logits, dataset=hyp_params.dataset)
                y_pred_cont = np.asarray(y_pred_cont).reshape(-1)
                print(f"[test_pred_cont] mean={float(y_pred_cont.mean()):.3f}, std={float(y_pred_cont.std()):.3f}, "
                      f"min={float(y_pred_cont.min()):.3f}, max={float(y_pred_cont.max()):.3f}")
            except Exception as e:
                print(f"[test_pred_cont] debug failed: {e}")
        else:
            print("[Regression-style metrics][TEST] skipped: regression_m not found in dataloader batch.")

        # ====== 6) SIMS Non0 print ======
        if 'SIMS' in hyp_params.dataset:
            valid_non0_acc2 = valid_metrics.get('Non0_acc_2', None)
            valid_non0_f1 = valid_metrics.get('Non0_F1_score', None)
            test_non0_acc2 = test_metrics.get('Non0_acc_2', None)
            test_non0_f1 = test_metrics.get('Non0_F1_score', None)

            if valid_non0_acc2 is not None and valid_non0_f1 is not None:
                print(f"[VALID Non0] Acc-2 (w/o neutral): {valid_non0_acc2:.4f} | F1: {valid_non0_f1:.4f}")
            if test_non0_acc2 is not None and test_non0_f1 is not None:
                print(f"[TEST  Non0] Acc-2 (w/o neutral): {test_non0_acc2:.4f} | F1: {test_non0_f1:.4f}")

        # ====== 7) early stop by valid loss（保留） ======
        if valid_loss < valid_best_loss:
            valid_best_loss = valid_loss
            curr_patience = hyp_params.patience
        else:
            curr_patience -= 1

        # ====== 8) save by VALID Has0_acc_2 ======
        cur_valid_acc = float(valid_metrics.get('Has0_acc_2', 0.0))
        if cur_valid_acc > best_has0_acc2:
            best_has0_acc2 = cur_valid_acc
            save_model(hyp_params, model, names={'model_name': 'PDCCModel_Has0_acc_2'})
            print(f">> Saved best PDCCModel_Has0_acc_2 by VALID: {best_has0_acc2:.4f}")

        print(f"Current Best VALID Has0_acc_2: {best_has0_acc2:.4f}")

        if curr_patience <= 0:
            print("Early stopping triggered.")
            break

    # ------------------ Robust load & evaluate best checkpoint ------------------
    loaded = load_model(hyp_params, names={'model_name': 'PDCCModel_Has0_acc_2'})

    best_model = None

    # 1) 如果 load_model 直接返回 state_dict
    if isinstance(loaded, dict):
        try:
            best_model = pdcc_models.PDCCModel(hyp_params)
            best_model.load_state_dict(loaded)
            print('Loaded state_dict into new PDCCModel instance.')
        except Exception as e:
            print(f'Warning: failed to load state_dict into model: {e}')
            best_model = None

    # 2) 如果 load_model 返回了一个 nn.Module
    elif isinstance(loaded, nn.Module):
        best_model = loaded
        print('Loaded best model as nn.Module from load_model.')

    # 3) 如果 load_model 返回的是路径（字符串），尝试 torch.load
    elif isinstance(loaded, (str, bytes, os.PathLike)):
        path = str(loaded)
        try:
            obj = torch.load(path, map_location='cpu')
            if isinstance(obj, dict):
                if 'state_dict' in obj and isinstance(obj['state_dict'], dict):
                    state = obj['state_dict']
                else:
                    state = obj
                best_model = pdcc_models.PDCCModel(hyp_params)
                best_model.load_state_dict(state)
                print(f'Loaded checkpoint from path: {path}')
            elif isinstance(obj, nn.Module):
                best_model = obj
                print(f'Loaded nn.Module object from path: {path}')
            else:
                print(f'Warning: torch.load returned unsupported type {type(obj)} from {path}')
        except Exception as e:
            print(f'Warning: failed to torch.load the path {path}: {e}')
            best_model = None

    else:
        print('Warning: load_model returned None or unsupported type; will fallback to trained model if available.')

    # 兜底：若无法加载 checkpoint，则使用训练期间的 model
    if best_model is None:
        print('Falling back to the trained model (no checkpoint loaded).')
        best_model = model

    # 把 best_model 移到 device 并设为 eval
    try:
        if hyp_params.use_cuda:
            best_model = best_model.to(device)
    except Exception as e:
        print(f'Warning: failed to move best_model to device: {e}')
    best_model.eval()

    # 评估并构建更丰富的返回 ans（包含所有指标）
    ans = {}
    if 'SIMS' in hyp_params.dataset:
        # === train_run.py 内，SIMS 分支 ===
        print('D_test...')
        _, results_test, truths_test, reg_truths_test = evaluate(
            best_model, task_criterion, contrastive_criterion, test=True, test_mode='test'
        )
        metrics_test = eval_sims_classification(results_test, truths_test)
        print("D_test metrics:")
        for k, v in metrics_test.items():
            print(f"  {k:20s}: {v:.4f}" if isinstance(v, (int, float)) else f"  {k:20s}: {v}")
        ans['D_test'] = {k: float(v) if isinstance(v, (int, float)) else v for k, v in metrics_test.items()}

        # D_test Non0
        if 'Non0_acc_2' in metrics_test and 'Non0_F1_score' in metrics_test:
            print(f"[D_test Non0] Acc-2: {metrics_test['Non0_acc_2']:.4f} | F1: {metrics_test['Non0_F1_score']:.4f}")

        # ✅ D_test Corr/MAE（立刻算，用 results_test/reg_truths_test）
        if reg_truths_test.numel() > 0:
            reg_metrics_test = eval_corr_mae_from_logits3(results_test, reg_truths_test, dataset=hyp_params.dataset)
            print("[D_test Regression-style metrics (Corr/MAE)]:")
            for k, v in reg_metrics_test.items():
                print(f"  {k:10s}: {float(v):.4f}")
            ans['D_test_reg'] = {k: float(v) for k, v in reg_metrics_test.items()}

        print('D_msc...')
        _, results_msc, truths_msc, reg_truths_msc = evaluate(
            best_model, task_criterion, contrastive_criterion, test=True, test_mode='D_msc'
        )
        metrics_msc = eval_sims_classification(results_msc, truths_msc)
        print("D_msc metrics:")
        for k, v in metrics_msc.items():
            print(f"  {k:20s}: {v:.4f}" if isinstance(v, (int, float)) else f"  {k:20s}: {v}")
        ans['D_msc'] = {k: float(v) if isinstance(v, (int, float)) else v for k, v in metrics_msc.items()}

        print('D_msi...')
        _, results_msi, truths_msi, reg_truths_msi = evaluate(
            best_model, task_criterion, contrastive_criterion, test=True, test_mode='D_msi'
        )
        metrics_msi = eval_sims_classification(results_msi, truths_msi)
        print("D_msi metrics:")
        for k, v in metrics_msi.items():
            print(f"  {k:20s}: {v:.4f}" if isinstance(v, (int, float)) else f"  {k:20s}: {v}")
        ans['D_msi'] = {k: float(v) if isinstance(v, (int, float)) else v for k, v in metrics_msi.items()}

    else:
        print('D_test...')
        _, test_logits, test_cls, test_reg = evaluate(
            best_model, task_criterion, contrastive_criterion, test=True, test_mode='test'
        )

        metrics_test = eval_mosi_classification(test_logits, test_cls)
        print("D_test metrics:")
        for k, v in metrics_test.items():
            print(f"  {k:20s}: {v:.4f}" if isinstance(v, (int, float)) else f"  {k:20s}: {v}")
        ans['D_test'] = {k: float(v) if isinstance(v, (int, float)) else v for k, v in metrics_test.items()}

        if 'Non0_acc_2' in metrics_test and 'Non0_F1_score' in metrics_test:
            print(f"[D_test Non0] Acc-2: {metrics_test['Non0_acc_2']:.4f} | F1: {metrics_test['Non0_F1_score']:.4f}")

        if test_reg.numel() > 0:
            reg_metrics_test = eval_corr_mae_from_logits3(test_logits, test_reg, dataset=hyp_params.dataset)
            print("[D_test Regression-style metrics (Corr/MAE)]:")
            for k, v in reg_metrics_test.items():
                print(f"  {k:10s}: {float(v):.4f}")
            print(f"[test_reg] min={float(test_reg.min()):.3f}, max={float(test_reg.max()):.3f}")
            ans['D_test_reg'] = {k: float(v) for k, v in reg_metrics_test.items()}

    ans['selection'] = {
        'best_valid_Has0_acc_2': float(best_has0_acc2),
        'selection_metric': 'valid/Has0_acc_2',
    }
    if getattr(hyp_params, 'use_label_smoothing_control', False):
        ans['selection']['label_smoothing_epsilon'] = float(
            hyp_params.label_smoothing_epsilon
        )

    return ans
