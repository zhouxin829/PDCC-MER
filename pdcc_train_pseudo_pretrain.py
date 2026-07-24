import pickle
import os
from torch import nn
import torch
import torch.nn.functional as F
import torch.optim as optim
import time

from src import pdcc_models
from src.pdcc_utils import *
from src.eval_metrics import *
from src.pdcc_utils import resolve_pseudo_dir, get_all_pseudo_label_paths

import csv
import numpy as np

####################################################################
# Construct the model
####################################################################

def initiate(hyp_params, dataloaders, device):
    t_model = pdcc_models.TextModel(hyp_params)
    a_model = pdcc_models.AudioModel(hyp_params)
    v_model = pdcc_models.VisionModel(hyp_params)
    ema_t_model = pdcc_models.TextModel(hyp_params).eval()
    ema_a_model = pdcc_models.AudioModel(hyp_params).eval()
    ema_v_model = pdcc_models.VisionModel(hyp_params).eval()
    ema_t_model.load_state_dict(t_model.state_dict())
    ema_a_model.load_state_dict(a_model.state_dict())
    ema_v_model.load_state_dict(v_model.state_dict())

    if hyp_params.use_cuda:
        t_model = t_model.to(device)
        a_model = a_model.to(device)
        v_model = v_model.to(device)
        ema_t_model = ema_t_model.to(device)
        ema_a_model = ema_a_model.to(device)
        ema_v_model = ema_v_model.to(device)

    bert_no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    bert_params = list(t_model.text_model.named_parameters())
    bert_params_decay = [p for n, p in bert_params if not any(nd in n for nd in bert_no_decay)]
    bert_params_no_decay = [p for n, p in bert_params if any(nd in n for nd in bert_no_decay)]
    model_params_other = [p for n, p in list(t_model.named_parameters()) if 'text_model' not in n]
    optimizer_grouped_parameters = [
        {'params': bert_params_decay, 'weight_decay': hyp_params.weight_decay_bert, 'lr': hyp_params.lr_bert},
        {'params': bert_params_no_decay, 'weight_decay': 0.0, 'lr': hyp_params.lr_bert},
        {'params': model_params_other, 'weight_decay': 0.0, 'lr': hyp_params.lr}
    ]
    t_optimizer = optim.Adam(optimizer_grouped_parameters)
    a_optimizer = optim.Adam(a_model.parameters())
    v_optimizer = optim.Adam(v_model.parameters())
    task_criterion = getattr(nn, hyp_params.criterion)()
    contrastive_criterion = pdcc_models.SupConLoss(temperature=hyp_params.pretrain_temperature)

    settings = {
        't_model': t_model,
        'a_model': a_model,
        'v_model': v_model,
        'ema_t_model': ema_t_model,
        'ema_a_model': ema_a_model,
        'ema_v_model': ema_v_model,
        't_optimizer': t_optimizer,
        'a_optimizer': a_optimizer,
        'v_optimizer': v_optimizer,
        'task_criterion': task_criterion,
        'contrastive_criterion': contrastive_criterion
    }
    return train_model(settings, hyp_params, dataloaders, device)

####################################################################
# Training and evaluation scripts
####################################################################

def train_model(settings, hyp_params, dataloaders, device):
    t_model = settings['t_model']
    a_model = settings['a_model']
    v_model = settings['v_model']
    ema_t_model = settings['ema_t_model']
    ema_a_model = settings['ema_a_model']
    ema_v_model = settings['ema_v_model']
    t_optimizer = settings['t_optimizer']
    a_optimizer = settings['a_optimizer']
    v_optimizer = settings['v_optimizer']
    task_criterion = settings['task_criterion']
    contrastive_criterion = settings['contrastive_criterion']

    train_pseudo_labels = {}
    valid_pseudo_labels = {}
    test_pseudo_labels = {}

    entropy_history_rows = []

    init_t_momentum = hyp_params.init_t_momentum
    init_a_momentum = hyp_params.init_a_momentum
    init_v_momentum = hyp_params.init_v_momentum
    init_ema_t_model_momentum = hyp_params.init_ema_t_model_momentum
    init_ema_a_model_momentum = hyp_params.init_ema_a_model_momentum
    init_ema_v_model_momentum = hyp_params.init_ema_v_model_momentum

    def train(models, ema_models, optimizers, task_criterion, contrastive_criterion, epoch):
        t_model, a_model, v_model = models
        ema_t_model, ema_a_model, ema_v_model = ema_models
        t_optimizer, a_optimizer, v_optimizer = optimizers
        t_epoch_loss = 0.0
        a_epoch_loss = 0.0
        v_epoch_loss = 0.0
        c_epoch_loss = 0.0

        t_model.train(); a_model.train(); v_model.train()

        results = {'T': [], 'A': [], 'V': []}
        truths = {'T': [], 'A': [], 'V': []}

        for batch_step, batch_data in enumerate(dataloaders['train']):
            batch_step += (epoch - 1) * len(dataloaders['train']) + 1
            text = batch_data['text'].to(device)
            audio = batch_data['audio'].to(device)
            vision = batch_data['vision'].to(device)
            video_id = batch_data['id']
            labels_mm = batch_data['labels']['M'].to(device)
            labels_tt = batch_data['labels']['T'].to(device)
            labels_aa = batch_data['labels']['A'].to(device)
            labels_vv = batch_data['labels']['V'].to(device)
            one_hot_labels_m = torch.nn.functional.one_hot(labels_mm.view(-1).long(), num_classes=3)

            batch_size = text.size(0)
            t_model.zero_grad(); a_model.zero_grad(); v_model.zero_grad()

            pseudo_labels_t, pseudo_labels_a, pseudo_labels_v = [], [], []
            for i in range(len(video_id)):
                if video_id[i] not in train_pseudo_labels:
                    train_pseudo_labels[video_id[i]] = {'T': None, 'A': None, 'V': None}
                    train_pseudo_labels[video_id[i]]['T'] = one_hot_labels_m[i].float()
                    train_pseudo_labels[video_id[i]]['A'] = one_hot_labels_m[i].float()
                    train_pseudo_labels[video_id[i]]['V'] = one_hot_labels_m[i].float()
                pseudo_labels_t.append(train_pseudo_labels[video_id[i]]['T'].unsqueeze(0))
                pseudo_labels_a.append(train_pseudo_labels[video_id[i]]['A'].unsqueeze(0))
                pseudo_labels_v.append(train_pseudo_labels[video_id[i]]['V'].unsqueeze(0))
            pseudo_labels_t = torch.cat(pseudo_labels_t)
            pseudo_labels_a = torch.cat(pseudo_labels_a)
            pseudo_labels_v = torch.cat(pseudo_labels_v)
            labels_t = torch.argmax(pseudo_labels_t, dim=-1, keepdim=True)
            labels_a = torch.argmax(pseudo_labels_a, dim=-1, keepdim=True)
            labels_v = torch.argmax(pseudo_labels_v, dim=-1, keepdim=True)

            t_outputs = t_model(text)
            a_outputs = a_model(audio)
            v_outputs = v_model(vision)

            one_hot_labels_t = torch.nn.functional.one_hot(labels_t.view(-1).long(), num_classes=3)
            one_hot_labels_a = torch.nn.functional.one_hot(labels_a.view(-1).long(), num_classes=3)
            one_hot_labels_v = torch.nn.functional.one_hot(labels_v.view(-1).long(), num_classes=3)

            use_tplr = getattr(hyp_params, "use_tplr", False)

            if use_tplr:
                if batch_step == 1 and epoch == 1:
                    print(
                        f"[TPLR TRAIN] enabled={use_tplr} mode={getattr(hyp_params, 'tplr_mode', 'full')} "
                        f"steps={getattr(hyp_params, 'tplr_steps', None)} "
                        f"lambda={getattr(hyp_params, 'tplr_lambda', None)} "
                        f"sigma={getattr(hyp_params, 'tplr_sigma', 0.01)}"
                    )

                # EMA Teacher forward (no grad) for teacher-guided diffusion refinement
                with torch.no_grad():
                    ema_t_outputs = ema_t_model(text)
                    ema_a_outputs = ema_a_model(audio)
                    ema_v_outputs = ema_v_model(vision)

                if batch_step == 1 and epoch == 1:
                    same_ptr = (ema_t_outputs['pred'].data_ptr() == t_outputs['pred'].data_ptr())
                    print("[TPLR TRAIN] teacher != student:", (not same_ptr))

                temp_pseudo_labels_t = tplr_refine_pseudo_labels(
                    student_logits=t_outputs['pred'].detach(),
                    one_hot_labels=one_hot_labels_t,
                    hyp_params=hyp_params,
                    teacher_logits=ema_t_outputs['pred'].detach(),
                )
                temp_pseudo_labels_a = tplr_refine_pseudo_labels(
                    student_logits=a_outputs['pred'].detach(),
                    one_hot_labels=one_hot_labels_a,
                    hyp_params=hyp_params,
                    teacher_logits=ema_a_outputs['pred'].detach(),
                )
                temp_pseudo_labels_v = tplr_refine_pseudo_labels(
                    student_logits=v_outputs['pred'].detach(),
                    one_hot_labels=one_hot_labels_v,
                    hyp_params=hyp_params,
                    teacher_logits=ema_v_outputs['pred'].detach(),
                )
                if batch_step == 1 and epoch == 1:
                    base = generate_unified_pseudo_labels(t_outputs['pred'].detach(), one_hot_labels_t)
                    diff = (temp_pseudo_labels_t - base).abs().mean().item()
                    print(f"[TPLR TRAIN] mean|diffused-base| = {diff:.6f}")

            else:
                temp_pseudo_labels_t = generate_unified_pseudo_labels(t_outputs['pred'].detach(), one_hot_labels_t)
                temp_pseudo_labels_a = generate_unified_pseudo_labels(a_outputs['pred'].detach(), one_hot_labels_a)
                temp_pseudo_labels_v = generate_unified_pseudo_labels(v_outputs['pred'].detach(), one_hot_labels_v)

            loss_t = task_criterion(t_outputs['pred'], labels_t.view(-1).long())
            loss_a = task_criterion(a_outputs['pred'], labels_a.view(-1).long())
            loss_v = task_criterion(v_outputs['pred'], labels_v.view(-1).long())

            pseudo_labels_t = t_momentum * pseudo_labels_t + (1 - t_momentum) * temp_pseudo_labels_t
            pseudo_labels_a = a_momentum * pseudo_labels_a + (1 - a_momentum) * temp_pseudo_labels_a
            pseudo_labels_v = v_momentum * pseudo_labels_v + (1 - v_momentum) * temp_pseudo_labels_v
            for i in range(len(video_id)):
                train_pseudo_labels[video_id[i]]['T'] = pseudo_labels_t[i]
                train_pseudo_labels[video_id[i]]['A'] = pseudo_labels_a[i]
                train_pseudo_labels[video_id[i]]['V'] = pseudo_labels_v[i]

            loss_t.backward(retain_graph=True)
            loss_a.backward(retain_graph=True)
            loss_v.backward(retain_graph=True)

            h_tav = torch.cat((t_outputs['h_l'], a_outputs['h_a'], v_outputs['h_v']), dim=0)
            labels_tav = torch.cat((labels_t, labels_a, labels_v), dim=0)
            if 'SIMS' in hyp_params.dataset:
                labels_tav = (labels_tav > 1).float()
            else:
                labels_tav = (labels_tav >= 1).float()
            loss_c = contrastive_criterion(h_tav, labels_tav)
            loss_c.backward()

            t_optimizer.step(); a_optimizer.step(); v_optimizer.step()

            # update ema models
            with torch.no_grad():
                for param, ema_param in zip(t_model.parameters(), ema_t_model.parameters()):
                    ema_param.data = ema_t_model_momentum * ema_param.data + (1 - ema_t_model_momentum) * param.data
                for param, ema_param in zip(a_model.parameters(), ema_a_model.parameters()):
                    ema_param.data = ema_a_model_momentum * ema_param.data + (1 - ema_a_model_momentum) * param.data
                for param, ema_param in zip(v_model.parameters(), ema_v_model.parameters()):
                    ema_param.data = ema_v_model_momentum * ema_param.data + (1 - ema_v_model_momentum) * param.data

            results['T'].append(t_outputs['pred'])
            results['A'].append(a_outputs['pred'])
            results['V'].append(v_outputs['pred'])
            truths['T'].append(labels_tt)
            truths['A'].append(labels_aa)
            truths['V'].append(labels_vv)

            t_epoch_loss += loss_t.item() * batch_size
            a_epoch_loss += loss_a.item() * batch_size
            v_epoch_loss += loss_v.item() * batch_size
            c_epoch_loss += loss_c.item() * batch_size

        denom = hyp_params.n_train
        t_epoch_loss /= denom; a_epoch_loss /= denom; v_epoch_loss /= denom; c_epoch_loss /= denom
        results['T'] = torch.cat(results['T']); results['A'] = torch.cat(results['A']); results['V'] = torch.cat(results['V'])
        truths['T'] = torch.cat(truths['T']); truths['A'] = torch.cat(truths['A']); truths['V'] = torch.cat(truths['V'])
        return (t_epoch_loss, a_epoch_loss, v_epoch_loss, c_epoch_loss), results, truths

    def evaluate(models, ema_models, task_criterion, contrastive_criterion, epoch, test=False):
        t_model, a_model, v_model = models
        ema_t_model, ema_a_model, ema_v_model = ema_models
        t_model.eval(); a_model.eval(); v_model.eval()
        loader = dataloaders['valid'] if not test else dataloaders['test']

        t_epoch_loss = 0.0; a_epoch_loss = 0.0; v_epoch_loss = 0.0; c_epoch_loss = 0.0
        results = {'T': [], 'A': [], 'V': []}; truths = {'T': [], 'A': [], 'V': []}

        with torch.no_grad():
            for batch_step, batch_data in enumerate(loader):
                batch_step += (epoch - 1) * len(loader) + 1
                text = batch_data['text'].to(device)
                audio = batch_data['audio'].to(device)
                vision = batch_data['vision'].to(device)
                video_id = batch_data['id']
                labels_mm = batch_data['labels']['M'].to(device)
                labels_tt = batch_data['labels']['T'].to(device)
                labels_aa = batch_data['labels']['A'].to(device)
                labels_vv = batch_data['labels']['V'].to(device)
                one_hot_labels_m = torch.nn.functional.one_hot(labels_mm.view(-1).long(), num_classes=3)

                batch_size = text.size(0)

                pseudo_labels = valid_pseudo_labels if not test else test_pseudo_labels
                pseudo_labels_t, pseudo_labels_a, pseudo_labels_v = [], [], []
                for i in range(len(video_id)):
                    if video_id[i] not in pseudo_labels:
                        pseudo_labels[video_id[i]] = {'T': None, 'A': None, 'V': None}
                        pseudo_labels[video_id[i]]['T'] = one_hot_labels_m[i].float()
                        pseudo_labels[video_id[i]]['A'] = one_hot_labels_m[i].float()
                        pseudo_labels[video_id[i]]['V'] = one_hot_labels_m[i].float()
                    pseudo_labels_t.append(pseudo_labels[video_id[i]]['T'].unsqueeze(0))
                    pseudo_labels_a.append(pseudo_labels[video_id[i]]['A'].unsqueeze(0))
                    pseudo_labels_v.append(pseudo_labels[video_id[i]]['V'].unsqueeze(0))
                pseudo_labels_t = torch.cat(pseudo_labels_t)
                pseudo_labels_a = torch.cat(pseudo_labels_a)
                pseudo_labels_v = torch.cat(pseudo_labels_v)
                labels_t = torch.argmax(pseudo_labels_t, dim=-1, keepdim=True)
                labels_a = torch.argmax(pseudo_labels_a, dim=-1, keepdim=True)
                labels_v = torch.argmax(pseudo_labels_v, dim=-1, keepdim=True)

                t_outputs = t_model(text)
                a_outputs = a_model(audio)
                v_outputs = v_model(vision)
                ema_t_outputs = ema_t_model(text)
                ema_a_outputs = ema_a_model(audio)
                ema_v_outputs = ema_v_model(vision)

                one_hot_labels_t = torch.nn.functional.one_hot(labels_t.view(-1).long(), num_classes=3)
                one_hot_labels_a = torch.nn.functional.one_hot(labels_a.view(-1).long(), num_classes=3)
                one_hot_labels_v = torch.nn.functional.one_hot(labels_v.view(-1).long(), num_classes=3)

                use_tplr = getattr(hyp_params, "use_tplr", False)

                if use_tplr:
                    # --- Debug print once (eval/valid/test) ---
                    if batch_step == 1 and epoch == 1:
                        phase = "TEST" if test else "VALID"
                        print(
                            f"[TPLR {phase}] enabled={use_tplr} mode={getattr(hyp_params, 'tplr_mode', 'full')} "
                            f"steps={getattr(hyp_params, 'tplr_steps', None)} "
                            f"lambda={getattr(hyp_params, 'tplr_lambda', None)} "
                            f"sigma={getattr(hyp_params, 'tplr_sigma', 0.01)}"
                        )

                    temp_pseudo_labels_t = tplr_refine_pseudo_labels(
                        student_logits=ema_t_outputs['pred'],
                        one_hot_labels=one_hot_labels_t,
                        hyp_params=hyp_params,
                        teacher_logits=ema_t_outputs['pred'],
                    )
                    temp_pseudo_labels_a = tplr_refine_pseudo_labels(
                        student_logits=ema_a_outputs['pred'],
                        one_hot_labels=one_hot_labels_a,
                        hyp_params=hyp_params,
                        teacher_logits=ema_a_outputs['pred'],
                    )
                    temp_pseudo_labels_v = tplr_refine_pseudo_labels(
                        student_logits=ema_v_outputs['pred'],
                        one_hot_labels=one_hot_labels_v,
                        hyp_params=hyp_params,
                        teacher_logits=ema_v_outputs['pred'],
                    )
                else:
                    temp_pseudo_labels_t = generate_unified_pseudo_labels(ema_t_outputs['pred'], one_hot_labels_t)
                    temp_pseudo_labels_a = generate_unified_pseudo_labels(ema_a_outputs['pred'], one_hot_labels_a)
                    temp_pseudo_labels_v = generate_unified_pseudo_labels(ema_v_outputs['pred'], one_hot_labels_v)

                loss_t = task_criterion(ema_t_outputs['pred'], labels_t.view(-1).long())
                loss_a = task_criterion(ema_a_outputs['pred'], labels_a.view(-1).long())
                loss_v = task_criterion(ema_v_outputs['pred'], labels_v.view(-1).long())

                h_tav = torch.cat((ema_t_outputs['h_l'], ema_a_outputs['h_a'], ema_v_outputs['h_v']), dim=0)
                labels_tav = torch.cat((labels_t, labels_a, labels_v), dim=0)
                if 'SIMS' in hyp_params.dataset:
                    labels_tav = (labels_tav > 1).float()
                else:
                    labels_tav = (labels_tav >= 1).float()
                loss_c = contrastive_criterion(h_tav, labels_tav)

                pseudo_labels_t = t_momentum * pseudo_labels_t + (1 - t_momentum) * temp_pseudo_labels_t
                pseudo_labels_a = a_momentum * pseudo_labels_a + (1 - a_momentum) * temp_pseudo_labels_a
                pseudo_labels_v = v_momentum * pseudo_labels_v + (1 - v_momentum) * temp_pseudo_labels_v
                for i in range(len(video_id)):
                    pseudo_labels[video_id[i]]['T'] = pseudo_labels_t[i]
                    pseudo_labels[video_id[i]]['A'] = pseudo_labels_a[i]
                    pseudo_labels[video_id[i]]['V'] = pseudo_labels_v[i]

                results['T'].append(ema_t_outputs['pred'])
                results['A'].append(ema_a_outputs['pred'])
                results['V'].append(ema_v_outputs['pred'])
                truths['T'].append(labels_tt)
                truths['A'].append(labels_aa)
                truths['V'].append(labels_vv)

                t_epoch_loss += loss_t.item() * batch_size
                a_epoch_loss += loss_a.item() * batch_size
                v_epoch_loss += loss_v.item() * batch_size
                c_epoch_loss += loss_c.item() * batch_size

        denom = hyp_params.n_valid if test is False else hyp_params.n_test
        t_epoch_loss /= denom; a_epoch_loss /= denom; v_epoch_loss /= denom; c_epoch_loss /= denom
        results['T'] = torch.cat(results['T']); results['A'] = torch.cat(results['A']); results['V'] = torch.cat(results['V'])
        truths['T'] = torch.cat(truths['T']); truths['A'] = torch.cat(truths['A']); truths['V'] = torch.cat(truths['V'])
        return (t_epoch_loss, a_epoch_loss, v_epoch_loss, c_epoch_loss), results, truths

    # logs
    text_parameters = sum([param.nelement() for param in t_model.parameters()])
    audio_parameters = sum([param.nelement() for param in a_model.parameters()])
    vision_parameters = sum([param.nelement() for param in v_model.parameters()])
    total_parameters = text_parameters + audio_parameters + vision_parameters
    bert_parameters = sum([param.nelement() for param in t_model.text_model.parameters()])
    print(f'Total Trainable Parameters: {total_parameters}...')
    print(f'BERT Parameters: {bert_parameters}...')
    print(f'TextModel Parameters: {text_parameters - bert_parameters}...')
    print(f'AudioModel Parameters: {audio_parameters}...')
    print(f'VisionModel Parameters: {vision_parameters}...')

    inf = 1e8
    valid_best_loss = {'T': inf, 'A': inf, 'V': inf, 'avg': inf}
    test_best_acc = {'T': 0.0, 'A': 0.0, 'V': 0.0}
    curr_patience = hyp_params.patience

    for epoch in range(1, hyp_params.num_epochs + 1):
        start = time.time()

        # momentums
        global t_momentum, a_momentum, v_momentum
        global ema_t_model_momentum, ema_a_model_momentum, ema_v_model_momentum
        t_momentum = get_momentum(init_t_momentum, epoch, gamma=hyp_params.t_momentum_gamma)
        a_momentum = get_momentum(init_a_momentum, epoch, gamma=hyp_params.a_momentum_gamma)
        v_momentum = get_momentum(init_v_momentum, epoch, gamma=hyp_params.v_momentum_gamma)
        ema_t_model_momentum = get_momentum(init_ema_t_model_momentum, epoch, gamma=hyp_params.ema_t_model_momentum_gamma)
        ema_a_model_momentum = get_momentum(init_ema_a_model_momentum, epoch, gamma=hyp_params.ema_a_model_momentum_gamma)
        ema_v_model_momentum = get_momentum(init_ema_v_model_momentum, epoch, gamma=hyp_params.ema_v_model_momentum_gamma)

        train_losses, train_results, train_truths = train(
            (t_model, a_model, v_model),
            (ema_t_model, ema_a_model, ema_v_model),
            (t_optimizer, a_optimizer, v_optimizer),
            task_criterion, contrastive_criterion, epoch
        )
        valid_losses, _, _ = evaluate(
            (t_model, a_model, v_model),
            (ema_t_model, ema_a_model, ema_v_model),
            task_criterion, contrastive_criterion, epoch, test=False
        )
        test_losses, results, truths = evaluate(
            (t_model, a_model, v_model),
            (ema_t_model, ema_a_model, ema_v_model),
            task_criterion, contrastive_criterion, epoch, test=True
        )

        # =========================
        # Record pseudo-label entropy history
        # =========================
        append_entropy_history(entropy_history_rows, epoch, "train", train_pseudo_labels)
        append_entropy_history(entropy_history_rows, epoch, "valid", valid_pseudo_labels)
        append_entropy_history(entropy_history_rows, epoch, "test", test_pseudo_labels)

        latest_train = entropy_history_rows[-3]
        latest_valid = entropy_history_rows[-2]
        latest_test = entropy_history_rows[-1]

        print(
            "[PseudoEntropy] "
            f"Epoch={epoch} | "
            f"Train(T/A/V)=({latest_train['T_mean']:.4f}/{latest_train['A_mean']:.4f}/{latest_train['V_mean']:.4f}) | "
            f"Valid(T/A/V)=({latest_valid['T_mean']:.4f}/{latest_valid['A_mean']:.4f}/{latest_valid['V_mean']:.4f}) | "
            f"Test(T/A/V)=({latest_test['T_mean']:.4f}/{latest_test['A_mean']:.4f}/{latest_test['V_mean']:.4f})"
        )

        end = time.time()
        duration = end - start

        t_train_loss, a_train_loss, v_train_loss, c_train_loss = train_losses
        t_valid_loss, a_valid_loss, v_valid_loss, c_valid_loss = valid_losses
        t_test_loss, a_test_loss, v_test_loss, c_test_loss = test_losses
        print("-" * 50)
        print(f'Epoch {epoch:2d} | Time {duration:5.4f} sec')
        print(f'Train Text Loss {t_train_loss:5.4f} | Train Audio Loss {a_train_loss:5.4f} | Train Vision Loss {v_train_loss:5.4f} | Train Contrastive Loss {c_train_loss:5.4f}')
        print(f'Valid Text Loss {t_valid_loss:5.4f} | Valid Audio Loss {a_valid_loss:5.4f} | Valid Vision Loss {v_valid_loss:5.4f} | Valid Contrastive Loss {c_valid_loss:5.4f}')
        print(f'Test  Text Loss {t_test_loss:5.4f} | Test  Audio Loss {a_test_loss:5.4f} | Test  Vision Loss {v_test_loss:5.4f} | Test  Contrastive Loss {c_test_loss:5.4f}')

        # monitor train & test acc for logs
        t_ans = eval_sims_classification(train_results['T'], train_truths['T'])['Has0_acc_2']
        a_ans = eval_sims_classification(train_results['A'], train_truths['A'])['Has0_acc_2']
        v_ans = eval_sims_classification(train_results['V'], train_truths['V'])['Has0_acc_2']
        print('Current Train Results:')
        print(f'T: {t_ans:.4f} | A: {a_ans:.4f} | V: {v_ans:.4f}')

        t_ans = eval_sims_classification(results['T'], truths['T'])['Has0_acc_2']
        a_ans = eval_sims_classification(results['A'], truths['A'])['Has0_acc_2']
        v_ans = eval_sims_classification(results['V'], truths['V'])['Has0_acc_2']
        print('Current Test Results:')
        print(f'T: {t_ans:.4f} | A: {a_ans:.4f} | V: {v_ans:.4f}')

        # 仅记录 Test 上的最佳 Has0_acc_2（作为日志，不再用于保存模型）
        if t_ans > test_best_acc['T']:
            test_best_acc['T'] = t_ans
        if a_ans > test_best_acc['A']:
            test_best_acc['A'] = a_ans
        if v_ans > test_best_acc['V']:
            test_best_acc['V'] = v_ans

        # === save by per-modality valid loss (rounded to 4 decimals) ===
        improved = False

        t_v = round(t_valid_loss, 4)
        a_v = round(a_valid_loss, 4)
        v_v = round(v_valid_loss, 4)

        if t_v < valid_best_loss['T']:
            improved = True
            valid_best_loss['T'] = t_v
            save_model(hyp_params, t_model, names={'model_name': 'Gen_Pseudo_TextModel'})
            save_model(hyp_params, ema_t_model, names={'model_name': 'Pseudo_TextModel'})

        if a_v < valid_best_loss['A']:
            improved = True
            valid_best_loss['A'] = a_v
            save_model(hyp_params, a_model, names={'model_name': 'Gen_Pseudo_AudioModel'})
            save_model(hyp_params, ema_a_model, names={'model_name': 'Pseudo_AudioModel'})

        if v_v < valid_best_loss['V']:
            improved = True
            valid_best_loss['V'] = v_v
            save_model(hyp_params, v_model, names={'model_name': 'Gen_Pseudo_VisionModel'})
            save_model(hyp_params, ema_v_model, names={'model_name': 'Pseudo_VisionModel'})

        # === write pseudo labels if ANY modality improved ===
        if improved:
            curr_patience = hyp_params.patience

            pseudo_labels_dir = resolve_pseudo_dir(hyp_params)
            os.makedirs(pseudo_labels_dir, exist_ok=True)

            pseudo_paths = get_all_pseudo_label_paths(hyp_params)

            with open(pseudo_paths["train"], 'wb') as f:
                pickle.dump(train_pseudo_labels, f)
            with open(pseudo_paths["valid"], 'wb') as f:
                pickle.dump(valid_pseudo_labels, f)
            with open(pseudo_paths["test"], 'wb') as f:
                pickle.dump(test_pseudo_labels, f)

            print(f"[PseudoLabel] mode = {'PDCC' if innovation_enabled(hyp_params) else 'BASE'}")
            print("[PseudoLabel] Saved:")
            print(f"  train -> {pseudo_paths['train']}")
            print(f"  valid -> {pseudo_paths['valid']}")
            print(f"  test  -> {pseudo_paths['test']}")
        else:
            curr_patience -= 1

        print('Current Test Best Results:')
        print(f'T: {test_best_acc["T"]:.4f} | A: {test_best_acc["A"]:.4f} | V: {test_best_acc["V"]:.4f}')
        if curr_patience <= 0:
            break

    # 保存 entropy history
    save_entropy_history_csv(hyp_params, entropy_history_rows)

    return test_best_acc

# ---------------- helpers ----------------
def compute_alpha(logits, base_labels, reversed_labels):
    probs = torch.softmax(logits, dim=-1)
    g_base = torch.norm(probs - base_labels, p=2, dim=-1)
    g_reverse = torch.norm(probs - reversed_labels, p=2, dim=-1)
    alpha = (g_base - g_reverse) / (g_base + g_reverse + 1e-8)
    return alpha

def reverse_labels(logits, one_hot_labels):
    batch_size, num_classes = one_hot_labels.shape
    reversed_labels = torch.zeros_like(one_hot_labels)
    probs = torch.softmax(logits, dim=-1)
    eye = torch.eye(num_classes, device=logits.device)
    for i in range(batch_size):
        original_class = torch.argmax(one_hot_labels[i]).item()
        min_norm = float('inf')
        best_class = original_class
        for candidate in range(num_classes):
            candidate_label = eye[candidate]
            norm = torch.norm((probs[i] - candidate_label).detach(), p=2)
            if norm < min_norm:
                min_norm = norm
                best_class = candidate
        reversed_labels[i] = eye[best_class]
    return reversed_labels

def generate_unified_pseudo_labels(logits, one_hot_labels):
    base_labels = one_hot_labels
    reversed_labels = reverse_labels(logits, one_hot_labels)
    alpha = compute_alpha(logits, base_labels, reversed_labels)
    pseudo_labels = (1 - alpha.unsqueeze(1)) * base_labels + alpha.unsqueeze(1) * reversed_labels
    return pseudo_labels

def get_momentum(init_momentum, epoch, gamma):
    return 1 - (1 - init_momentum) / (epoch ** gamma if epoch > 0 else 1.0)

def tplr_refine_pseudo_labels(student_logits, one_hot_labels, hyp_params, teacher_logits=None):
    """Generate pseudo labels under one controlled TPLR mechanism variant.

    ``full``: original EMA-teacher-guided progressive refinement.
    ``teacher_only``: an EMA-teacher pseudo-label update without diffusion/noise
        or iterative refinement.  This isolates the value of the progressive
        refinement mechanism from a teacher-only baseline.
    ``student_only``: retains the same progressive noisy refinement but replaces
        the EMA teacher target with the student prediction.  This isolates the
        value of teacher guidance from perturbation/refinement alone.
    """
    base_pseudo = generate_unified_pseudo_labels(student_logits, one_hot_labels)
    if teacher_logits is None:
        teacher_logits = student_logits

    mode = str(getattr(hyp_params, 'tplr_mode', 'full')).lower()
    if mode not in {'full', 'teacher_only', 'student_only'}:
        raise ValueError(f'Unsupported tplr_mode={mode!r}')

    # EMA-only control: same teacher but no stochastic/multi-step refinement.
    if mode == 'teacher_only':
        return generate_unified_pseudo_labels(teacher_logits, one_hot_labels)

    guide_logits = student_logits if mode == 'student_only' else teacher_logits
    guide_probs = torch.softmax(guide_logits, dim=-1)

    tplr_steps = max(1, int(getattr(hyp_params, 'tplr_steps', 1)))
    tplr_lambda = float(getattr(hyp_params, 'tplr_lambda', 1.0))
    tplr_sigma = float(getattr(hyp_params, 'tplr_sigma', 0.01))
    step = tplr_lambda / float(tplr_steps)

    pseudo = base_pseudo
    for _ in range(tplr_steps):
        noise = torch.randn_like(pseudo) * tplr_sigma
        noisy = pseudo + noise
        noisy = torch.clamp(noisy, 1e-6, 1.0)
        noisy = noisy / noisy.sum(dim=-1, keepdim=True)

        pseudo = noisy + step * (guide_probs - noisy)
        pseudo = torch.clamp(pseudo, 1e-6, 1.0)
        pseudo = pseudo / pseudo.sum(dim=-1, keepdim=True)
    return pseudo

# ---------------- entropy helpers ----------------
def pseudo_entropy(p, eps=1e-12):
    """
    p: (B, C) or (C,)
    return:
      if (B, C) -> (B,)
      if (C,)   -> scalar tensor
    """
    if p.dim() == 1:
        p = p.unsqueeze(0)
        squeeze_back = True
    else:
        squeeze_back = False

    p = torch.clamp(p, min=eps, max=1.0)
    p = p / p.sum(dim=-1, keepdim=True)
    ent = -(p * (p + eps).log()).sum(dim=-1)

    if squeeze_back:
        return ent[0]
    return ent


def summarize_entropy_list(vals):
    arr = np.asarray(vals, dtype=np.float64)
    if arr.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "q25": float("nan"),
            "median": float("nan"),
            "q75": float("nan"),
            "max": float("nan"),
        }
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "q25": float(np.quantile(arr, 0.25)),
        "median": float(np.quantile(arr, 0.50)),
        "q75": float(np.quantile(arr, 0.75)),
        "max": float(np.max(arr)),
    }


def summarize_pseudo_memory_entropy(memory_dict):
    """
    memory_dict:
      {
        video_id: {'T': tensor(3,), 'A': tensor(3,), 'V': tensor(3,)}
      }
    """
    bank = {"T": [], "A": [], "V": []}
    for _, v in memory_dict.items():
        for m in ["T", "A", "V"]:
            p = v[m]
            if not isinstance(p, torch.Tensor):
                p = torch.tensor(p, dtype=torch.float32)
            bank[m].append(float(pseudo_entropy(p.detach().cpu()).item()))

    summary = {}
    for m in ["T", "A", "V"]:
        st = summarize_entropy_list(bank[m])
        for k, vv in st.items():
            summary[f"{m}_{k}"] = vv
    return summary


def append_entropy_history(history_rows, epoch, split_name, memory_dict):
    summary = summarize_pseudo_memory_entropy(memory_dict)
    row = {"epoch": epoch, "split": split_name}
    row.update(summary)
    history_rows.append(row)


def save_entropy_history_csv(args, history_rows, filename=None):
    if filename is None:
        tag = "PDCC" if any([
            getattr(args, "use_tplr", False),
            getattr(args, "use_pcrp", False),
            getattr(args, "use_rccr", False),
        ]) else "BASE"
        filename = f"{args.dataset}_{tag}_pseudo_entropy_history.csv"

    out_dir = resolve_pseudo_dir(args)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, filename)

    if len(history_rows) == 0:
        print(f"[EntropyHistory] Empty history, skip saving: {out_csv}")
        return out_csv

    fieldnames = list(history_rows[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history_rows)

    print(f"[EntropyHistory] Saved to {out_csv}")
    return out_csv





