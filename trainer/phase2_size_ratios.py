#!/usr/bin/env python3
"""
PHASE 2: Teacher-Student Size Ratio Experiments
Tests different combinations of teacher and student sizes:
1. Large Teacher → Small Student
2. Medium Teacher → Small Student
3. Small Teacher → Similar-sized Student
"""

import argparse
import time
import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import json
import wandb
import os
import sys
import configparser

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data_utils import get_lm_corpus
import model.rnn as rnn


parser = argparse.ArgumentParser(description='LoT Phase 2: Teacher-Student Size Ratios')
parser.add_argument('--exp_name', type=str, default='LoT_Phase2')
parser.add_argument('--track', action='store_true', help='Enable WandB tracking')

# Phase 2 specific arguments
parser.add_argument('--teacher_size', type=str, default='medium',
                    choices=['large', 'medium', 'small'],
                    help='Teacher model size')
parser.add_argument('--student_size', type=str, default='small',
                    choices=['large', 'medium', 'small'],
                    help='Student model size')

parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--detach', type=int, default=1)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--T', type=float, default=1.5)

# Training
parser.add_argument('--data', type=str, default='ptb', choices=['ptb', 'wt103'])
parser.add_argument('--lr', type=float, default=20)
parser.add_argument('--clip', type=float, default=0.25)
parser.add_argument('--epochs', type=int, default=40)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--batch_chunk', type=int, default=1)
parser.add_argument('--bptt', type=int, default=35)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--student_steps_ratio', type=int, default=4)

randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument('--save', type=str, default='ckpt/Phase2/model.pt')
parser.add_argument('--opt', type=str, default='SGD')

args = parser.parse_args()

# Model size configurations
MODEL_CONFIGS = {
    'large': {'emsize': 1024, 'nhid': 1024, 'nlayers': 2},  # ~100M params (PTB)
    'medium': {'emsize': 650, 'nhid': 650, 'nlayers': 2},   # ~40M params (PTB)
    'small': {'emsize': 400, 'nhid': 400, 'nlayers': 2},    # ~15M params (PTB)
}

teacher_config = MODEL_CONFIGS[args.teacher_size]
student_config = MODEL_CONFIGS[args.student_size]

# WandB initialization
wandb_enabled = False
if args.track:
    config = configparser.ConfigParser()
    config.read('key.config')
    wandb_username = config.get('WANDB', 'USER_NAME')
    wandb_key = config.get('WANDB', 'API_KEY')
    try:
        wandb.login(key=wandb_key)
        wandb.init(
            project=f'LoT_Phase2_{args.data}',
            entity=None,
            name=f"T{args.teacher_size}_S{args.student_size}_alpha{args.alpha}_seed{args.seed}",
            config=vars(args)
        )
        wandb_enabled = True
        print("✓ WandB initialized")
    except:
        wandb.init(mode="disabled")
        print("⚠ WandB disabled")
else:
    wandb.init(mode="disabled")
    print("ℹ WandB tracking disabled")

torch.cuda.set_device(int(args.gpu))
device = torch.device(f'cuda:{args.gpu}')


def set_random_seed(s):
    np.random.seed(args.seed + s)
    torch.manual_seed(args.seed + s)
    torch.cuda.manual_seed(args.seed + s)


def kl_div_logits(p, q, T):
    """KL divergence between two logit distributions"""
    loss_func = nn.KLDivLoss(reduction='batchmean', log_target=True)
    loss = loss_func(F.log_softmax(p/T, dim=-1), F.log_softmax(q/T, dim=-1)) * T * T
    return loss


# Data loading
set_random_seed(0)
eval_batch_size = 10
assert args.batch_size % args.batch_chunk == 0

if args.data == 'ptb':
    datadir = 'data/ptb'
elif args.data == 'wt103':
    datadir = 'data/wikitext-103'

corpus = get_lm_corpus(datadir, args.data)
ntokens = len(corpus.vocab)
train_data = corpus.get_iterator('train', args.batch_size, args.bptt, device=device, ext_len=0)
val_data = corpus.get_iterator('valid', eval_batch_size, args.bptt, device=device, ext_len=0)
test_data = corpus.get_iterator('test', eval_batch_size, args.bptt, device=device, ext_len=0)

args.eval_interval = 1000
args.max_step = args.epochs * math.ceil(train_data.data.size(0) / args.bptt)

print("="*80)
print(f"PHASE 2: TEACHER-STUDENT SIZE RATIO EXPERIMENT")
print(f"Teacher: {args.teacher_size.upper()} | Student: {args.student_size.upper()}")
print("="*80)

# Initialize teacher and student with different sizes
set_random_seed(0)
teacher_model = rnn.RNNModel(
    ntokens,
    teacher_config['emsize'],
    teacher_config['nhid'],
    teacher_config['nlayers'],
    args.dropout
).to(device)

set_random_seed(1)
student_model = rnn.RNNModel(
    ntokens,
    student_config['emsize'],
    student_config['nhid'],
    student_config['nlayers'],
    args.dropout
).to(device)

teacher_params = sum(p.numel() for p in teacher_model.parameters())
student_params = sum(p.numel() for p in student_model.parameters())
ratio = teacher_params / student_params

print(f"Teacher parameters: {teacher_params:,} ({args.teacher_size})")
print(f"Student parameters: {student_params:,} ({args.student_size})")
print(f"Size ratio (T/S): {ratio:.2f}x")
print("="*80)
print(json.dumps(vars(args), indent=4))
print("="*80)

criterion = nn.CrossEntropyLoss().cuda()

# Optimizers
if args.opt == 'SGD':
    teacher_opt = torch.optim.SGD(teacher_model.parameters(), lr=args.lr)
    student_opt = torch.optim.SGD(student_model.parameters(), lr=args.lr)
elif args.opt == 'Adam':
    teacher_opt = torch.optim.Adam(teacher_model.parameters(), lr=args.lr)
    student_opt = torch.optim.Adam(student_model.parameters(), lr=args.lr)

# Learning rate scheduler
teacher_scheduler = torch.optim.lr_scheduler.StepLR(teacher_opt, step_size=1, gamma=0.25)
student_scheduler = torch.optim.lr_scheduler.StepLR(student_opt, step_size=1, gamma=0.25)


def repackage_hidden(h):
    return tuple(v.clone().detach() for v in h)


def evaluate(model, model_name, data_source):
    """Evaluate a single model"""
    with torch.no_grad():
        model.eval()
        total_loss = 0
        hidden = model.init_hidden(eval_batch_size)
        for i in range(0, data_source.data.size(0) - 1, args.bptt):
            data, targets, seq_len = data_source.get_batch(i)
            targets = targets.clone().detach().view(-1)
            output, hidden = model(data, hidden)
            total_loss += len(data) * criterion(output, targets).data
            hidden = repackage_hidden(hidden)
    model.train()
    return total_loss / data_source.data.size(0)


def compute_kl_divergence(teacher_logits, student_logits):
    """
    Compute bidirectional KL divergence for imitability analysis
    Returns:
        kl_t2s: How well student can imitate teacher (low = good generalization)
        kl_s2t: How easy is student for teacher to imitate (high = harder for teacher)
    """
    with torch.no_grad():
        kl_t2s = kl_div_logits(student_logits, teacher_logits, args.T).item()
        kl_s2t = kl_div_logits(teacher_logits, student_logits, args.T).item()
    return kl_t2s, kl_s2t


def train_epoch():
    """Train for one epoch with LoT"""
    teacher_model.train()
    student_model.train()
    
    total_teacher_loss = 0
    total_student_loss = 0
    total_kl_t2s = 0
    total_kl_s2t = 0
    batch_count = 0
    start_time = time.time()
    
    teacher_hidden = teacher_model.init_hidden(args.batch_size)
    student_hidden = student_model.init_hidden(args.batch_size)
    
    for batch_idx, i in enumerate(range(0, train_data.data.size(0) - 1, args.bptt)):
        data, targets, seq_len = train_data.get_batch(i)
        targets = targets.clone().detach().view(-1)
        
        teacher_hidden = repackage_hidden(teacher_hidden)
        student_hidden = repackage_hidden(student_hidden)
        
        # Forward pass
        teacher_logits, teacher_hidden = teacher_model(data, teacher_hidden)
        student_logits, student_hidden = student_model(data, student_hidden)
        
        # LoT loss (original mixed feedback)
        if args.detach:
            s_logits_for_teacher = student_logits.detach()
            t_logits_for_student = teacher_logits.detach()
        else:
            s_logits_for_teacher = student_logits
            t_logits_for_student = teacher_logits
        
        teacher_loss = F.cross_entropy(teacher_logits, targets) + \
                      args.alpha * kl_div_logits(teacher_logits, s_logits_for_teacher, args.T)
        student_loss = F.cross_entropy(student_logits, targets) + \
                      args.alpha * kl_div_logits(student_logits, t_logits_for_student, args.T)
        
        # Backward pass
        teacher_opt.zero_grad()
        student_opt.zero_grad()
        teacher_loss.backward(retain_graph=True)
        student_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(teacher_model.parameters(), args.clip)
        torch.nn.utils.clip_grad_norm_(student_model.parameters(), args.clip)
        
        # Optimizer step
        teacher_opt.step()
        student_opt.step()
        
        # Track metrics
        kl_t2s, kl_s2t = compute_kl_divergence(teacher_logits, student_logits)
        total_teacher_loss += teacher_loss.item()
        total_student_loss += student_loss.item()
        total_kl_t2s += kl_t2s
        total_kl_s2t += kl_s2t
        batch_count += 1
        
        # Logging
        if (batch_idx + 1) % 200 == 0:
            elapsed = time.time() - start_time
            print(f"Batch {batch_idx+1}/{train_data.data.size(0)//args.bptt} | "
                  f"T_Loss: {teacher_loss.item():.4f} | S_Loss: {student_loss.item():.4f} | "
                  f"KL(T→S): {kl_t2s:.4f} | KL(S→T): {kl_s2t:.4f}")
    
    avg_metrics = {
        'teacher_loss': total_teacher_loss / batch_count,
        'student_loss': total_student_loss / batch_count,
        'kl_teacher_to_student': total_kl_t2s / batch_count,
        'kl_student_to_teacher': total_kl_s2t / batch_count,
        'imitability_ratio': (total_kl_t2s / batch_count) / (total_kl_s2t / batch_count + 1e-8)
    }
    return avg_metrics


# Training loop
print("\nStarting training...")
best_val_student_ppl = float('inf')

for epoch in range(1, args.epochs + 1):
    epoch_start_time = time.time()
    
    # Train
    train_metrics = train_epoch()
    
    # Evaluate
    val_teacher_loss = evaluate(teacher_model, 'Teacher', val_data)
    val_student_loss = evaluate(student_model, 'Student', val_data)
    val_teacher_ppl = math.exp(val_teacher_loss)
    val_student_ppl = math.exp(val_student_loss)
    
    elapsed = time.time() - epoch_start_time
    
    print("="*80)
    print(f"Epoch {epoch}/{args.epochs} | Time: {elapsed:.2f}s")
    print(f"Valid - Teacher PPL: {val_teacher_ppl:.2f} | Student PPL: {val_student_ppl:.2f}")
    print(f"KL(T→S): {train_metrics['kl_teacher_to_student']:.4f} | "
          f"KL(S→T): {train_metrics['kl_student_to_teacher']:.4f}")
    print(f"Imitability Ratio: {train_metrics['imitability_ratio']:.4f}")
    print("="*80)
    
    # WandB logging
    if wandb_enabled:
        wandb.log({
            'epoch': epoch,
            'train/teacher_loss': train_metrics['teacher_loss'],
            'train/student_loss': train_metrics['student_loss'],
            'train/kl_teacher_to_student': train_metrics['kl_teacher_to_student'],
            'train/kl_student_to_teacher': train_metrics['kl_student_to_teacher'],
            'train/imitability_ratio': train_metrics['imitability_ratio'],
            'valid/teacher_ppl': val_teacher_ppl,
            'valid/student_ppl': val_student_ppl,
            'valid/ppl_gap': val_teacher_ppl - val_student_ppl,
            'learning_rate': teacher_opt.param_groups[0]['lr'],
            'size_ratio': ratio
        })
    
    # Save best model
    if val_student_ppl < best_val_student_ppl:
        best_val_student_ppl = val_student_ppl
        save_dir = os.path.dirname(args.save)
        if save_dir:  # Only create directory if path contains a directory
            os.makedirs(save_dir, exist_ok=True)
        torch.save({
            'epoch': epoch,
            'teacher_state_dict': teacher_model.state_dict(),
            'student_state_dict': student_model.state_dict(),
            'teacher_optimizer': teacher_opt.state_dict(),
            'student_optimizer': student_opt.state_dict(),
            'best_val_ppl': best_val_student_ppl,
            'teacher_params': teacher_params,
            'student_params': student_params,
            'size_ratio': ratio,
            'args': vars(args)
        }, args.save)
        print(f"✓ Saved best model (Student PPL: {best_val_student_ppl:.2f})")
    
    # Learning rate scheduling
    teacher_scheduler.step()
    student_scheduler.step()

# Final test evaluation
print("\n" + "="*80)
print("FINAL TEST EVALUATION")
print("="*80)
test_teacher_loss = evaluate(teacher_model, 'Teacher', test_data)
test_student_loss = evaluate(student_model, 'Student', test_data)
test_teacher_ppl = math.exp(test_teacher_loss)
test_student_ppl = math.exp(test_student_loss)

print(f"Test - Teacher PPL: {test_teacher_ppl:.2f} | Student PPL: {test_student_ppl:.2f}")
print(f"Generalization Gap: {test_student_ppl - test_teacher_ppl:.2f}")

if wandb_enabled:
    wandb.log({
        'test/teacher_ppl': test_teacher_ppl,
        'test/student_ppl': test_student_ppl,
        'test/generalization_gap': test_student_ppl - test_teacher_ppl
    })
    wandb.finish()

print("\n✓ Training completed!")
print(f"Results saved to: {args.save}")
