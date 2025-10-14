#!/usr/bin/env python3
"""
PHASE 1: Feedback Balance Variants Implementation
Implements three LoT loss variants:
1. Positive-only feedback (reward if student copies teacher well)
2. Negative-only feedback (penalty if student is hard to imitate)
3. Mixed feedback (original LoT with both terms)
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
from itertools import islice

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data_utils import get_lm_corpus
import model.rnn as rnn


parser = argparse.ArgumentParser(description='LoT Phase 1: Feedback Balance Variants')
parser.add_argument('--exp_name', type=str, default='LoT_Phase1')
parser.add_argument('--track', action='store_true', help='Enable WandB tracking')

# Phase 1 specific argument
parser.add_argument('--feedback_type', type=str, default='mixed', 
                    choices=['positive', 'negative', 'mixed'],
                    help='Feedback variant: positive-only, negative-only, or mixed')

parser.add_argument('--alpha', type=float, default=0.1, help='LoT regularization strength')
parser.add_argument('--models_num', type=int, default=2)
parser.add_argument('--detach', type=int, default=1)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--T', type=float, default=1.5, help='Temperature for KL divergence')

# Model & Data
parser.add_argument('--data', type=str, default='ptb', choices=['ptb', 'wt103'])
parser.add_argument('--emsize', type=int, default=650)
parser.add_argument('--nhid', type=int, default=650)
parser.add_argument('--nlayers', type=int, default=2)

# Training
parser.add_argument('--lr', type=float, default=20)
parser.add_argument('--clip', type=float, default=0.25)
parser.add_argument('--epochs', type=int, default=40)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--batch_chunk', type=int, default=1)
parser.add_argument('--bptt', type=int, default=35)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--student_steps_ratio', type=int, default=4)

randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument('--save', type=str, default='ckpt/Phase1/model.pt')
parser.add_argument('--opt', type=str, default='SGD')

args = parser.parse_args()

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
            project=f'LoT_Phase1_{args.data}',
            entity=None,
            name=f"{args.feedback_type}_alpha{args.alpha}_seed{args.seed}",
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


def compute_lot_loss(teacher_logits, student_logits, targets, feedback_type, alpha, T, detach=True):
    """
    Compute LoT loss based on feedback type
    
    Args:
        teacher_logits: Teacher model outputs
        student_logits: Student model outputs
        targets: Ground truth labels
        feedback_type: 'positive', 'negative', or 'mixed'
        alpha: LoT regularization strength
        T: Temperature for KL divergence
        detach: Whether to detach gradients
    
    Returns:
        teacher_loss, student_loss
    """
    # Base cross-entropy loss
    teacher_ce = F.cross_entropy(teacher_logits, targets)
    student_ce = F.cross_entropy(student_logits, targets)
    
    # Detach if needed
    if detach:
        s_logits_for_teacher = student_logits.detach()
        t_logits_for_student = teacher_logits.detach()
    else:
        s_logits_for_teacher = student_logits
        t_logits_for_student = teacher_logits
    
    # Compute KL divergences
    # Teacher->Student: How well student imitates teacher
    kl_teacher_to_student = kl_div_logits(student_logits, t_logits_for_student, T)
    
    # Student->Teacher: How easy is student for teacher to imitate (teachability)
    kl_student_to_teacher = kl_div_logits(teacher_logits, s_logits_for_teacher, T)
    
    if feedback_type == 'positive':
        # POSITIVE-ONLY: Reward if student copies teacher well
        # Teacher gets reward when student is similar (minimize KL)
        # Mathematically: Only keep positive reward (negative KL)
        teacher_loss = teacher_ce - alpha * kl_teacher_to_student  # Negative KL = reward
        student_loss = student_ce + alpha * kl_teacher_to_student  # Student tries to match teacher
        
    elif feedback_type == 'negative':
        # NEGATIVE-ONLY: Penalty if student is hard to imitate
        # Teacher penalized when it's hard for student to learn (high KL student->teacher)
        # Mathematically: Only keep penalty term
        teacher_loss = teacher_ce + alpha * kl_student_to_teacher  # Penalty for being hard to imitate
        student_loss = student_ce + alpha * kl_teacher_to_student  # Student learns from teacher
        
    elif feedback_type == 'mixed':
        # MIXED (ORIGINAL LoT): Both positive and negative terms
        # Teacher gets both reward (when student matches) and penalty (when hard to imitate)
        teacher_loss = teacher_ce + alpha * kl_student_to_teacher  # Original LoT
        student_loss = student_ce + alpha * kl_teacher_to_student
    
    return teacher_loss, student_loss, kl_teacher_to_student.item(), kl_student_to_teacher.item()


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
print(f"PHASE 1: {args.feedback_type.upper()} FEEDBACK VARIANT")
print("="*80)
print(json.dumps(vars(args), indent=4))
print("="*80)

# Model initialization
models = []
for k in range(args.models_num):
    set_random_seed(k)
    model = rnn.RNNModel(ntokens, args.emsize, args.nhid, args.nlayers, args.dropout).to(device)
    models.append(model)
    print(f"Model {k}: {sum(p.numel() for p in model.parameters())} parameters")

criterion = nn.CrossEntropyLoss().cuda()

# Optimizers
if args.opt == 'SGD':
    teacher_opt = torch.optim.SGD(models[0].parameters(), lr=args.lr)
    student_opt = torch.optim.SGD(models[1].parameters(), lr=args.lr)
elif args.opt == 'Adam':
    teacher_opt = torch.optim.Adam(models[0].parameters(), lr=args.lr)
    student_opt = torch.optim.Adam(models[1].parameters(), lr=args.lr)

# Learning rate scheduler
teacher_scheduler = torch.optim.lr_scheduler.StepLR(teacher_opt, step_size=1, gamma=0.25)
student_scheduler = torch.optim.lr_scheduler.StepLR(student_opt, step_size=1, gamma=0.25)


def repackage_hidden(h):
    return tuple(v.clone().detach() for v in h)


def evaluate(data_source):
    """Evaluate both models"""
    losses = []
    for k in range(args.models_num):
        with torch.no_grad():
            models[k].eval()
            total_loss = 0
            hidden = models[k].init_hidden(eval_batch_size)
            for i in range(0, data_source.data.size(0) - 1, args.bptt):
                data, targets, seq_len = data_source.get_batch(i)
                targets = targets.clone().detach().view(-1)
                output, hidden = models[k](data, hidden)
                total_loss += len(data) * criterion(output, targets).data
                hidden = repackage_hidden(hidden)
        losses.append(total_loss / data_source.data.size(0))
    
    for model in models:
        model.train()
    return losses


def train_epoch():
    """Train for one epoch"""
    for k in range(args.models_num):
        models[k].train()
    
    total_teacher_loss = 0
    total_student_loss = 0
    total_kl_t2s = 0  # Teacher to student KL
    total_kl_s2t = 0  # Student to teacher KL
    batch_count = 0
    start_time = time.time()
    
    teacher_hidden = models[0].init_hidden(args.batch_size)
    student_hidden = models[1].init_hidden(args.batch_size)
    
    for batch_idx, i in enumerate(range(0, train_data.data.size(0) - 1, args.bptt)):
        data, targets, seq_len = train_data.get_batch(i)
        targets = targets.clone().detach().view(-1)
        
        teacher_hidden = repackage_hidden(teacher_hidden)
        student_hidden = repackage_hidden(student_hidden)
        
        # Forward pass
        teacher_logits, teacher_hidden = models[0](data, teacher_hidden)
        student_logits, student_hidden = models[1](data, student_hidden)
        
        # Compute LoT loss based on feedback type
        teacher_loss, student_loss, kl_t2s, kl_s2t = compute_lot_loss(
            teacher_logits, student_logits, targets,
            args.feedback_type, args.alpha, args.T, args.detach
        )
        
        # Backward pass
        teacher_opt.zero_grad()
        student_opt.zero_grad()
        teacher_loss.backward(retain_graph=True)
        student_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(models[0].parameters(), args.clip)
        torch.nn.utils.clip_grad_norm_(models[1].parameters(), args.clip)
        
        # Optimizer step
        teacher_opt.step()
        student_opt.step()
        
        # Track metrics
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
                  f"KL(T→S): {kl_t2s:.4f} | KL(S→T): {kl_s2t:.4f} | "
                  f"Time: {elapsed:.2f}s")
    
    avg_metrics = {
        'teacher_loss': total_teacher_loss / batch_count,
        'student_loss': total_student_loss / batch_count,
        'kl_teacher_to_student': total_kl_t2s / batch_count,
        'kl_student_to_teacher': total_kl_s2t / batch_count
    }
    return avg_metrics


# Training loop
print("\nStarting training...")
best_val_loss = float('inf')

for epoch in range(1, args.epochs + 1):
    epoch_start_time = time.time()
    
    # Train
    train_metrics = train_epoch()
    
    # Evaluate
    val_losses = evaluate(val_data)
    val_teacher_ppl = math.exp(val_losses[0])
    val_student_ppl = math.exp(val_losses[1])
    
    elapsed = time.time() - epoch_start_time
    
    print("="*80)
    print(f"Epoch {epoch}/{args.epochs} | Time: {elapsed:.2f}s")
    print(f"Train - Teacher Loss: {train_metrics['teacher_loss']:.4f} | "
          f"Student Loss: {train_metrics['student_loss']:.4f}")
    print(f"Valid - Teacher PPL: {val_teacher_ppl:.2f} | Student PPL: {val_student_ppl:.2f}")
    print(f"KL(T→S): {train_metrics['kl_teacher_to_student']:.4f} | "
          f"KL(S→T): {train_metrics['kl_student_to_teacher']:.4f}")
    print("="*80)
    
    # WandB logging
    if wandb_enabled:
        wandb.log({
            'epoch': epoch,
            'train/teacher_loss': train_metrics['teacher_loss'],
            'train/student_loss': train_metrics['student_loss'],
            'train/kl_teacher_to_student': train_metrics['kl_teacher_to_student'],
            'train/kl_student_to_teacher': train_metrics['kl_student_to_teacher'],
            'valid/teacher_ppl': val_teacher_ppl,
            'valid/student_ppl': val_student_ppl,
            'learning_rate': teacher_opt.param_groups[0]['lr']
        })
    
    # Save best model
    if val_student_ppl < best_val_loss:
        best_val_loss = val_student_ppl
        save_dir = os.path.dirname(args.save)
        if save_dir:  # Only create directory if path contains a directory
            os.makedirs(save_dir, exist_ok=True)
        torch.save({
            'epoch': epoch,
            'teacher_state_dict': models[0].state_dict(),
            'student_state_dict': models[1].state_dict(),
            'teacher_optimizer': teacher_opt.state_dict(),
            'student_optimizer': student_opt.state_dict(),
            'best_val_ppl': best_val_loss,
            'args': vars(args)
        }, args.save)
        print(f"✓ Saved best model (PPL: {best_val_loss:.2f})")
    
    # Learning rate scheduling
    teacher_scheduler.step()
    student_scheduler.step()

# Final test evaluation
print("\n" + "="*80)
print("FINAL TEST EVALUATION")
print("="*80)
test_losses = evaluate(test_data)
test_teacher_ppl = math.exp(test_losses[0])
test_student_ppl = math.exp(test_losses[1])

print(f"Test - Teacher PPL: {test_teacher_ppl:.2f} | Student PPL: {test_student_ppl:.2f}")

if wandb_enabled:
    wandb.log({
        'test/teacher_ppl': test_teacher_ppl,
        'test/student_ppl': test_student_ppl
    })
    wandb.finish()

print("\n✓ Training completed!")
print(f"Results saved to: {args.save}")
