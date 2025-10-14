#!/usr/bin/env python3
"""
Phase 3: Alternative Imitability Metrics for LoT
Implements baseline LoT with KL divergence and three alternative metrics:
1. KL Divergence (baseline)
2. L2 Loss (mean squared error between logits)
3. JS Divergence (Jensen-Shannon divergence)
4. Cosine Similarity (cosine distance between logits)

Dataset: CIFAR-10 or CIFAR-100
Models: ResNet-18 (Teacher and Student CNNs)
"""

import argparse
import os
import time
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np

# Set random seeds
def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# IMITABILITY METRICS
# ============================================================================

def kl_divergence(student_logits, teacher_logits, T=1.5):
    """KL Divergence: D_KL(Teacher || Student)"""
    student_probs = F.log_softmax(student_logits / T, dim=1)
    teacher_probs = F.softmax(teacher_logits / T, dim=1)
    kl = F.kl_div(student_probs, teacher_probs, reduction='batchmean') * (T * T)
    return kl


def l2_loss(student_logits, teacher_logits):
    """L2 Loss: Mean squared error between logits"""
    return F.mse_loss(student_logits, teacher_logits)


def js_divergence(student_logits, teacher_logits, T=1.5):
    """Jensen-Shannon Divergence: Symmetric version of KL"""
    student_probs = F.softmax(student_logits / T, dim=1)
    teacher_probs = F.softmax(teacher_logits / T, dim=1)
    
    # Mean distribution
    m = 0.5 * (student_probs + teacher_probs)
    
    # JS = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
    js = 0.5 * F.kl_div(F.log_softmax(student_logits / T, dim=1), m, reduction='batchmean')
    js += 0.5 * F.kl_div(F.log_softmax(teacher_logits / T, dim=1), m, reduction='batchmean')
    js *= (T * T)
    return js


def cosine_similarity_loss(student_logits, teacher_logits):
    """Cosine Similarity Loss: 1 - cosine_similarity"""
    # Normalize to unit vectors
    student_norm = F.normalize(student_logits, p=2, dim=1)
    teacher_norm = F.normalize(teacher_logits, p=2, dim=1)
    
    # Cosine similarity
    cos_sim = (student_norm * teacher_norm).sum(dim=1).mean()
    
    # Convert to loss (distance)
    return 1.0 - cos_sim


# ============================================================================
# METRICS DISPATCHER
# ============================================================================

METRICS = {
    'kl': kl_divergence,
    'l2': l2_loss,
    'js': js_divergence,
    'cosine': cosine_similarity_loss
}


def compute_lot_loss(teacher_logits, student_logits, targets, metric_type, alpha, T=1.5):
    """
    Compute LoT loss with specified imitability metric
    
    Args:
        teacher_logits: Teacher model outputs
        student_logits: Student model outputs
        targets: Ground truth labels
        metric_type: Type of imitability metric ('kl', 'l2', 'js', 'cosine')
        alpha: Regularization strength
        T: Temperature for KL/JS
    """
    # Cross-entropy losses
    teacher_ce = F.cross_entropy(teacher_logits, targets)
    student_ce = F.cross_entropy(student_logits, targets)
    
    # Compute imitability metric
    metric_fn = METRICS[metric_type]
    
    if metric_type in ['kl', 'js']:
        imitability = metric_fn(student_logits, teacher_logits.detach(), T)
    else:
        imitability = metric_fn(student_logits, teacher_logits.detach())
    
    # LoT losses (using negative feedback formulation)
    teacher_loss = teacher_ce + alpha * imitability
    student_loss = student_ce + alpha * imitability
    
    return teacher_loss, student_loss, imitability.item()


# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================

# Using torchvision's ResNet
def get_resnet18(num_classes):
    """Get ResNet-18 model"""
    from torchvision.models import resnet18
    model = resnet18(num_classes=num_classes)
    return model


# ============================================================================
# DATA LOADING
# ============================================================================

def get_cifar_dataloaders(dataset='cifar10', batch_size=128, num_workers=4):
    """Get CIFAR-10 or CIFAR-100 dataloaders"""
    
    # Data augmentation for training
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # No augmentation for test
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    if dataset == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test
        )
        num_classes = 10
    else:  # cifar100
        train_dataset = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=transform_train
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True, transform=transform_test
        )
        num_classes = 100
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, test_loader, num_classes


# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(teacher_model, student_model, train_loader, teacher_opt, student_opt, 
                metric_type, alpha, T, device, epoch):
    """Train for one epoch"""
    teacher_model.train()
    student_model.train()
    
    total_loss = 0
    total_teacher_correct = 0
    total_student_correct = 0
    total_samples = 0
    total_imitability = 0
    
    for batch_idx, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)
        
        # Forward pass
        teacher_logits = teacher_model(data)
        student_logits = student_model(data)
        
        # Compute LoT loss
        teacher_loss, student_loss, imitability = compute_lot_loss(
            teacher_logits, student_logits, targets, metric_type, alpha, T
        )
        
        # Backward pass
        teacher_opt.zero_grad()
        student_opt.zero_grad()
        teacher_loss.backward(retain_graph=True)
        student_loss.backward()
        
        # Optimizer step
        teacher_opt.step()
        student_opt.step()
        
        # Track metrics
        total_loss += student_loss.item()
        total_imitability += imitability
        
        _, teacher_pred = teacher_logits.max(1)
        _, student_pred = student_logits.max(1)
        total_teacher_correct += teacher_pred.eq(targets).sum().item()
        total_student_correct += student_pred.eq(targets).sum().item()
        total_samples += targets.size(0)
        
        if (batch_idx + 1) % 50 == 0:
            print(f'Epoch {epoch} | Batch {batch_idx+1}/{len(train_loader)} | '
                  f'T_Acc: {100.*total_teacher_correct/total_samples:.2f}% | '
                  f'S_Acc: {100.*total_student_correct/total_samples:.2f}% | '
                  f'{metric_type.upper()}: {total_imitability/(batch_idx+1):.4f}')
    
    avg_loss = total_loss / len(train_loader)
    teacher_acc = 100. * total_teacher_correct / total_samples
    student_acc = 100. * total_student_correct / total_samples
    avg_imitability = total_imitability / len(train_loader)
    
    return avg_loss, teacher_acc, student_acc, avg_imitability


def evaluate(teacher_model, student_model, test_loader, device):
    """Evaluate on test set"""
    teacher_model.eval()
    student_model.eval()
    
    total_teacher_correct = 0
    total_student_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            
            teacher_logits = teacher_model(data)
            student_logits = student_model(data)
            
            _, teacher_pred = teacher_logits.max(1)
            _, student_pred = student_logits.max(1)
            
            total_teacher_correct += teacher_pred.eq(targets).sum().item()
            total_student_correct += student_pred.eq(targets).sum().item()
            total_samples += targets.size(0)
    
    teacher_acc = 100. * total_teacher_correct / total_samples
    student_acc = 100. * total_student_correct / total_samples
    
    return teacher_acc, student_acc


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Phase 3: Alternative Imitability Metrics')
    
    # Experiment settings
    parser.add_argument('--metric', type=str, default='kl', 
                       choices=['kl', 'l2', 'js', 'cosine'],
                       help='Imitability metric to use')
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['cifar10', 'cifar100'])
    parser.add_argument('--alpha', type=float, default=0.1,
                       help='LoT regularization strength')
    parser.add_argument('--T', type=float, default=1.5,
                       help='Temperature for KL/JS divergence')
    
    # Training settings
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    
    # System settings
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save', type=str, default='')
    
    args = parser.parse_args()
    
    # Set random seed
    set_random_seed(args.seed)
    
    # Device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Data
    print(f'\nLoading {args.dataset.upper()}...')
    train_loader, test_loader, num_classes = get_cifar_dataloaders(
        args.dataset, args.batch_size, args.num_workers
    )
    print(f'Dataset: {args.dataset} | Classes: {num_classes}')
    print(f'Train samples: {len(train_loader.dataset)}')
    print(f'Test samples: {len(test_loader.dataset)}')
    
    # Models
    print('\nInitializing models...')
    teacher_model = get_resnet18(num_classes).to(device)
    student_model = get_resnet18(num_classes).to(device)
    
    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    student_params = sum(p.numel() for p in student_model.parameters())
    print(f'Teacher parameters: {teacher_params:,}')
    print(f'Student parameters: {student_params:,}')
    
    # Optimizers with learning rate schedule
    teacher_opt = optim.SGD(teacher_model.parameters(), lr=args.lr, 
                           momentum=args.momentum, weight_decay=args.weight_decay)
    student_opt = optim.SGD(student_model.parameters(), lr=args.lr,
                           momentum=args.momentum, weight_decay=args.weight_decay)
    
    # Learning rate scheduler (cosine annealing)
    teacher_scheduler = optim.lr_scheduler.CosineAnnealingLR(teacher_opt, T_max=args.epochs)
    student_scheduler = optim.lr_scheduler.CosineAnnealingLR(student_opt, T_max=args.epochs)
    
    # Print configuration
    print('\n' + '='*80)
    print(f'PHASE 3: {args.metric.upper()} IMITABILITY METRIC')
    print('='*80)
    print(json.dumps(vars(args), indent=4))
    print('='*80)
    
    # Training loop
    print('\nStarting training...')
    best_student_acc = 0.0
    history = {
        'train_loss': [],
        'train_teacher_acc': [],
        'train_student_acc': [],
        'test_teacher_acc': [],
        'test_student_acc': [],
        'imitability': []
    }
    
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        # Train
        train_loss, train_teacher_acc, train_student_acc, avg_imitability = train_epoch(
            teacher_model, student_model, train_loader, teacher_opt, student_opt,
            args.metric, args.alpha, args.T, device, epoch
        )
        
        # Evaluate
        test_teacher_acc, test_student_acc = evaluate(
            teacher_model, student_model, test_loader, device
        )
        
        # Update learning rate
        teacher_scheduler.step()
        student_scheduler.step()
        
        epoch_time = time.time() - start_time
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_teacher_acc'].append(train_teacher_acc)
        history['train_student_acc'].append(train_student_acc)
        history['test_teacher_acc'].append(test_teacher_acc)
        history['test_student_acc'].append(test_student_acc)
        history['imitability'].append(avg_imitability)
        
        # Print epoch summary
        print('='*80)
        print(f'Epoch {epoch}/{args.epochs} | Time: {epoch_time:.2f}s')
        print(f'Train - Teacher: {train_teacher_acc:.2f}% | Student: {train_student_acc:.2f}%')
        print(f'Test  - Teacher: {test_teacher_acc:.2f}% | Student: {test_student_acc:.2f}%')
        print(f'{args.metric.upper()} Metric: {avg_imitability:.4f}')
        print('='*80)
        
        # Save best model
        if test_student_acc > best_student_acc:
            best_student_acc = test_student_acc
            
            if args.save:
                save_dir = os.path.dirname(args.save)
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                
                torch.save({
                    'epoch': epoch,
                    'teacher_state_dict': teacher_model.state_dict(),
                    'student_state_dict': student_model.state_dict(),
                    'teacher_optimizer': teacher_opt.state_dict(),
                    'student_optimizer': student_opt.state_dict(),
                    'best_student_acc': best_student_acc,
                    'test_teacher_acc': test_teacher_acc,
                    'history': history,
                    'args': vars(args)
                }, args.save)
                print(f'✓ Saved best model (Student Acc: {best_student_acc:.2f}%)')
    
    # Final evaluation
    print('\n' + '='*80)
    print('FINAL RESULTS')
    print('='*80)
    test_teacher_acc, test_student_acc = evaluate(
        teacher_model, student_model, test_loader, device
    )
    print(f'Teacher Test Accuracy: {test_teacher_acc:.2f}%')
    print(f'Student Test Accuracy: {test_student_acc:.2f}%')
    print(f'Best Student Accuracy: {best_student_acc:.2f}%')
    print('='*80)


if __name__ == '__main__':
    main()
