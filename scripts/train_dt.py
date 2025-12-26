# -*- coding: utf-8 -*-
"""
Train Decision Transformer on collected trajectories.

Usage:
    python scripts/train_dt.py \
        --data_path data/dt_trajectories.pkl \
        --output_dir results/dt/run_1
"""
import os
import sys
import argparse
import pickle
from typing import List, Dict, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.training.decision_transformer import DecisionTransformer


def _safe_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # numerically stable softmax wrapper
    return torch.softmax(x, dim=dim)


class TrajectoryDataset(Dataset):
    """Dataset for Decision Transformer training."""
    
    def __init__(
        self,
        trajectories: List[Dict[str, Any]],
        context_length: int = 20,
        action_repr: str = 'ratio',
    ):
        self.trajectories = trajectories
        self.context_length = context_length
        self.action_repr = str(action_repr)
        assert self.action_repr in ['ratio'], "Only action_repr='ratio' is supported in this script version"
        
        # Get observation space shapes from first trajectory
        first_obs = trajectories[0]['observations'][0]
        self.obs_keys = sorted(first_obs.keys())
        self.obs_shapes = {k: first_obs[k].shape for k in self.obs_keys}
        
        # Action dimension for DT input: [slice_strategy_one_hot(5) + dest_ratio(7)]
        # Note: num_slice_sizes is assumed to be 5 (from your env slicing strategies).
        # num_destinations is inferred from trajectories if present, else fallback to 7.
        try:
            nds = [int(t.get('num_destinations')) for t in trajectories if t.get('num_destinations') is not None]
            self.num_destinations = int(max(nds)) if len(nds) > 0 else 7
        except Exception:
            self.num_destinations = 7
        self.num_slice_sizes = 5
        self.action_dim = int(self.num_slice_sizes + self.num_destinations)
        
        print(f"Dataset initialized:")
        print(f"  Trajectories: {len(trajectories)}")
        print(f"  Observation keys: {self.obs_keys}")
        print(f"  Observation shapes: {self.obs_shapes}")
        print(f"  Action dimension: {self.action_dim}")
    
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        traj_len = len(traj['rewards'])
        
        # Sample a random starting point
        if traj_len > self.context_length:
            start_idx = np.random.randint(0, traj_len - self.context_length + 1)
            end_idx = start_idx + self.context_length
        else:
            start_idx = 0
            end_idx = traj_len
        
        # Extract sequence
        observations = {}
        for key in self.obs_keys:
            obs_seq = np.stack([traj['observations'][i][key] for i in range(start_idx, end_idx)])
            observations[key] = torch.FloatTensor(obs_seq)
        
        # Extract actions and convert to proper format for Decision Transformer
        actions_data = traj['actions'][start_idx:end_idx]

        # Prefer using trajectory-stored action space info (collected from env)
        traj_max_k = traj.get('max_k', None)
        traj_num_destinations = traj.get('num_destinations', None)
        if traj_max_k is not None:
            try:
                traj_max_k = int(traj_max_k)
            except Exception:
                traj_max_k = None
        if traj_num_destinations is not None:
            try:
                traj_num_destinations = int(traj_num_destinations)
            except Exception:
                traj_num_destinations = None
        
        # Debug: Print the structure of the first action to understand the format
        if len(actions_data) > 0 and start_idx == 0:
            print(f"Debug - First action structure: {type(actions_data[0])}, shape: {getattr(actions_data[0], 'shape', 'N/A') if hasattr(actions_data[0], 'shape') else 'N/A'}, value: {actions_data[0]}")
        
        # Handle different action formats
        # In this project, trajectories are usually collected with FlattenedDictActionWrapper,
        # so each action is a flat vector: [slice_strategy_idx] + assignment_vec(max_k).
        slice_strategies = []
        assignments = []

        for action in actions_data:
            if isinstance(action, dict):
                # Dict action
                ss = action.get('slice_strategy')
                aa = action.get('assignment')
                # normalize
                if isinstance(ss, (list, tuple, np.ndarray)):
                    ss_val = int(np.asarray(ss).reshape(-1)[0])
                else:
                    ss_val = int(ss)
                slice_strategies.append(ss_val)
                if aa is None:
                    assignments.append([])
                else:
                    assignments.append(np.asarray(aa).reshape(-1).tolist())
                continue

            # Flat action from FlattenedDictActionWrapper
            if isinstance(action, (list, tuple, np.ndarray)):
                arr = np.asarray(action).reshape(-1)
                if arr.size >= 2:
                    ss_val = int(arr[0])
                    assign_vec = arr[1:].astype(np.int64).tolist()
                    slice_strategies.append(ss_val)
                    assignments.append(assign_vec)
                    continue

            # Fallback
            slice_val = int(action.item()) if hasattr(action, 'item') else int(action)
            slice_strategies.append(slice_val)
            assignments.append([])
        
        # Convert to tensors properly
        slice_strategies = torch.LongTensor(slice_strategies)
        
        # Note: for 'ratio' representation, we keep assignments as Python lists for bincount.
        # (No need to pad/truncate to max_k here.)
        
        # Build per-step destination allocation ratios (shape: [seq_len, num_destinations])
        num_slice_sizes = self.num_slice_sizes
        num_destinations = int(traj_num_destinations) if traj_num_destinations is not None else self.num_destinations

        # Determine per-step k for normalization (prefer recorded calculated_ks)
        ks_seq = None
        if 'calculated_ks' in traj and traj.get('calculated_ks') is not None:
            try:
                ks_seq = [int(x) for x in traj.get('calculated_ks')[start_idx:end_idx]]
            except Exception:
                ks_seq = None
        if ks_seq is None:
            # fallback: assume full length of assignment vector
            ks_seq = [len(a) for a in assignments]

        ratios = torch.zeros(len(assignments), num_destinations, dtype=torch.float32)
        counts = torch.zeros(len(assignments), num_destinations, dtype=torch.float32)
        for t_i, a in enumerate(assignments):
            a_vec = list(a) if isinstance(a, (list, tuple)) else []
            k_i = int(ks_seq[t_i]) if t_i < len(ks_seq) else len(a_vec)
            if k_i <= 0:
                continue
            a_vec = a_vec[:k_i]
            if len(a_vec) == 0:
                continue
            binc = np.bincount(np.asarray(a_vec, dtype=np.int64), minlength=num_destinations)[:num_destinations]
            c = torch.tensor(binc, dtype=torch.float32)
            counts[t_i] = c
            ratios[t_i] = c / float(max(k_i, 1))

        # Create DT input action vector as [slice_strategy_one_hot, ratios]
        seq_len = len(slice_strategies)
        actions_flat = torch.zeros(seq_len, num_slice_sizes + num_destinations, dtype=torch.float32)
        for i in range(seq_len):
            ss = int(slice_strategies[i].item())
            if 0 <= ss < num_slice_sizes:
                actions_flat[i, ss] = 1.0
            actions_flat[i, num_slice_sizes:] = ratios[i]

        actions = actions_flat
        
        returns_to_go = torch.FloatTensor(traj['returns_to_go'][start_idx:end_idx]).unsqueeze(-1)
        timesteps = torch.LongTensor(traj['timesteps'][start_idx:end_idx])
        
        # Padding if needed
        seq_len = end_idx - start_idx
        if seq_len < self.context_length:
            pad_len = self.context_length - seq_len
            
            for key in observations:
                pad_shape = (pad_len,) + observations[key].shape[1:]
                observations[key] = torch.cat([
                    observations[key],
                    torch.zeros(pad_shape)
                ], dim=0)
            
            # Pad actions with the correct device
            actions = torch.cat([
                actions,
                torch.zeros(pad_len, actions.shape[1], device=actions.device if hasattr(actions, 'device') else None)
            ])
            
            # Pad slice strategies and assignments with the correct device
            device = slice_strategies.device if hasattr(slice_strategies, 'device') else None
            slice_strategies = torch.cat([
                slice_strategies,
                torch.zeros(pad_len, dtype=torch.long, device=device)
            ])
            
            # In ratio mode, we no longer need to pad the raw assignments list.
            # Padding is handled for slice_strategies, dest_ratios, and dest_counts.
            ratios = torch.cat([ratios, torch.zeros(pad_len, num_destinations)])
            counts = torch.cat([counts, torch.zeros(pad_len, num_destinations)])
            
            returns_to_go = torch.cat([returns_to_go, torch.zeros(pad_len, 1)])
            timesteps = torch.cat([timesteps, torch.zeros(pad_len, dtype=torch.long)])
            
            # Attention mask
            attention_mask = torch.cat([
                torch.ones(seq_len),
                torch.zeros(pad_len)
            ])
        else:
            attention_mask = torch.ones(self.context_length)
        
        # For ratio training, we only need timestep-level attention_mask.
        # Still compute per-step k (useful for debugging/optional weighting).
        with torch.no_grad():
            ks = None
            if 'calculated_ks' in traj and traj.get('calculated_ks') is not None:
                try:
                    ks = [int(x) for x in traj.get('calculated_ks')[start_idx:end_idx]]
                except Exception:
                    ks = None
            if ks is None:
                ks = [0 for _ in range(seq_len)]
            ks = ks + [0] * max(0, self.context_length - len(ks))
            ks = ks[: self.context_length]
            ks_tensor = torch.tensor(ks, dtype=torch.long)

        return {
            'observations': observations,
            'actions': actions,
            'slice_strategies': slice_strategies,
            'dest_ratios': ratios,
            'dest_counts': counts,
            'returns_to_go': returns_to_go,
            'timesteps': timesteps,
            'attention_mask': attention_mask,
            'ks': ks_tensor,
        }


def get_memory_usage(device):
    """Get memory usage information."""
    if device.startswith('cuda'):
        allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(device) / 1024**3  # GB
        return f"GPU: {allocated:.2f}GB / {reserved:.2f}GB"
    return "CPU mode - Memory usage not tracked"

def train_decision_transformer(
    data_path: str,
    output_dir: str,
    context_length: int = 20,
    hidden_dim: int = 128,
    n_layers: int = 3,
    n_heads: int = 4,
    dropout: float = 0.1,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    num_epochs: int = 100,
    eval_freq: int = 5,
    device: str = "cuda",
    save_val_returns: bool = True,
    val_return_jsonl: str = "val_episode_returns.jsonl",
):
    """Train Decision Transformer with enhanced debugging."""
    import time
    import psutil
    from datetime import datetime
    
    # Debug info
    print("\n" + "="*70)
    print(f"🚀 Starting Training - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if device.startswith('cuda'):
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"Num CPUs: {psutil.cpu_count()}")
    print(f"RAM: {psutil.virtual_memory().total / 1024**3:.2f} GB")
    print("="*70 + "\n")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: {os.path.abspath(output_dir)}")
    
    # Initialize TensorBoard
    writer = SummaryWriter(os.path.join(output_dir, 'tensorboard'))
    print("📊 TensorBoard writer initialized")
    
    # Create TXT log files
    train_log_path = os.path.join(output_dir, 'training_log.txt')
    epoch_log_path = os.path.join(output_dir, 'epoch_metrics.txt')
    debug_log_path = os.path.join(output_dir, 'debug_log.txt')
    
    # Write headers
    with open(train_log_path, 'w') as f:
        f.write("# Training Log - Decision Transformer\n")
        f.write("# Format: global_step epoch batch train_loss train_acc learning_rate\n")
        f.write("global_step,epoch,batch,train_loss,train_acc,learning_rate\n")
    
    with open(epoch_log_path, 'w') as f:
        f.write("# Epoch-level Metrics - Decision Transformer\n")
        f.write("# Format: epoch train_loss_mean train_loss_std train_acc_mean train_acc_std val_loss val_acc best_val_loss val_return_mean val_return_std val_return_min val_return_max val_len_mean val_len_std\n")
        f.write("epoch,train_loss_mean,train_loss_std,train_acc_mean,train_acc_std,val_loss,val_acc,best_val_loss,val_return_mean,val_return_std,val_return_min,val_return_max,val_len_mean,val_len_std\n")
    
    # Debug log
    with open(debug_log_path, 'w') as f:
        f.write(f"Training started at {datetime.now()}\n")
        f.write(f"Data path: {data_path}\n")
        f.write(f"Device: {device}\n\n")
    
    # Load data from file or glob pattern
    import glob
    # Treat patterns containing *, ?, or [...] as glob patterns
    if any(ch in data_path for ch in ['*', '?', '[']):
        print(f"Loading trajectories from glob pattern: {data_path}...")
        batch_files = sorted(glob.glob(data_path))
        if not batch_files:
            raise FileNotFoundError(f"No files found for pattern: {data_path}")
        
        trajectories = []
        for batch_file in batch_files:
            try:
                with open(batch_file, 'rb') as f:
                    batch = pickle.load(f)
                trajectories.extend(batch)
                print(f"  ✓ Loaded {os.path.basename(batch_file)}: {len(batch)} trajectories")
            except Exception as e:
                print(f"  ❌ Failed to load {batch_file}: {e}")
    else:
        print(f"Loading trajectories from single file: {data_path}...")
        with open(data_path, 'rb') as f:
            trajectories = pickle.load(f)
    
    # Filter out trajectories with inconsistent num_destinations (must match the majority config)
    # This avoids DataLoader collate crashes due to shape mismatch of dest_ratios/dest_counts.
    expected_num_destinations = 7
    nd_vals = [int(t.get('num_destinations')) for t in trajectories if t.get('num_destinations') is not None and str(t.get('num_destinations')).isdigit()]
    if len(nd_vals) > 0:
        # Use the mode as expected value if available
        from collections import Counter
        expected_num_destinations = Counter(nd_vals).most_common(1)[0][0]

    before = len(trajectories)
    trajectories = [t for t in trajectories if int(t.get('num_destinations', expected_num_destinations)) == expected_num_destinations]
    removed = before - len(trajectories)
    if removed > 0:
        print(f"Filtered out {removed}/{before} trajectories with num_destinations != {expected_num_destinations}")

    # Split train/val (shuffle first to reduce near-duplicate leakage from sequential rollouts)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(trajectories))
    trajectories = [trajectories[i] for i in perm]

    split_idx = int(0.9 * len(trajectories))
    train_trajectories = trajectories[:split_idx]
    val_trajectories = trajectories[split_idx:]
    
    print(f"Train trajectories: {len(train_trajectories)}")
    print(f"Val trajectories: {len(val_trajectories)}")
    
    # Create datasets
    train_dataset = TrajectoryDataset(train_trajectories, context_length=context_length)
    val_dataset = TrajectoryDataset(val_trajectories, context_length=context_length)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    # Create model
    # Get action space configuration from the first trajectory
    first_traj = train_trajectories[0]
    action_space_config = {
        'num_slice_sizes': 5,
        'num_destinations': int(first_traj.get('num_destinations', 7) or 7),
        # ratio-mode does not use max_k for model dimensions
        'max_k': int(first_traj.get('max_k', 0) or 0),
    }
    
    print("\nAction space configuration (ratio-mode):")
    print(f"  Number of slice strategies: {action_space_config['num_slice_sizes']}")
    print(f"  Number of destinations: {action_space_config['num_destinations']}")
    print(f"  (Info) Raw max_k in data: {action_space_config['max_k']}")
    
    model = DecisionTransformer(
        observation_space_shapes=train_dataset.obs_shapes,
        action_space_config=action_space_config,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        dropout=dropout,
        max_ep_len=200,
    ).to(device)
    
    print(f"\nModel architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_losses = []
        train_accs = []
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for batch in pbar:
                observations = {k: v.to(device) for k, v in batch['observations'].items()}
                actions = batch['actions'].to(device)
                returns_to_go = batch['returns_to_go'].to(device)
                timesteps = batch['timesteps'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                # Forward pass
                outputs = model(
                    returns_to_go=returns_to_go,
                    observations=observations,
                    actions=actions,
                    timesteps=timesteps,
                    attention_mask=attention_mask,
                )
                
                # Compute loss for slice strategy
                slice_strategy_logits = outputs['slice_strategy_logits']
                slice_strategy_target = batch['slice_strategies'].to(device)  # [batch, seq_len]
                
                # Mask for valid positions (timestep-level)
                mask = attention_mask.bool()  # [batch, seq_len]

                # Slice strategy loss
                slice_loss = criterion(
                    slice_strategy_logits[mask].view(-1, action_space_config['num_slice_sizes']),
                    slice_strategy_target[mask].view(-1)
                )

                # Ratio-mode destination loss (normalized multinomial NLL)
                # target: dest_ratios in [0,1], sum=1
                dest_logits = outputs['assignment_logits']  # [batch, seq_len, num_destinations]
                dest_log_probs = torch.log_softmax(dest_logits, dim=-1)
                dest_ratio_target = batch['dest_ratios'].to(device)  # [batch, seq_len, num_destinations]
                # NLL per timestep: -sum_i ratio_i * log p_i
                assignment_loss = -(dest_ratio_target[mask] * dest_log_probs[mask]).sum(dim=-1).mean()
                
                # Total loss (weighted sum)
                loss = slice_loss + assignment_loss
                
                # Compute accuracy
                slice_preds = slice_strategy_logits.argmax(dim=-1)
                slice_acc = (slice_preds[mask] == slice_strategy_target[mask]).float().mean()
                
                # Ratio-mode "assignment" metric: compare argmax destination of predicted ratio vs target ratio
                dest_pred = dest_logits.argmax(dim=-1)
                dest_tgt = dest_ratio_target.argmax(dim=-1)
                assignment_acc = (dest_pred[mask] == dest_tgt[mask]).float().mean()
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                # Record
                train_losses.append(loss.item())
                # Use average of slice and assignment accuracies
                combined_acc = (slice_acc + assignment_acc) / 2
                train_accs.append(combined_acc.item())
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'slice_loss': f"{slice_loss.item():.4f}",
                    'assign_loss': f"{assignment_loss.item():.4f}",
                    'slice_acc': f"{slice_acc.item():.4f}",
                    'assign_acc': f"{assignment_acc.item():.4f}",
                })
                
                # Log to tensorboard
                writer.add_scalar('train/loss', loss.item(), global_step)
                writer.add_scalar('train/slice_loss', slice_loss.item(), global_step)
                writer.add_scalar('train/assignment_loss', assignment_loss.item(), global_step)
                writer.add_scalar('train/slice_accuracy', slice_acc.item(), global_step)
                writer.add_scalar('train/assignment_accuracy', assignment_acc.item(), global_step)
                writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], global_step)
                
                # Log to TXT file
                with open(train_log_path, 'a') as f:
                    f.write(f"{global_step},{epoch+1},{len(train_losses)},{loss.item():.6f},"
                           f"{slice_loss.item():.6f},{assignment_loss.item():.6f},"
                           f"{slice_acc.item():.6f},{assignment_acc.item():.6f},"
                           f"{optimizer.param_groups[0]['lr']:.8f}\n")
                
                global_step += 1
        
        # Validation
        if (epoch + 1) % eval_freq == 0:
            model.eval()
            val_losses = []
            val_accs = []
            
            with torch.no_grad():
                for batch in val_loader:
                    observations = {k: v.to(device) for k, v in batch['observations'].items()}
                    actions = batch['actions'].to(device)
                    returns_to_go = batch['returns_to_go'].to(device)
                    timesteps = batch['timesteps'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    
                    outputs = model(
                        returns_to_go=returns_to_go,
                        observations=observations,
                        actions=actions,
                        timesteps=timesteps,
                        attention_mask=attention_mask,
                    )
                    
                    mask = attention_mask.bool()
                    
                    # Slice strategy loss
                    slice_strategy_logits = outputs['slice_strategy_logits']
                    slice_strategy_target = batch['slice_strategies'].to(device)
                    slice_loss = criterion(
                        slice_strategy_logits[mask].view(-1, action_space_config['num_slice_sizes']),
                        slice_strategy_target[mask].view(-1)
                    )
                    
                    # Ratio-mode destination loss (normalized multinomial NLL)
                    dest_logits = outputs['assignment_logits']  # [batch, seq_len, num_destinations]
                    dest_log_probs = torch.log_softmax(dest_logits, dim=-1)
                    dest_ratio_target = batch['dest_ratios'].to(device)
                    assignment_loss = -(dest_ratio_target[mask] * dest_log_probs[mask]).sum(dim=-1).mean()
                    
                    # Total loss
                    loss = slice_loss + assignment_loss
                    
                    # Accuracy
                    slice_preds = slice_strategy_logits.argmax(dim=-1)
                    slice_acc = (slice_preds[mask] == slice_strategy_target[mask]).float().mean()
                    
                    dest_pred = dest_logits.argmax(dim=-1)
                    dest_tgt = dest_ratio_target.argmax(dim=-1)
                    assignment_acc = (dest_pred[mask] == dest_tgt[mask]).float().mean()
                    
                    val_losses.append(loss.item())
                    val_accs.append(slice_acc.item())  # Using slice accuracy as main metric
            
            val_loss = np.mean(val_losses)
            val_acc = np.mean(val_accs)
            
            print(f"\nValidation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
            writer.add_scalar('val/loss', val_loss, epoch)
            writer.add_scalar('val/accuracy', val_acc, epoch)
            
            # Log epoch-level metrics to TXT
            train_loss_mean = np.mean(train_losses)
            train_loss_std = np.std(train_losses)
            train_acc_mean = np.mean(train_accs)
            train_acc_std = np.std(train_accs)

            # ---- Extra: validation "episode return" stats computed from val trajectories ----
            # For offline DT training, this is the sum of rewards per trajectory in the validation split.
            val_returns = []
            val_lens = []
            for t in val_trajectories:
                r = t.get('rewards', [])
                try:
                    r_sum = float(np.sum(r))
                    r_len = int(len(r))
                except Exception:
                    # Fallback in case rewards are not numeric list
                    r_sum = 0.0
                    r_len = 0
                val_returns.append(r_sum)
                val_lens.append(r_len)

            val_return_mean = float(np.mean(val_returns)) if val_returns else 0.0
            val_return_std = float(np.std(val_returns)) if val_returns else 0.0
            val_return_min = float(np.min(val_returns)) if val_returns else 0.0
            val_return_max = float(np.max(val_returns)) if val_returns else 0.0
            val_len_mean = float(np.mean(val_lens)) if val_lens else 0.0
            val_len_std = float(np.std(val_lens)) if val_lens else 0.0

            # TensorBoard logging for returns
            writer.add_scalar('val/return_mean', val_return_mean, epoch)
            writer.add_scalar('val/return_std', val_return_std, epoch)
            writer.add_scalar('val/len_mean', val_len_mean, epoch)

            # Optionally dump per-trajectory returns to jsonl (once, at first eval)
            if save_val_returns and (epoch + 1) == eval_freq:
                import json
                out_path = os.path.join(output_dir, val_return_jsonl)
                with open(out_path, 'w') as jf:
                    for i, (ret, ln) in enumerate(zip(val_returns, val_lens)):
                        jf.write(json.dumps({'traj_index': i, 'return': float(ret), 'length': int(ln)}) + "\n")
                print(f"Saved validation trajectory returns to: {out_path}")

            with open(epoch_log_path, 'a') as f:
                f.write(f"{epoch+1},{train_loss_mean:.6f},{train_loss_std:.6f},"
                       f"{train_acc_mean:.6f},{train_acc_std:.6f},"
                       f"{val_loss:.6f},{val_acc:.6f},{best_val_loss:.6f},"
                       f"{val_return_mean:.6f},{val_return_std:.6f},{val_return_min:.6f},{val_return_max:.6f},"
                       f"{val_len_mean:.6f},{val_len_std:.6f}\n")
            
            # Additional TensorBoard logging
            writer.add_scalar('epoch/train_loss_mean', train_loss_mean, epoch)
            writer.add_scalar('epoch/train_loss_std', train_loss_std, epoch)
            writer.add_scalar('epoch/train_acc_mean', train_acc_mean, epoch)
            writer.add_scalar('epoch/train_acc_std', train_acc_std, epoch)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                }, os.path.join(output_dir, 'best_model.pt'))
                print(f"Saved best model (val_loss: {val_loss:.4f})")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pt'))
    
    writer.close()
    
    # Save final summary
    summary_path = os.path.join(output_dir, 'training_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("Decision Transformer Training Summary\n")
        f.write("="*70 + "\n\n")
        f.write(f"Data path: {data_path}\n")
        f.write(f"Output directory: {output_dir}\n\n")
        f.write("Model Configuration:\n")
        f.write(f"  Context length: {context_length}\n")
        f.write(f"  Hidden dimension: {hidden_dim}\n")
        f.write(f"  Number of layers: {n_layers}\n")
        f.write(f"  Number of heads: {n_heads}\n")
        f.write(f"  Dropout: {dropout}\n\n")
        f.write("Training Configuration:\n")
        f.write(f"  Batch size: {batch_size}\n")
        f.write(f"  Learning rate: {learning_rate}\n")
        f.write(f"  Number of epochs: {num_epochs}\n")
        f.write(f"  Device: {device}\n\n")
        f.write("Training Results:\n")
        f.write(f"  Best validation loss: {best_val_loss:.6f}\n")
        f.write(f"  Total training steps: {global_step}\n\n")
        f.write("Output Files:\n")
        f.write(f"  - best_model.pt: Best model checkpoint\n")
        f.write(f"  - training_log.txt: Step-by-step training metrics\n")
        f.write(f"  - epoch_metrics.txt: Epoch-level aggregated metrics\n")
        f.write(f"  - tensorboard/: TensorBoard event files\n")
        f.write("="*70 + "\n")
    
    print(f"\nTraining complete! Models saved to {output_dir}")
    print(f"\nGenerated files:")
    print(f"  - {os.path.join(output_dir, 'best_model.pt')}")
    print(f"  - {os.path.join(output_dir, 'training_log.txt')}")
    print(f"  - {os.path.join(output_dir, 'epoch_metrics.txt')}")
    print(f"  - {os.path.join(output_dir, 'training_summary.txt')}")
    print(f"\nTo visualize training:")
    print(f"  tensorboard --logdir {output_dir}/tensorboard")


def main():
    parser = argparse.ArgumentParser(description="Train Decision Transformer")
    parser.add_argument("--data_pattern", type=str, required=True, help="Path or glob pattern for trajectory data files (e.g., 'data/dt_trajectories_batch_*.pkl')")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--context_length", type=int, default=20)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--eval_freq", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_val_returns", action="store_true", help="If set, save validation split per-trajectory return/length to jsonl")
    parser.add_argument("--val_return_jsonl", type=str, default="val_episode_returns.jsonl", help="Output jsonl filename under output_dir")
    
    args = parser.parse_args()
    
    train_decision_transformer(
        data_path=args.data_pattern,
        output_dir=args.output_dir,
        context_length=args.context_length,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        eval_freq=args.eval_freq,
        device=args.device,
        save_val_returns=bool(args.save_val_returns),
        val_return_jsonl=args.val_return_jsonl,
    )


if __name__ == "__main__":
    main()

