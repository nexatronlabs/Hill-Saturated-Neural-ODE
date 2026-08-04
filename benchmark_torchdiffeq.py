"""
Publication-Grade Benchmark: Surgical Hill-Kinetics Saturation in Neural ODEs
Target Dynamics: Multi-Trajectory Stiff Van der Pol Oscillator (mu=3.0)
Framework: PyTorch + torchdiffeq (Adjoint Method)
"""

import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from torchdiffeq import odeint_adjoint as odeint
except ImportError:
    raise ImportError("Please install torchdiffeq: pip install torchdiffeq")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[System Info] Running benchmark on device: {device}")


class VanDerPol(nn.Module):
    """Ground truth target system: Stiff Van der Pol Oscillator (mu=3.0)"""
    def __init__(self, mu=3.0):
        super().__init__()
        self.mu = mu

    def forward(self, t, z):
        x1, x2 = z[..., 0:1], z[..., 1:2]
        dx1 = x2
        dx2 = self.mu * (1.0 - x1**2) * x2 - x1
        return torch.cat([dx1, dx2], dim=-1)


# Simulation evaluation time grid
t_eval = torch.linspace(0.0, 8.0, 80).to(device)

# Multi-trajectory initial conditions
z0_true = torch.tensor([
    [2.0, 0.0],
    [-1.0, 2.0],
    [0.5, -1.5]
], device=device)

# Generate Ground Truth Trajectories
with torch.no_grad():
    vdp_true = VanDerPol(mu=3.0).to(device)
    z_true = odeint(vdp_true, z0_true, t_eval, method='dopri5', rtol=1e-8, atol=1e-8)


class VanillaNeuralODEFunc(nn.Module):
    """Standard Unregularized Neural ODE Architecture"""
    def __init__(self, state_dim=2, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim)
        )
        self.nfe = 0

    def forward(self, t, z):
        self.nfe += 1
        return self.net(z)


class SaturatedNeuralODEFunc(nn.Module):
    """Proposed Surgical Hill-Kinetics Saturated Neural ODE"""
    def __init__(self, state_dim=2, hidden_dim=64, phi_init=0.33):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim)
        )
        self.raw_phi = nn.Parameter(torch.tensor(phi_init))
        self.nfe = 0

    @property
    def phi(self):
        return torch.nn.functional.softplus(self.raw_phi) + 1e-5

    def forward(self, t, z):
        self.nfe += 1
        f_base = self.net(z)
        
        # Norm-squared of vector state ||z||_2^2
        z_norm_sq = torch.sum(z**2, dim=-1, keepdim=True)
        
        # Surgical rational field saturation + linear damping
        saturated_field = f_base / (1.0 + self.phi * z_norm_sq)
        damping = - self.phi * z
        
        return saturated_field + damping


def train_and_benchmark(model_func, model_name, epochs=50, lr=0.01):
    print(f"\n" + "="*60)
    print(f"Executing Experiment: {model_name}")
    print("="*60)
    
    model_func = model_func.to(device)
    optimizer = optim.Adam(model_func.parameters(), lr=lr)
    
    history = {'loss': [], 'nfe_forward': [], 'epoch_time': [], 'predictions': []}
    start_total_time = time.time()
    
    for epoch in range(1, epochs + 1):
        model_func.nfe = 0
        optimizer.zero_grad()
        t0 = time.time()
        
        pred_z = odeint(model_func, z0_true, t_eval, method='dopri5', rtol=1e-4, atol=1e-5)
        nfe_forward = model_func.nfe
        
        loss = torch.mean((pred_z - z_true) ** 2)
        loss.backward()
        optimizer.step()
        
        elapsed = time.time() - t0
        
        history['loss'].append(loss.item())
        history['nfe_forward'].append(nfe_forward)
        history['epoch_time'].append(elapsed)
        
        if epoch % 10 == 0 or epoch == 1:
            phi_str = f" | Phi: {model_func.phi.item():.4f}" if hasattr(model_func, 'phi') else ""
            print(f"Epoch {epoch:02d}/{epochs:02d} | Loss: {loss.item():.6f} | NFE: {nfe_forward:4d} | Time: {elapsed*1000:6.1f}ms{phi_str}")

    total_time = time.time() - start_total_time
    history['predictions'] = pred_z.detach().cpu().numpy()
    
    print("-" * 60)
    print(f"Completed in: {total_time:.2f}s | Mean NFE: {np.mean(history['nfe_forward']):.1f}")
    return history, model_func


if __name__ == "__main__":
    EPOCHS = 100
    
    vanilla_func = VanillaNeuralODEFunc(state_dim=2, hidden_dim=128)
    history_vanilla, _ = train_and_benchmark(vanilla_func, "Standard Vanilla Neural ODE", epochs=EPOCHS)

    saturated_func = SaturatedNeuralODEFunc(state_dim=2, hidden_dim=128, phi_init=0.10)
    history_sat, final_sat_model = train_and_benchmark(saturated_func, "Surgical Hill-Kinetics Neural ODE (Proposed)", epochs=EPOCHS)
    
    # Summary Metrics
    mean_nfe_vanilla = np.mean(history_vanilla['nfe_forward'])
    mean_nfe_sat = np.mean(history_sat['nfe_forward'])
    
    print("\n" + "="*60)
    print("FINAL BENCHMARK SUMMARY")
    print("="*60)
    print(f"1. Vanilla Neural ODE:          Final Loss = {history_vanilla['loss'][-1]:.6f} | Mean NFE = {mean_nfe_vanilla:.1f}")
    print(f"2. Proposed Saturated Model:   Final Loss = {history_sat['loss'][-1]:.6f} | Mean NFE = {mean_nfe_sat:.1f}")
    print("="*60)
    
    # Visualization Plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Phase Portrait Trajectories
    z_true_np = z_true.cpu().numpy()
    axes[0].plot(z_true_np[:, 0, 0], z_true_np[:, 0, 1], 'k--', label='Ground Truth', linewidth=2)
    axes[0].plot(history_vanilla['predictions'][:, 0, 0], history_vanilla['predictions'][:, 0, 1], 'r:', label='Vanilla ODE', linewidth=2)
    axes[0].plot(history_sat['predictions'][:, 0, 0], history_sat['predictions'][:, 0, 1], 'g-', label='Proposed Saturated', linewidth=2)
    axes[0].set_title("Multi-Traj Phase Space (IC #1)")
    axes[0].set_xlabel("x1"); axes[0].set_ylabel("x2")
    axes[0].grid(True, linestyle='--', alpha=0.6); axes[0].legend()
    
    # Loss Convergence Plot
    axes[1].plot(range(1, EPOCHS+1), history_vanilla['loss'], 'r--o', label='Vanilla Loss')
    axes[1].plot(range(1, EPOCHS+1), history_sat['loss'], 'g-s', label='Proposed Saturated Loss')
    axes[1].set_yscale('log')
    axes[1].set_title("Training Loss Convergence")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("MSE Loss")
    axes[1].grid(True, linestyle='--', alpha=0.6); axes[1].legend()
    
    # Computational Stiffness (NFE) Plot
    axes[2].plot(range(1, EPOCHS+1), history_vanilla['nfe_forward'], 'r--o', label='Vanilla NFE')
    axes[2].plot(range(1, EPOCHS+1), history_sat['nfe_forward'], 'g-s', label='Proposed Saturated NFE')
    axes[2].set_title("Computational Stiffness (NFE)")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Forward NFE (dopri5)")
    axes[2].grid(True, linestyle='--', alpha=0.6); axes[2].legend()
    
    plt.tight_layout()
    plt.savefig("benchmark_results.png", dpi=300)
    print("\n[Output] Benchmark figures saved to 'benchmark_results.png'")