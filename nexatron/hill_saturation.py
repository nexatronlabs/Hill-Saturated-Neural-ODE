import torch
import torch.nn as nn
import torch.nn.functional as F

class SurgicalHillSaturation(nn.Module):
    """
    Surgical Hill-Kinetics Field Saturation Regularizer for Neural ODEs.
    Reference: Nexatron Labs Technical Report NXL-2026-01.
    
    Formula:
        f_reg(z) = f_base(z) / (1 + phi * ||z||_2^2) - phi * z
    """
    def __init__(self, phi_init: float = 0.10):
        super().__init__()
        self.raw_phi = nn.Parameter(torch.tensor(float(phi_init), dtype=torch.float32))

    @property
    def phi(self) -> torch.Tensor:
        return F.softplus(self.raw_phi) + 1e-5

    def forward(self, f_base: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f_base (torch.Tensor): Output from base neural network vector field.
            z (torch.Tensor): Current state tensor of shape (..., state_dim).
            
        Returns:
            torch.Tensor: Bounded C^inf regularized vector field.
        """
        z_norm_sq = torch.sum(z**2, dim=-1, keepdim=True)
        saturated_field = f_base / (1.0 + self.phi * z_norm_sq)
        damping = - self.phi * z
        return saturated_field + damping