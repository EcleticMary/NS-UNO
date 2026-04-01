
"""
When using the polytropic dataset we use data_utils_p in file main_p.py and model.py if using GP
is data_utils_p2
"""


from data_gp_pt import args, device,dataname  # Import only what you need
print('data is in model : ',dataname)
import torch
from torch.nn import functional as F
from nflows.flows.base import Flow
from nflows.transforms.base import CompositeTransform
from nflows import distributions, flows, transforms, utils
from nflows.nn import nets
import numpy as np
import nflows

from torch import nn


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Flow layers:

# Here we use a Piecewise Rational Quadratic spline in a coupling transform (like in the paper)
# but simpler options are also available.

def PiecewiseRationalQuadraticCouplingTransform(iflow, input_size,context_features=10, num_blocks=args.num_layers_block, activation=F.elu, num_bins=4):
    return transforms.PiecewiseRationalQuadraticCouplingTransform(
        mask=utils.create_alternating_binary_mask(input_size, even=(iflow % 2 == 0)),
        transform_net_create_fn=(lambda in_features, out_features: nets.ResidualNet(
                    in_features=in_features,
                    out_features=out_features,
                    hidden_features=args.mhidden_features,
                    context_features=context_features,
                    num_blocks=args.num_layers_block,
                    activation=args.activation,
                    use_batch_norm=False,
                )),
        num_bins=num_bins, tails='linear', tail_bound=10, apply_unconditional_transform=False)

#####  this is in the coupling.py file of transforms the piecewi....

# Between each Coupling layer we have a layer that suffle the order of the variables
def create_linear_transform(param_dim):
    return transforms.CompositeTransform([
        transforms.RandomPermutation(features=param_dim),
        transforms.LULinear(param_dim, identity_init=True)])

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

base_dist = nflows.distributions.StandardNormal((args.dim,)) # Multivariate Gaussian distribution

class SigmoidTransform(transforms.Transform):
    def forward(self, y, context=None):
        eps = 1e-6  # avoid exact 0 or 1
        y = torch.clamp(y, eps, 1 - eps)
        x = torch.log(y) - torch.log(1 - y)
        # Compute elementwise log-determinant and sum over features:
        logabsdet = (F.softplus(-x) + F.softplus(x)).sum(dim=1)
        return x, logabsdet

    def inverse(self, x, context=None):
        y = torch.sigmoid(x)
        logabsdet = -(F.softplus(-x) + F.softplus(x)).sum(dim=1)
        return y, logabsdet
    
transformsi = []
if args.simgoidtransform:transformsi.append(SigmoidTransform()) # I added sigmoid layer

for _ in range(args.num_flows):
    transformsi.append(create_linear_transform(param_dim=args.dim))

    transformsi.append(PiecewiseRationalQuadraticCouplingTransform(iflow=_, input_size=args.dim, context_features=args.context))
transformsi.append(create_linear_transform(param_dim=args.dim))
transformflow = CompositeTransform(transformsi)
# The Flow class creates the flow model by joining the base distribution and the previously defined layers.
model = Flow(transformflow, base_dist).to(device)


# Optimization of the normalizing flow

def monotonicity_penalty(y):
    """
    Computes the monotonicity penalty for a batch of outputs.
    Penalizes if the vector y is not monotonically increasing.

    Args:
        y (torch.Tensor): Output tensor of shape (batch_size, n).

    Returns:
        torch.Tensor: Penalty term for each batch sample.
    """
    diffs = y[:, 1:] - y[:, :-1]  # Compute differences between consecutive elements
    penalty = torch.clamp(-diffs, min=0)  # Penalize only negative differences
    return penalty.sum(dim=1)  # Sum penalties for each sample in the batch

def flow_loss(inputs, context, flow, lambda_penalty=1.0):
    """
    Computes the total loss for the conditional normalizing flow, including
    monotonicity regularization.

    Args:
        inputs (torch.Tensor): Observed data (e.g., true pressure vectors).
        context (torch.Tensor): Conditioning variables.
        flow: Flow model with _transform and _distribution attributes.
        lambda_penalty (float): Weight for the monotonicity penalty.

    Returns:
        torch.Tensor: Total loss.
    """
    # Forward pass through the flow
    # embedded_context = flow._embedding_net(context)
    noise, logabsdet = flow._transform(inputs, context=context)
    log_prob = flow._distribution.log_prob(noise, context=context)

    # print('inputs',torch.isnan(inputs).any(),torch.isnan(embedded_context).any(),inputs.min().item(),embedded_context.min().item())
    # print('inputs',inputs)
    # print('noise',noise, logabsdet)
    # Negative log-likelihood loss

    nll_loss = -(log_prob + logabsdet)

    # Monotonicity penalty
    outputs, _ = flow._transform.inverse(noise, context=context)  # Generate outputs
    monotonicity_loss = monotonicity_penalty(outputs).mean()  # Average over batch

    #     print("Noise:", noise)
    # print("Context:", embedded_context)
    # print("Transform Params:", transform_params)
    # Total loss

    total_loss = nll_loss.mean()+ lambda_penalty * monotonicity_loss
    # print(nll_loss.mean(), lambda_penalty * monotonicity_loss)
    return total_loss,lambda_penalty * monotonicity_loss
import torch
import torch.nn as nn

def masked_mean(x, mask, dim, eps=1e-6):
    """
    x:    (..., dim_size, ..., feat)
    mask: same shape as x without feat dim
    """
    mask = mask.to(x.dtype)
    x = x * mask
    denom = mask.sum(dim=dim).clamp_min(eps)
    num = x.sum(dim=dim)
    return num / denom


class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, n_layers=2):
        super().__init__()
        layers = []
        d = d_in
        for _ in range(n_layers - 1):
            layers += [nn.Linear(d, d_hidden), nn.ReLU()]
            d = d_hidden
        layers += [nn.Linear(d, d_out)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class HierarchicalDeepSetsEncoder(nn.Module):
    def __init__(self, d_in=2, d_hidden=128, d_hidden_2=256,d_obs=128, d_ctx=128):
        super().__init__()

        # Level 1: samples → observation embedding
        self.phi1 = MLP(d_in, d_hidden, d_hidden)
        self.rho1 = MLP(d_hidden, d_hidden, d_obs)

        # Level 2: observations → EoS context embedding
        self.phi2 = MLP(d_obs, d_hidden, d_hidden_2)
        self.rho2 = MLP(d_hidden_2, d_hidden_2, d_ctx)

    def forward(self, x_padded, mask):
        """
        x_padded: (B, O, S, d_in)
        mask:     (B, O, S)   True = valid sample
        returns:  (B, d_ctx)
        """

        # ----- Level 1 -----
        h = self.phi1(x_padded)                    # (B, O, S, d_hidden)

        h_obs_pre = masked_mean(
            h,
            mask.unsqueeze(-1),
            dim=2
        )                                          # (B, O, d_hidden)

        h_obs = self.rho1(h_obs_pre)               # (B, O, d_obs)

        # observation is valid if any sample exists
        obs_mask = mask.any(dim=2)                 # (B, O)

        # ----- Level 2 -----
        u = self.phi2(h_obs)                       # (B, O, d_hidden)

        ctx_pre = masked_mean(
            u,
            obs_mask.unsqueeze(-1),
            dim=1
        )                                          # (B, d_hidden)

        ctx = self.rho2(ctx_pre)                   # (B, d_ctx)

        return ctx
# encoder = HierarchicalDeepSetsEncoder(d_in=2, d_hidden=128, d_obs=128, d_ctx=128).to(device)


def masked_mean(x, mask, dim, eps=1e-6):
    """
    x:    (..., set, feat)
    mask: (..., set) bool
    """
    mask_f = mask.to(x.dtype)
    num = (x * mask_f.unsqueeze(-1)).sum(dim=dim)
    den = mask_f.sum(dim=dim).clamp_min(eps)
    return num / den.unsqueeze(-1)


class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, n_layers=2):
        super().__init__()
        layers = []
        d = d_in
        for _ in range(n_layers - 1):
            layers += [nn.Linear(d, d_hidden), nn.ReLU()]
            d = d_hidden
        layers += [nn.Linear(d, d_out)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class HierarchicalPhiPhiRhoMean(nn.Module):
    """
    φ1: sample encoder
    mean pool over S
    φ2: observation encoder
    mean pool over O
    ρ : final context head
    """
    def __init__(self, d_in=2, d_hidden=128,d_hidden_2=256, d_ctx=128,
                 phi1_layers=2, phi2_layers=2, rho_layers=2):
        super().__init__()
        self.phi1 = MLP(d_in, d_hidden, d_hidden, n_layers=phi1_layers)
        self.phi2 = MLP(d_hidden, d_hidden, d_hidden_2, n_layers=phi2_layers)
        self.rho  = MLP(d_hidden_2, d_hidden, d_ctx, n_layers=rho_layers)

    def forward(self, x_padded, mask):
        """
        x_padded: (B, O, S, d_in)
        mask:     (B, O, S) bool
        returns:  (B, d_ctx)
        """
        # Encode samples
        h = self.phi1(x_padded)                 # (B,O,S,H)

        # Pool over samples -> per observation
        obs_mask = mask.any(dim=2)              # (B,O) observation exists?
        h_obs = masked_mean(h, mask, dim=2)     # (B,O,H)

        # IMPORTANT: zero invalid observations so phi2 never sees garbage
        h_obs = h_obs * obs_mask.unsqueeze(-1).to(h_obs.dtype)

        # Encode observations
        u = self.phi2(h_obs)                    # (B,O,H)
        u = u * obs_mask.unsqueeze(-1).to(u.dtype)

        # Pool over observations -> per EoS
        eos_feat = masked_mean(u, obs_mask, dim=1)  # (B,H)

        # Final context
        ctx = self.rho(eos_feat)                # (B,d_ctx)
        return ctx
# encoder = HierarchicalPhiPhiRhoMean(d_in=2, d_hidden=128, d_ctx=128).to(device)


def masked_mean(x, mask, dim, eps=1e-6):
    mask = mask.to(x.dtype)
    x = x * mask
    denom = mask.sum(dim=dim).clamp_min(eps)
    num = x.sum(dim=dim)
    return num / denom

def masked_max(x, mask, dim):
    """
    Safe masked max.
    Invalid entries are set to -inf.
    If all entries are invalid, result becomes -inf → replace with 0.
    """
    x_masked = x.masked_fill(~mask, float("-inf"))
    m = x_masked.max(dim=dim).values
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    return m

def masked_mean_max(x, mask, dim, eps=1e-6):
    """
    Returns concat([mean, max]) along last dim.
    """
    mean = masked_mean(x, mask, dim=dim, eps=eps)
    max_ = masked_max(x, mask, dim=dim)
    return torch.cat([mean, max_], dim=-1)

def masked_mean_second(x, mask, dim, eps=1e-6):
    mask = mask.to(x.dtype)
    x_masked = x * mask

    denom = mask.sum(dim=dim).clamp_min(eps)

    mean = x_masked.sum(dim=dim) / denom
    second = (x_masked ** 2).sum(dim=dim) / denom
    # return torch.cat([mean, second], dim=-1)
    return torch.cat([mean, torch.sqrt(second - mean**2 + 1e-6)], dim=-1)

class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, n_layers=2):
        super().__init__()
        layers = []
        d = d_in
        for _ in range(n_layers - 1):
            layers += [nn.Linear(d, d_hidden), nn.ReLU()]
            d = d_hidden
        layers += [nn.Linear(d, d_out)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class HierarchicalDeepSetsEncoder(nn.Module):
    def __init__(self, d_in=2, d_hidden=128, d_hidden_2=256, d_obs=128, d_ctx=128):
        super().__init__()

        # Level 1
        self.phi1 = MLP(d_in, d_hidden, d_hidden)

        # mean+max doubles dimension
        self.rho1 = MLP(2 * d_hidden, d_hidden, d_obs)

        # Level 2
        self.phi2 = MLP(d_obs, d_hidden, d_hidden_2)

        # again mean+max doubles dimension
        self.rho2 = MLP(d_hidden_2, d_hidden_2, d_ctx)

    def forward(self, x_padded, mask):
        """
        x_padded: (B, O, S, d_in)
        mask:     (B, O, S) bool
        returns:  (B, d_ctx)
        """

        # ----- Level 1 -----
        h = self.phi1(x_padded)                          # (B,O,S,d_hidden)

        h_obs_pre = masked_mean_second(
            h,
            mask.unsqueeze(-1),
            dim=2
        )                                                # (B,O,2*d_hidden)

        h_obs = self.rho1(h_obs_pre)                     # (B,O,d_obs)

        # observation valid if any sample exists
        obs_mask = mask.any(dim=2)                       # (B,O)

        # Zero invalid observations (important for stability)
        h_obs = h_obs * obs_mask.unsqueeze(-1).to(h_obs.dtype)

        # ----- Level 2 -----
        u = self.phi2(h_obs)                             # (B,O,d_hidden_2)

        ctx_pre = masked_mean(
            u,
            obs_mask.unsqueeze(-1),
            dim=1
        )                                                # (B,2*d_hidden_2)

        ctx = self.rho2(ctx_pre)                         # (B,d_ctx)
        return ctx
class HierarchicalDeepSetsEncoder(nn.Module):
    def __init__(self, d_in=2, d_hidden=128, d_hidden_2=256, d_obs=128, d_ctx=128):
        super().__init__()

        # Level 1
        self.phi1 = MLP(d_in, d_hidden, d_hidden)

        # mean+max doubles dimension
        self.rho1 = MLP( d_hidden, d_hidden, d_obs)

        # Level 2
        self.phi2 = MLP(d_obs, d_hidden, d_hidden_2)

        # again mean+max doubles dimension
        self.rho2 = MLP(d_hidden_2, d_hidden_2, d_ctx)

    def forward(self, x_padded, mask):
        """
        x_padded: (B, O, S, d_in)
        mask:     (B, O, S) bool
        returns:  (B, d_ctx)
        """

        # ----- Level 1 -----
        h = self.phi1(x_padded)                          # (B,O,S,d_hidden)

        h_obs_pre = masked_mean(
            h,
            mask.unsqueeze(-1),
            dim=2
        )                                                # (B,O,2*d_hidden)

        h_obs = self.rho1(h_obs_pre)                     # (B,O,d_obs)

        # observation valid if any sample exists
        obs_mask = mask.any(dim=2)                       # (B,O)

        # Zero invalid observations (important for stability)
        h_obs = h_obs * obs_mask.unsqueeze(-1).to(h_obs.dtype)

        # ----- Level 2 -----
        u = self.phi2(h_obs)                             # (B,O,d_hidden_2)

        ctx_pre = masked_mean(
            u,
            obs_mask.unsqueeze(-1),
            dim=1
        )                                                # (B,2*d_hidden_2)

        ctx = self.rho2(ctx_pre)                         # (B,d_ctx)
        return ctx

encoder = HierarchicalDeepSetsEncoder(
    d_in=2,
    d_hidden=128,
    d_hidden_2=256,
    d_obs=128,
    d_ctx=128
).to(device)