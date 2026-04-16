
"""
This is for when using with attention base mechanism
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

def PiecewiseRationalQuadraticCouplingTransform(iflow, input_size,context_features=10, num_blocks=args.num_layers_block, activation=args.activation, num_bins=4):
    return transforms.PiecewiseRationalQuadraticCouplingTransform(
        mask=utils.create_alternating_binary_mask(input_size, even=(iflow % 2 == 0)),
        transform_net_create_fn=(lambda in_features, out_features: nets.ResidualNet(
                    in_features=in_features,
                    out_features=out_features,
                    hidden_features=args.mhidden_features,
                    context_features=context_features,
                    num_blocks=args.num_layers_block,
                    activation=activation,
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
    total_loss = nll_loss.mean()+ lambda_penalty * monotonicity_loss
    # print(nll_loss.mean(), lambda_penalty * monotonicity_loss)
    return total_loss,lambda_penalty * monotonicity_loss



def masked_mean(x, mask, dim, eps=1e-6):
    mask = mask.to(x.dtype)
    x = x * mask
    denom = mask.sum(dim=dim).clamp_min(eps)
    num = x.sum(dim=dim)
    return num / denom


def masked_mean_second(x, mask, dim, eps=1e-6):
    mask = mask.to(x.dtype)
    x_masked = x * mask

    denom = mask.sum(dim=dim).clamp_min(eps)

    mean = x_masked.sum(dim=dim) / denom
    second = (x_masked ** 2).sum(dim=dim) / denom
    return torch.cat([mean, second], dim=-1)
    # return torch.cat([mean, torch.sqrt(second - mean**2 + 1e-6)], dim=-1)

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

class AttentionPool(nn.Module):
    
    def __init__(self, d_in, d_attn=128):
        super().__init__()
        self.score_net = nn.Sequential(
            nn.Linear(d_in, d_attn),
            nn.Tanh(),
            nn.Linear(d_attn, 1)
        )

    def forward(self, x, mask):
        """
        x:    (B, O, D)
        mask: (B, O) bool
        """
        scores = self.score_net(x).squeeze(-1)   # (B, O)
        scores = scores.masked_fill(~mask, -1e9)

        attn = torch.softmax(scores, dim=1)      # (B, O)
        attn = attn * mask.to(attn.dtype)
        attn = attn / (attn.sum(dim=1, keepdim=True) + 1e-8)

        pooled = torch.sum(attn.unsqueeze(-1) * x, dim=1)  # (B, D)
        return pooled, attn
    
class HierarchicalDeepSetsEncoder(nn.Module):
    def __init__(self, d_in=2, d_hidden=128, d_hidden_2=256, d_obs=128, d_ctx=128):
        super().__init__()

        self.phi1 = MLP(d_in, d_hidden, d_hidden)
        self.rho1 = MLP(2 * d_hidden, d_hidden, d_obs)

        self.phi2 = MLP(d_obs, d_hidden, d_hidden_2)

        self.attn_pool = AttentionPool(d_hidden_2, d_attn=128)
        self.rho2 = MLP(d_hidden_2, d_hidden_2, d_ctx)

    def forward(self, x_padded, mask):
        h = self.phi1(x_padded)   # (B,O,S,d_hidden)

        h_obs_pre = masked_mean_second(
            h,
            mask.unsqueeze(-1),
            dim=2
        )                         # (B,O,2*d_hidden)

        h_obs = self.rho1(h_obs_pre)   # (B,O,d_obs)

        obs_mask = mask.any(dim=2)     # (B,O)
        h_obs = h_obs * obs_mask.unsqueeze(-1).to(h_obs.dtype)

        u = self.phi2(h_obs)           # (B,O,d_hidden_2)

        pooled, attn = self.attn_pool(u, obs_mask)   # (B,d_hidden_2)
        # print('pooled',pooled.shape)
        ctx = self.rho2(pooled)        # (B,d_ctx)
        # print('ctx',ctx.shape)
        return ctx

print("with attention layer")
encoder = HierarchicalDeepSetsEncoder(
    d_in=2,
    d_hidden=128,
    d_hidden_2=256,
    d_obs=128,
    d_ctx=args.context
).to(device)