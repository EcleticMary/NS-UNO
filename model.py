"""
Model definition for the NS-UNO normalizing-flow framework.

This version uses a hierarchical DeepSets encoder with an attention
pooling layer to encode an arbitrary number of neutron-star observations.
"""

import torch
from torch import nn
from torch.nn import functional as F

import nflows
from nflows import transforms, utils
from nflows.flows.base import Flow
from nflows.nn import nets
from nflows.transforms.base import CompositeTransform

from data_gp_pt import args, device, dataname


print("Data is in model:", dataname)


# ============================================================================
# Normalizing-flow transformations
# ============================================================================

def piecewise_rational_quadratic_coupling_transform(
    iflow,
    input_size,
    context_features=10,
    num_blocks=args.num_layers_block,
    activation=args.activation,
    num_bins=4,
):
    """
    Create one piecewise rational-quadratic coupling transform.

    Alternating binary masks are used between coupling layers.
    """

    return transforms.PiecewiseRationalQuadraticCouplingTransform(
        mask=utils.create_alternating_binary_mask(
            input_size,
            even=(iflow % 2 == 0),
        ),
        transform_net_create_fn=lambda in_features, out_features: (
            nets.ResidualNet(
                in_features=in_features,
                out_features=out_features,
                hidden_features=args.mhidden_features,
                context_features=context_features,
                num_blocks=num_blocks,
                activation=activation,
                use_batch_norm=False,
            )
        ),
        num_bins=num_bins,
        tails="linear",
        tail_bound=10,
        apply_unconditional_transform=False,
    )


def create_linear_transform(param_dim):
    """
    Create a permutation + LU linear transformation.
    """

    return transforms.CompositeTransform(
        [
            transforms.RandomPermutation(features=param_dim),
            transforms.LULinear(param_dim, identity_init=True),
        ]
    )


class SigmoidTransform(transforms.Transform):
    """
    Element-wise sigmoid transform.

    Used to map values to the interval (0, 1).
    """

    def forward(self, y, context=None):
        eps = 1e-6

        y = torch.clamp(y, eps, 1 - eps)

        x = torch.log(y) - torch.log(1 - y)

        logabsdet = (
            F.softplus(-x) + F.softplus(x)
        ).sum(dim=1)

        return x, logabsdet

    def inverse(self, x, context=None):
        y = torch.sigmoid(x)

        logabsdet = -(
            F.softplus(-x) + F.softplus(x)
        ).sum(dim=1)

        return y, logabsdet


# ============================================================================
# Build normalizing flow
# ============================================================================

base_dist = nflows.distributions.StandardNormal(
    (args.dim,)
)

flow_transforms = []

# Optional sigmoid transformation.
# NOTE: keep the argument name consistent with the configuration file.
if args.sigmoid_transform:
    flow_transforms.append(SigmoidTransform())


for iflow in range(args.num_flows):

    flow_transforms.append(
        create_linear_transform(param_dim=args.dim)
    )

    flow_transforms.append(
        piecewise_rational_quadratic_coupling_transform(
            iflow=iflow,
            input_size=args.dim,
            context_features=args.context,
        )
    )


flow_transforms.append(
    create_linear_transform(param_dim=args.dim)
)

transform_flow = CompositeTransform(flow_transforms)

model = Flow(
    transform_flow,
    base_dist,
).to(device)


# ============================================================================
# Physics-informed loss
# ============================================================================

def monotonicity_penalty(y):
    """
    Penalize outputs that are not monotonically increasing.

    Parameters
    ----------
    y : torch.Tensor
        Output tensor with shape (batch_size, n).

    Returns
    -------
    torch.Tensor
        Monotonicity penalty for each sample in the batch.
    """

    differences = y[:, 1:] - y[:, :-1]

    penalty = torch.clamp(
        -differences,
        min=0,
    )

    return penalty.sum(dim=1)


def flow_loss(inputs, context, flow, lambda_penalty=1.0):
    """
    Compute the negative log-likelihood plus the monotonicity penalty.

    Parameters
    ----------
    inputs : torch.Tensor
        Target EoS parameter vectors.

    context : torch.Tensor
        Encoded observational context.

    flow : nflows.flows.base.Flow
        Conditional normalizing-flow model.

    lambda_penalty : float
        Weight of the monotonicity penalty.

    Returns
    -------
    total_loss : torch.Tensor
        Total training loss.

    weighted_penalty : torch.Tensor
        Monotonicity penalty multiplied by lambda_penalty.
    """

    # Transform the inputs to the latent space.
    noise, logabsdet = flow._transform(
        inputs,
        context=context,
    )

    log_prob = flow._distribution.log_prob(
        noise,
        context=context,
    )

    # Negative log-likelihood.
    nll_loss = -(log_prob + logabsdet)

    # Transform back to the EoS space for the physics constraint.
    outputs, _ = flow._transform.inverse(
        noise,
        context=context,
    )

    monotonicity_loss = monotonicity_penalty(
        outputs
    ).mean()

    weighted_penalty = (
        lambda_penalty * monotonicity_loss
    )

    total_loss = (
        nll_loss.mean()
        + weighted_penalty
    )

    return total_loss, weighted_penalty


# ============================================================================
# Masked statistics
# ============================================================================

def masked_mean(x, mask, dim, eps=1e-6):
    """
    Compute a masked mean along a specified dimension.
    """

    mask = mask.to(x.dtype)

    x_masked = x * mask

    denominator = mask.sum(dim=dim).clamp_min(eps)
    numerator = x_masked.sum(dim=dim)

    return numerator / denominator


def masked_mean_second(x, mask, dim, eps=1e-6):
    """
    Compute the masked mean and variance along a specified dimension.

    The two quantities are concatenated along the final dimension.
    """

    mask = mask.to(x.dtype)

    x_masked = x * mask

    denominator = mask.sum(dim=dim).clamp_min(eps)

    mean = x_masked.sum(dim=dim) / denominator

    second_moment = (
        (x_masked ** 2).sum(dim=dim)
        / denominator
    )

    variance = torch.clamp(
        second_moment - mean ** 2,
        min=eps,
    )

    return torch.cat(
        [mean, variance],
        dim=-1,
    )


# ============================================================================
# Encoder
# ============================================================================

class MLP(nn.Module):
    """
    Simple multilayer perceptron.
    """

    def __init__(
        self,
        d_in,
        d_hidden,
        d_out,
        n_layers=2,
    ):
        super().__init__()

        layers = []
        dimension = d_in

        for _ in range(n_layers - 1):
            layers.extend(
                [
                    nn.Linear(dimension, d_hidden),
                    nn.ReLU(),
                ]
            )

            dimension = d_hidden

        layers.append(
            nn.Linear(dimension, d_out)
        )

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class AttentionPool(nn.Module):
    """
    Attention-based pooling over a variable number of observations.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor with shape (..., N, D).

    mask : torch.Tensor
        Boolean mask with shape (..., N).

    Returns
    -------
    pooled : torch.Tensor
        Pooled representation with shape (..., D).

    attention : torch.Tensor
        Attention weights with shape (..., N).
    """

    def __init__(
        self,
        d_in,
        d_attn=128,
    ):
        super().__init__()

        self.score_net = nn.Sequential(
            nn.Linear(d_in, d_attn),
            nn.Tanh(),
            nn.Linear(d_attn, 1),
        )

    def forward(self, x, mask):

        scores = self.score_net(x).squeeze(-1)

        mask = mask.bool()

        scores = scores.masked_fill(
            ~mask,
            -1e9,
        )

        attention = torch.softmax(
            scores,
            dim=-1,
        )

        attention = (
            attention
            * mask.to(attention.dtype)
        )

        attention = attention / (
            attention.sum(
                dim=-1,
                keepdim=True,
            )
            + 1e-8
        )

        pooled = torch.sum(
            attention.unsqueeze(-1) * x,
            dim=-2,
        )

        return pooled, attention


class HierarchicalDeepSetsEncoder(nn.Module):
    """
    Hierarchical DeepSets encoder with attention pooling.

    The input consists of a variable number of neutron stars,
    each of which can contain a variable number of observations.
    """

    def __init__(
        self,
        d_in=2,
        d_hidden=128,
        d_hidden_2=256,
        d_obs=128,
        d_ctx=128,
    ):
        super().__init__()

        # Encode individual observations.
        self.phi1 = MLP(
            d_in,
            d_hidden,
            d_hidden,
        )

        # Aggregate observations belonging to each neutron star.
        self.rho1 = MLP(
            2 * d_hidden,
            d_hidden,
            d_obs,
        )

        # Encode the neutron-star representation.
        self.phi2 = MLP(
            d_obs,
            d_hidden,
            d_hidden_2,
        )

        # Aggregate the neutron stars using attention.
        self.attn_pool = AttentionPool(
            d_hidden_2,
            d_attn=128,
        )

        # Produce the final context vector.
        self.rho2 = MLP(
            d_hidden_2,
            d_hidden_2,
            d_ctx,
        )

    def forward(self, x_padded, mask):

        # x_padded: (B, O, S, d_in)
        # mask:     (B, O, S)

        # -------------------------------------------------------------
        # Encode individual observations
        # -------------------------------------------------------------

        h = self.phi1(x_padded)
        # (B, O, S, d_hidden)

        # -------------------------------------------------------------
        # Aggregate observations within each neutron star
        # -------------------------------------------------------------

        h_obs_pre = masked_mean_second(
            h,
            mask.unsqueeze(-1),
            dim=2,
        )
        # (B, O, 2 * d_hidden)

        h_obs = self.rho1(h_obs_pre)
        # (B, O, d_obs)

        # Identify neutron stars with at least one valid observation.
        obs_mask = mask.any(dim=2)

        h_obs = (
            h_obs
            * obs_mask.unsqueeze(-1).to(h_obs.dtype)
        )

        # -------------------------------------------------------------
        # Encode neutron-star representations
        # -------------------------------------------------------------

        u = self.phi2(h_obs)
        # (B, O, d_hidden_2)

        # -------------------------------------------------------------
        # Attention pooling over neutron stars
        # -------------------------------------------------------------

        pooled, attention = self.attn_pool(
            u,
            obs_mask,
        )
        # (B, d_hidden_2)

        # -------------------------------------------------------------
        # Final context representation
        # -------------------------------------------------------------

        context = self.rho2(pooled)
        # (B, d_ctx)

        return context


# ============================================================================
# Initialize encoder
# ============================================================================

print("Using attention-based encoder.")

encoder = HierarchicalDeepSetsEncoder(
    d_in=2,
    d_hidden=128,
    d_hidden_2=256,
    d_obs=128,
    d_ctx=args.context,
).to(device)