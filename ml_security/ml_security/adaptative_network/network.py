"""
Hybrid Linear KAN Layer

This module implements the Hybrid Linear KAN layer, which is a linear layer with a
Kolmogorov-Arnold representation. The layer is defined by the following equation:

    y = (W_spline * B_spline(x)) * W_linear.T

where:
    - y is the output tensor,
    - W_spline is the spline weight tensor,
    - B_spline(x) is the B-spline bases tensor,
    - W_linear is the linear weight tensor,
    - * denotes the tensor dot product.


The idea is to use Kolmogorov-Arnold representation to approximate the function and 
iteractively refine the approximation by adjusting the spline weights. The B-spline
bases are computed from the input tensor x, and the spline weights are learned by
interpolating the input-output pairs.
This way, we can both approximate the "activation function" and the "weight matrix"
of a linear layer.
Hopefully, this will give more expressive power to the model and make it more robust
to adversarial attacks in the future.

Greatly inspired by the paper "Kolmogorov-Arnold Representation Neural Networks"
and by https://github.com/Blealtan/efficient-kan implementation.
"""

import math

import torch
import torch.nn.functional as F


class HybridLinearKAN(torch.nn.Module):
    """
    Hybrid Linear KAN layer.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        grid_size (int): Number of grid points.
        spline_order (int): Order of the B-spline bases.
        scale_noise (float): Scale of the noise added to the spline weights.
        scale_base (float): Scale of the base weight initialization.
        scale_spline (float): Scale of the spline weight initialization.
        base_activation (torch.nn.Module): Base activation function.
        grid_eps (float): Grid epsilon.
        grid_range (list): Grid range.

    Attributes:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        grid_size (int): Number of grid points.
        spline_order (int): Order of the B-spline bases.
        grid (torch.Tensor): Grid tensor.
        base_weight (torch.nn.Parameter): Base weight tensor.
        spline_weight (torch.nn.Parameter): Spline weight tensor.
        linear_weight (torch.nn.Parameter): Linear weight tensor.
        spline_scaler (torch.nn.Parameter): Spline scaler tensor.
        scale_noise (float): Scale of the noise added to the spline weights.
        scale_base (float): Scale of the base weight initialization.
        scale_spline (float): Scale of the spline weight initialization.
        base_activation (torch.nn.Module): Base activation function.
        grid_eps (float): Grid epsilon.
    """

    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.01,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.LeakyReLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(HybridLinearKAN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.grid = torch.nn.Parameter(grid)  # Making grid learnable

        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        self.linear_weight = torch.nn.Parameter(
            torch.Tensor(out_features, out_features)
        )
        torch.nn.init.xavier_uniform_(self.linear_weight)

        self.spline_scaler = torch.nn.Parameter(torch.Tensor(out_features, in_features))

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset the parameters of the layer (except the linear weight) to their initial values.
        """
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                self.scale_spline
                * self._curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            torch.nn.init.kaiming_uniform_(
                self.spline_scaler, a=math.sqrt(5) * self.scale_spline
            )

    def _b_splines(self, x: torch.Tensor):
        """
        Computes the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def _curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Computes the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        # (in_features, batch_size, grid_size + spline_order)
        A = self._b_splines(x).transpose(0, 1)
        # (in_features, batch_size, out_features)
        B = y.transpose(0, 1)
        # (in_features, grid_size + spline_order, out_features)
        solution = torch.linalg.lstsq(A, B).solution
        # (out_features, in_features, grid_size + spline_order)
        result = solution.permute(2, 0, 1)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (self.spline_scaler.unsqueeze(-1))

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        spline_output = F.linear(
            self._b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = F.linear(spline_output, self.linear_weight.T)
        output = output.reshape(*original_shape[:-1], self.out_features)
        return output
