import torch
import torch.nn as nn

from src.utils.logger_helper import add_member_variables_to_hparams_dict


# Class inheriting from nn.Module implementing the Barlow Twins loss
class BarlowTwinsLoss(nn.Module):
    def __init__(
        self,
        z_dim: int,
        effective_batch_size: int = None,
        corr_neg_one: bool = False,
    ):
        """Implementation of the Barlow Twins loss function. See paper for details:
        https://arxiv.org/abs/2103.03230 The argument corr_neg_one is used to determine whether the
        off-diagonal elements of the cross-correlation matrix should be penalized to be -1 or 0.
        The original implementation penalizes them to be 0, but this can lead to a degenerate
        solution where the cross-correlation matrix is all zeros. See this paper for details:
        https://arxiv.org/abs/2104.13712.

        Args:
            z_dim (int, optional): The dimensionality of the output of the projection head.
            effective_batch_size (int): The effective batch size, i.e. the batch size multiplied by the number of GPUs. Defaults to None -> gets inferred from the tensor (does not work with distributed training!)
            corr_neg_one (bool, optional): Whether to penalize the off-diagonal elements of the cross-correlation matrix to be -1. Defaults to False.
        """
        super().__init__()

        self.effective_batch_size = effective_batch_size
        self.z_dim = z_dim
        self.lambda_coeff = 1.0 / z_dim
        self.corr_neg_one = corr_neg_one

        self.bn = nn.BatchNorm1d(self.z_dim, affine=False)

    def save_hyperparameters(self, hparams_dict) -> None:
        add_member_variables_to_hparams_dict(
            hparams_dict,
            dict_params={
                "effective_batch_size": self.effective_batch_size,
                "z_dim": self.z_dim,
                "lambda_coeff": self.lambda_coeff,
                "corr_neg_one": self.corr_neg_one,
            },
            key_prefix="loss_global/",
            dict_additional_params={"name": "BarlowTwins"},
        )

    def off_diagonal_ele(self, x):
        # taken from: https://github.com/facebookresearch/barlowtwins/blob/main/main.py
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, z1, z2):
        # due to the linear fashion of the cross-correlation matrix, we can compute the the cross-correlation matrix
        # on each gpu independently and then sum them up afterwards
        # https://github.com/facebookresearch/barlowtwins/issues/33

        num_batches = z1.size(0)
        loss = torch.tensor(0.0, device=z1.device)

        for i in range(num_batches):
            z1_batch = z1[i]
            z2_batch = z2[i]

            # empirical cross-correlation matrix
            cross_corr = self.bn(z1_batch).T @ self.bn(z2_batch)

            effective_batch_size = (
                self.effective_batch_size if self.effective_batch_size is not None else z1_batch.size(0)
            )
            cross_corr.div_(effective_batch_size)

            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(cross_corr)

            on_diag = torch.diagonal(cross_corr).add_(-1).pow_(2).sum()

            if not self.corr_neg_one:
                off_diag = self.off_diagonal_ele(cross_corr).pow_(2).sum()
            else:
                off_diag = self.off_diagonal_ele(cross_corr).add_(1).pow_(2).sum()

            loss += on_diag + self.lambda_coeff * off_diag

        return {"value": loss / num_batches}
