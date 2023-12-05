import torch.nn as nn

from src.utils.logger_helper import add_member_variables_to_hparams_dict


class LinearProjectionHead(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim_1=2048, hidden_dim_2=0, output_dim=128, bias_hidden_layers=True):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2
        self.output_dim = output_dim
        self.bias_hidden_layers = bias_hidden_layers

        if hidden_dim_2 == 0:
            self.projection_head = nn.Sequential(
                nn.Linear(input_dim, hidden_dim_1, bias=bias_hidden_layers),
                # nn.BatchNorm1d(hidden_dim_1),
                nn.ReLU(),
                nn.Linear(hidden_dim_1, output_dim, bias=False),
            )
        else:
            self.projection_head = nn.Sequential(
                nn.Linear(input_dim, hidden_dim_1, bias=bias_hidden_layers),
                # nn.BatchNorm1d(hidden_dim_1),
                nn.ReLU(),
                nn.Linear(hidden_dim_1, hidden_dim_2, bias=bias_hidden_layers),
                # nn.BatchNorm1d(hidden_dim_2),
                nn.ReLU(),
                nn.Linear(hidden_dim_2, output_dim, bias=False),
            )

    def forward(self, x):
        if x.ndim == 2:
            return self.projection_head(x)

        elif x.ndim in [3, 5]:
            batch_size, num_patches, _ = x.shape
            x_proj = self.projection_head(x.flatten(start_dim=0, end_dim=1))
            x_proj = x_proj.view(batch_size, num_patches, -1)

            return x_proj
        else:
            raise ValueError("The input tensor has an unsupported number of dimensions.")

    def save_hyperparameters(self, hparams_dict, name) -> None:
        add_member_variables_to_hparams_dict(
            hparams_dict,
            dict_params={
                "input_dim": self.input_dim,
                "hidden_dim_1": self.hidden_dim_1,
                "hidden_dim_2": self.hidden_dim_2,
                "output_dim": self.output_dim,
                "bias_hidden_layers": self.bias_hidden_layers,
            },
            key_prefix=f"{name}/",
        )
