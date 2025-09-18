# python
# https://github.com/kostas1515/AGLU/blob/master/classification/cbam.py
#
import torch
import torch.nn as nn


class AGLU(nn.Module):
    __constants__ = ['num_parameters']
    num_parameters: int

    def __init__(self, num_parameters: int = 1, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_parameters = num_parameters
        super(AGLU, self).__init__()
        lambda_param = torch.nn.init.uniform_(torch.empty(self.num_parameters, **factory_kwargs))
        kappa_param = torch.nn.init.uniform_(torch.empty(self.num_parameters, **factory_kwargs))
        self.softplus = nn.Softplus(beta=-1)

        if num_parameters > 1:
            self.lambda_param = nn.Parameter(lambda_param.unsqueeze(0).unsqueeze(-1).unsqueeze(-1))
            self.kappa_param = nn.Parameter(kappa_param.unsqueeze(0).unsqueeze(-1).unsqueeze(-1))
        else:
            self.lambda_param = nn.Parameter(lambda_param)
            self.kappa_param = nn.Parameter(kappa_param)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        l = torch.clamp(self.lambda_param, min=0.0001)
        p = torch.exp((1/l) * self.softplus((self.kappa_param*input) - torch.log(l)))

        return p*input

    def extra_repr(self) -> str:
        return 'num_parameters={}'.format(self.num_parameters)




