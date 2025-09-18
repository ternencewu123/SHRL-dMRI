import torch.nn as nn
from nets.activations import AGLU
# -------------------------------
# decoder implemented by a simple MLP
# -------------------------------
class MLP(nn.Module):
    def __init__(self, in_dim=128 + 3, out_dim=1, depth=4, width=256):
        super(MLP, self).__init__()
        stage_one = []
        stage_two = []
        for i in range(depth):
            if i == 0:
                stage_one.append(nn.Linear(in_dim, width))
                stage_two.append(nn.Linear(in_dim, width))

                stage_one.append(nn.ReLU())
                stage_two.append(nn.ReLU())
            elif i == depth - 1:
                stage_one.append(nn.Linear(width, in_dim))
                stage_two.append(nn.Linear(width, out_dim))

                stage_one.append(nn.ReLU())
            else:
                stage_one.append(nn.Linear(width, width))
                stage_two.append(nn.Linear(width, width))

                stage_one.append(nn.ReLU())
                stage_two.append(nn.ReLU())

        self.stage_one = nn.Sequential(*stage_one)
        self.stage_two = nn.Sequential(*stage_two)

    def forward(self, x):
        h = self.stage_one(x)
        return self.stage_two(x + h)


# AGLU
class AGLU_MLP(nn.Module):
    def __init__(self, in_dim, out_dim, width):
        super(AGLU_MLP, self).__init__()
        self.mlp1 = nn.Linear(in_dim, width)
        self.aglu1 = AGLU(num_parameters=1)
        self.mlp2 = nn.Linear(width, width)
        self.aglu2 = AGLU(num_parameters=1)
        self.mlp3 = nn.Linear(width, width)
        self.aglu3 = AGLU(num_parameters=1)
        self.mlp4 = nn.Linear(width, in_dim)
        self.aglu4 = AGLU(num_parameters=1)

        self.mlp5 = nn.Linear(in_dim, width)
        self.aglu5 = AGLU(num_parameters=1)
        self.mlp6 = nn.Linear(width, width)
        self.aglu6 = AGLU(num_parameters=1)
        self.mlp7 = nn.Linear(width, width)
        self.aglu7 = AGLU(num_parameters=1)
        self.mlp8 = nn.Linear(width, out_dim)
        self.aglu8 = nn.ReLU()

    def forward(self, x):
        x1 = self.mlp1(x)
        a1 = self.aglu1(x1)
        x2 = self.mlp2(a1)
        a2 = self.aglu2(x2)
        x3 = self.mlp3(a2)
        a3 = self.aglu3(x3)
        x4 = self.mlp4(a3)
        a4 = self.aglu4(x4)

        x5 = self.mlp5(a4 + x)
        a5 = self.aglu5(x5)
        x6 = self.mlp6(a5)
        a6 = self.aglu6(x6)
        x7 = self.mlp7(a6)
        a7 = self.aglu7(x7)
        x8 = self.mlp8(a7)
        a8 = self.aglu8(x8)

        return a8


