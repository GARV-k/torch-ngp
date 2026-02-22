import torch
import torch.nn as nn
import torch.nn.functional as F
class MeshSDFDecoder(nn.Module):
    def __init__(self,
                #  encoding="hashgrid",
                 num_layers=8,
                 skips=[],
                 hidden_dim=128,
                 clip_sdf=None,
                 in_dim = 32,
                 out_dim = 1,
                 ):
        super().__init__()


        self.num_layers = num_layers
        self.skips = skips
        self.hidden_dim = hidden_dim
        self.clip_sdf = clip_sdf

        self.in_dim = in_dim
        self.out_dim = out_dim

        backbone = []

        for l in range(num_layers):
            if l == 0:
                in_dim = self.in_dim
            elif l in self.skips:
                in_dim = self.hidden_dim + self.in_dim
            else:
                in_dim = self.hidden_dim
            
            if l == num_layers - 1:
                out_dim = 1
            else:
                out_dim = self.hidden_dim
            
            backbone.append(nn.Linear(in_dim, out_dim, bias=False))

        self.backbone = nn.ModuleList(backbone)

    
    def forward(self, h):
        
        for l in range(self.num_layers):
            if l in self.skips:
                h = torch.cat([h, x], dim=-1)
            h = self.backbone[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        if self.clip_sdf is not None:
            h = h.clamp(-self.clip_sdf, self.clip_sdf)

        return h