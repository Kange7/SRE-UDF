import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import snntorch as snn
from snntorch import surrogate
from models.embedder import get_embedder


class IntermediateSpikingLayer(nn.Module):
    """
    Intermediate Spiking Layer, ISL
    """

    def __init__(self, channels, num_steps=4, beta=0.9, threshold=0.5,
                 surrogate_slope=25):
        super().__init__()
        self.num_steps = num_steps
        self.lif = snn.Leaky(
            beta=beta,
            threshold=threshold,
            spike_grad=surrogate.fast_sigmoid(slope=surrogate_slope),
            learn_beta=True,
            learn_threshold=True,
            reset_mechanism="subtract",
        )
        self.proj = nn.Linear(channels, channels)
        nn.init.normal_(self.proj.weight, 0.0, 0.01)
        nn.init.zeros_(self.proj.bias)
        self.last_spike_rate = None

    def forward(self, x):
        # x: [B, C]
        feat = self.proj(x)
        mem = self.lif.init_leaky()
        spike_sum = torch.zeros_like(feat)

        for t in range(self.num_steps):
            spk, mem = self.lif(feat, mem)
            spike_sum = spike_sum + spk

        spike_rate = spike_sum / self.num_steps  # [B, C]
        self.last_spike_rate = spike_rate
        
        return x + spike_rate


class DualPathSpikingOutput(nn.Module):


    def __init__(self, in_features, num_steps=4, beta=0.9, threshold=0.5,
                 surrogate_slope=25, out_scale_init=0.05):
        super().__init__()
        self.num_steps = num_steps

        
        self.main = nn.Linear(in_features, 1)
        nn.init.normal_(self.main.weight, 0.0, 0.01)
        nn.init.constant_(self.main.bias, 0.0)
        self.out_scale = nn.Parameter(torch.tensor(float(out_scale_init)))

        self.lif_proj = nn.Linear(in_features, 1)
        nn.init.normal_(self.lif_proj.weight, 0.0, 0.01)
        nn.init.constant_(self.lif_proj.bias, 0.5)
        self.lif = snn.Leaky(
            beta=beta,
            threshold=threshold,
            spike_grad=surrogate.fast_sigmoid(slope=surrogate_slope),
            learn_beta=True,
            learn_threshold=True,
            reset_mechanism="subtract",
        )
        self.last_spike_rate = None

    def forward(self, x):
        
        out = torch.tanh(self.main(x)) * self.out_scale

        # ---- SNN spike_rate ----
        lif_in = self.lif_proj(x)
        mem = self.lif.init_leaky()
        spike_sum = torch.zeros(x.shape[0], 1, device=x.device)
        for _ in range(self.num_steps):
            spk, mem = self.lif(lif_in, mem)
            spike_sum = spike_sum + spk
        self.last_spike_rate = spike_sum / self.num_steps

        return out


class SREUDFNetwork(nn.Module):
    """
    ANN-SNN 
    """

    def __init__(self,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,
                 skip_in=(4,),
                 multires=0,
                 bias=0.5,
                 scale=1,
                 geometric_init=True,
                 weight_norm=True,
                 inside_outside=False,
                 use_snn=True,
                 snn_num_steps=4,
                 snn_beta=0.9,
                 snn_threshold=0.5,
                 snn_surrogate_slope=25,
                 snn_output_layers=2):
        super().__init__()

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn_fine = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn_fine = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale
        self.use_snn = use_snn

        
        if use_snn:
            self.ann_layers = self.num_layers - 1 - snn_output_layers
        else:
            self.ann_layers = self.num_layers - 1

        # ---- ANN  ----
        for l in range(0, self.ann_layers):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.ann_layers - 1:
                    if not inside_outside:
                        torch.nn.init.normal_(
                            lin.weight,
                            mean=np.sqrt(np.pi) / np.sqrt(dims[l]),
                            std=0.0001)
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(
                            lin.weight,
                            mean=-np.sqrt(np.pi) / np.sqrt(dims[l]),
                            std=0.0001)
                        torch.nn.init.constant_(lin.bias, bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(
                        lin.weight[:, :3], 0.0,
                        np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(
                        lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(
                        lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(
                        lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

       
        if use_snn:
            self.intermediate_snn = IntermediateSpikingLayer(
                channels=d_hidden,
                num_steps=snn_num_steps,
                beta=snn_beta,
                threshold=snn_threshold,
                surrogate_slope=snn_surrogate_slope,
            )
        else:
            self.intermediate_snn = None

        
        if use_snn:
            self.snn_output = DualPathSpikingOutput(
                in_features=d_hidden,
                num_steps=snn_num_steps,
                beta=snn_beta,
                threshold=snn_threshold,
                surrogate_slope=snn_surrogate_slope,
            )
        else:
            lin = nn.Linear(dims[self.ann_layers], d_out)
            if geometric_init:
                torch.nn.init.normal_(
                    lin.weight,
                    mean=np.sqrt(np.pi) / np.sqrt(dims[self.ann_layers]),
                    std=0.0001)
                torch.nn.init.constant_(lin.bias, -bias)
            if weight_norm:
                lin = nn.utils.weight_norm(lin)
            setattr(self, "lin" + str(self.ann_layers), lin)

        self.activation = nn.ReLU()

    def forward(self, inputs):
        inputs = inputs * self.scale
        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs)

        x = inputs

        for l in range(0, self.ann_layers):
            lin = getattr(self, "lin" + str(l))
            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)
            x = lin(x)
            if l < self.ann_layers - 1:
                x = self.activation(x)

        if self.use_snn and self.intermediate_snn is not None:
            x = self.intermediate_snn(x)

        if self.use_snn:
            res = self.snn_output(x)
        else:
            res = self.activation(x)
            res = getattr(self, "lin" + str(self.ann_layers))(res)
            res = torch.abs(res) / self.scale

        return res

    def udf(self, x):
        return self.forward(x)

    def udf_hidden_appearance(self, x):
        return self.forward(x)

    def sdf(self, x, index=None):
        """ sdf(x, index=..."""
        return self.forward(x)

    def gradient(self, x, index=None):

        x = x.detach().requires_grad_(True)
        y = self.udf(x)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)

    def get_intermediate_spike_rate(self):
        if self.use_snn and self.intermediate_snn is not None:
            return self.intermediate_snn.last_spike_rate
        return None