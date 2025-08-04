import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm


class Chomp1d(nn.Module):

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class CausalConv1d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 dropout=0.0,
                 bias=True,
                 activation='ReLU'):
        super(CausalConv1d, self).__init__()

        self.padding = (kernel_size - 1) * dilation

        # PyTorch 2.0 or later version---> torch.nn.utils.parametrizations.weight_norm
        self.conv = weight_norm(nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,  # 先做 padding
            dilation=dilation,
            bias=bias
        ))

        self.chomp = Chomp1d(self.padding)

        self.activation = getattr(nn, activation)()

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.init_weights()

    def init_weights(self):
        self.conv.weight.data.normal_(0, 0.01)
        if self.conv.bias is not None:
            self.conv.bias.data.zero_()

    def forward(self, x):
        out = self.conv(x)

        out = self.chomp(out)

        out = self.activation(out)
        out = self.dropout(out)

        return out
