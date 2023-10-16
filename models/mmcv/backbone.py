import torch
import torch.nn as nn

from models.mmcv.cnn import ConvModule

class ShallowCNN(nn.Module):
    """Implement Shallow CNN block for SATRN.

    SATRN: `On Recognizing Texts of Arbitrary Shapes with 2D Self-Attention
    <https://arxiv.org/pdf/1910.04396.pdf>`_.

    Args:
        base_channels (int): Number of channels of input image tensor
            :math:`D_i`.
        hidden_dim (int): Size of hidden layers of the model :math:`D_m`.
        init_cfg (dict or list[dict], optional): Initialization configs.
    """

    def __init__(self,
                 input_channels=3,
                 hidden_dim=512,
                ):
        super().__init__()
        assert isinstance(input_channels, int)
        assert isinstance(hidden_dim, int)

        self.conv1 = ConvModule(
            input_channels,
            hidden_dim // 2,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'))
        self.conv2 = ConvModule(
            hidden_dim // 2,
            hidden_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'))
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input image feature :math:`(N, D_i, H, W)`.

        Returns:
            Tensor: A tensor of shape :math:`(N, D_m, H/4, W/4)`.
        """

        x = self.conv1(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.pool(x)

        return x
    

if __name__ == '__main__':
    model = ShallowCNN()
    x = torch.randn(1, 3, 32, 100)
    y = model(x)
    print(y.shape)
    print(f'num params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')