import torch
from torch import nn


class EfficientNetModel(nn.Module):
    def __init__(self):
        from efficientnet_pytorch import EfficientNet
        super(EfficientNetModel, self).__init__()
        self.feat_extractor = EfficientNet.from_pretrained("efficientnet-b3", advprop=True)
        self.classifier = nn.Sequential(
            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 1)
        )

    def forward(self, x):
        x = self.feat_extractor(x)
        x = self.classifier(x)
        return x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class ConvMixerModel(nn.Module):
    def __init__(self, huge=False):
        super(ConvMixerModel, self).__init__()

        # in_shape = (3, 224, 224)
        h = 1536 if huge else 768
        depth = 32 if huge else 20
        p = 7
        k = 9 if huge else 7

        self.patch_embedding_net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=h, kernel_size=p, stride=p),
            nn.GELU(),
            nn.BatchNorm2d(h)
        )

        self.conv_mixer_layer = nn.Sequential(
            *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(h, h, kernel_size=k, groups=h, padding="same"),
                    nn.GELU(),
                    nn.BatchNorm2d(h)
                )),
                nn.Conv2d(h, h, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(h)
            ) for _ in range(depth)]
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(h, h // 2),
            nn.BatchNorm1d(h // 2),
            nn.ReLU(inplace=True),
            nn.Linear(h // 2, 1)
        )

    def forward(self, input_data):
        x = self.patch_embedding_net(input_data)
        x = self.conv_mixer_layer(x)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    m = EfficientNetModel()
    m.eval()
    a = torch.randn((1, 3, 224, 224))
    out = m(a)
    print()
