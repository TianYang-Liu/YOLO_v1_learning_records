import torch
import torch.nn as nn

# 卷积网络参数
architecture_config = [
    # Tuple:(kernel_size, num_filters, stride, padding)
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (5, 512, 1, 1),
    "M",
    # List:[tuple,tuple,num_repeats]
    [(1, 256, 1, 0), (3, 5121, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1)
]


# 定义卷积模块
class CNNblock(nn.Module):
    '''卷积模块nn.Conv2d(), nn.BatchNorm2d(), nn.LeakyReLU()。
    需要定义输入输出通道等参数'''

    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNblock, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, bias=False, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        out = self.network(x)
        return out


# 定义Yolov1网络架构
class Yolov1(nn.Module):
    "定义输入的通道数、darknet、全连接层"

    def __init__(self, in_channels=3, **kwargs):
        super(Yolov1, self).__init__()
        self.architecture_config = architecture_config
        self.in_channels = in_channels
        # 主要由darknet和fc组成。
        self.darknet = self._create_conv_layers(self.architecture_config)
        self.fcs = self._create_fcs(**kwargs)

    def forward(self, x):
        conv_out = self.darknet(x)
        fc_out = self.fcs(conv_out)
        return fc_out

    # define darknet
    def _create_conv_layers(self, architecture_config):
        layers = []
        in_channels = self.in_channels

        for config in architecture_config:
            if type(config) == tuple:
                layers += [
                    CNNblock(in_channels=in_channels, out_channels=config[1], kernel_size=config[0], stride=config[2],
                             padding=config[3])]
                in_channels = config[1]

            elif type(config) == str:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

            elif type(config) == list:
                conv1 = config[0]
                conv2 = config[1]
                num_repeats = config[2]

                for _ in range(num_repeats):
                    layers += [CNNblock(in_channels=in_channels, out_channels=conv1[1], kernel_size=conv1[0],
                                        stride=conv1[2], padding=conv1[3])]
                    layers += [CNNblock(in_channels=conv1[1], out_channels=conv2[1], kernel_size=conv2[0],
                                        stride=conv2[2], padding=conv2[3])]
                    in_channels = conv2[1]

        # *代表拆包，将可迭代类型（列表）拆包送入nn.Sequential
        network = nn.Sequential(*layers)
        return network

    # 定义全连接层
    def _create_fcs(self, split_size, num_boxes, num_classes):
        '''全连接层包括nn.Linear(),nn.Dropout(),nn.LeakyReLU()'''
        S, B, C = split_size, num_boxes, num_classes
        out = nn.Sequential(
            nn.Flatten(), # flatten from dim=1, flatten之后的形状：(batch_size, num_channelsxheightxwidth)
            nn.Linear(1024 * S * S, 496),  # 原文是4096
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(496, S * S * (C + B * 5)) # （S,S,30) where C+B*5=30
        )
        return out


def test(S=7, B=2, C=20):
    model = Yolov1(split_size=S, num_boxes=B, num_classes=C)
    x = torch.randn((2, 3, 448, 448))  # input shape:(batch_size, channels, height, width)
    print(model(x).size())


test()
