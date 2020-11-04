import torch
import torch.nn as nn

from neat_eo.core import load_module


class ConvRelu(nn.Module):
    """3x3 convolution followed by ReLU activation building block."""

    def __init__(self, num_in, num_out):
        super().__init__()
        self.block = nn.Conv2d(num_in, num_out, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        return nn.functional.relu(self.block(x), inplace=True)


class DecoderBlock(nn.Module):
    """Decoder building block upsampling resolution by a factor of two."""

    def __init__(self, num_in, num_out):
        super().__init__()
        self.block = ConvRelu(num_in, num_out)

    def forward(self, x):
        return self.block(nn.functional.interpolate(x, scale_factor=2, mode="nearest"))


class Albunet(nn.Module):
    def __init__(self, shape_in, shape_out, encoder="resnet50", train_config=None):
        super().__init__()

        doc = "U-Net like encoder-decoder architecture with a ResNet, ResNext or WideResNet encoder.\n\n"
        doc += " - https://arxiv.org/abs/1505.04597 - U-Net: Convolutional Networks for Biomedical Image Segmentation\n"

        if encoder in ["resnet50", "resnet101", "resnet152"]:
            doc += " - https://arxiv.org/abs/1512.03385 - Deep Residual Learning for Image Recognition\n"
        elif encoder in ["resnext50_32x4d", "resnext101_32x8d"]:
            doc += " - https://arxiv.org/pdf/1611.05431 - Aggregated Residual Transformations for DNN\n"
        elif encoder in ["wide_resnet50_2", "wide_resnet101_2"]:
            doc += " - https://arxiv.org/abs/1605.07146 - Wide Residual Networks\n"
        else:
            encoders = "resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d, wide_resnet50_2, wide_resnet101_2"
            assert False, "Albunet, expects as encoder: " + encoders

        self.version = 2
        self.doc_string = doc

        num_filters = 32
        num_channels = shape_in[0]
        num_classes = shape_out[0]

        assert num_channels, "Empty Channels"
        assert num_classes, "Empty Classes"

        try:
            pretrained = train_config["pretrained"]
        except:
            pretrained = False

        models = load_module("torchvision.models")
        self.encoder = getattr(models, encoder)(pretrained=pretrained)
        # https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

        if num_channels != 3:
            weights = nn.init.kaiming_normal_(torch.zeros((64, num_channels, 7, 7)), mode="fan_out", nonlinearity="relu")
            if pretrained:
                for c in range(min(num_channels, 3)):
                    weights.data[:, c, :, :] = self.encoder.conv1.weight.data[:, c, :, :]
            self.encoder.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.encoder.conv1.weight = nn.Parameter(weights)

        self.center = DecoderBlock(2048, num_filters * 8)

        self.dec0 = DecoderBlock(2048 + num_filters * 8, num_filters * 8)
        self.dec1 = DecoderBlock(1024 + num_filters * 8, num_filters * 8)
        self.dec2 = DecoderBlock(512 + num_filters * 8, num_filters * 2)
        self.dec3 = DecoderBlock(256 + num_filters * 2, num_filters * 2 * 2)
        self.dec4 = DecoderBlock(num_filters * 2 * 2, num_filters)
        self.dec5 = ConvRelu(num_filters, num_filters)

        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):

        enc0 = self.encoder.conv1(x)
        enc0 = self.encoder.bn1(enc0)
        enc0 = self.encoder.relu(enc0)
        enc0 = self.encoder.maxpool(enc0)

        enc1 = self.encoder.layer1(enc0)
        enc2 = self.encoder.layer2(enc1)
        enc3 = self.encoder.layer3(enc2)
        enc4 = self.encoder.layer4(enc3)

        center = self.center(nn.functional.max_pool2d(enc4, kernel_size=2, stride=2))

        dec0 = self.dec0(torch.cat([enc4, center], dim=1))
        dec1 = self.dec1(torch.cat([enc3, dec0], dim=1))
        dec2 = self.dec2(torch.cat([enc2, dec1], dim=1))
        dec3 = self.dec3(torch.cat([enc1, dec2], dim=1))
        dec4 = self.dec4(dec3)
        dec5 = self.dec5(dec4)

        return self.final(dec5)
