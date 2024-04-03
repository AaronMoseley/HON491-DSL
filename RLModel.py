from torch import nn
from torch.nn import functional as F
import torch
import torchvision.models
from typing import Type, Union, Optional, Callable, List 
import numpy as np
import MinMaxAI
import copy

class WarGamesAI(nn.Module):
    def __init__(self, device) -> None:
        super().__init__()

        self.device = device
        self.unet = ResUNet().to(device)
        self.finalEncoder = ResNetEncoder(inlayers=7, block=BasicBlock, layers=[2, 2, 2, 2], num_classes=2).to(device)

    def forward(self, levelStates, currPlayer, outputMove=True):
        modelInput = None
        for levelState in levelStates:
            newState = copy.deepcopy(levelState)

            newState = np.asarray(newState)
            newState = np.pad(newState, ((0, max(0, 8 - len(newState))), (0, max(0, 8 - len(newState[0])))), mode="constant", constant_values=(0))

            signs = torch.mul(torch.from_numpy(np.sign(newState)), currPlayer).to(self.device)
            absVal = np.absolute(newState)

            levelStateInput = torch.from_numpy((absVal >= 1).astype(int)).to(self.device).unsqueeze(dim=0)

            for i in range(6):
                if i in [0, 3]:
                    continue

                levelStateInput = torch.cat((levelStateInput, torch.mul(torch.from_numpy((absVal == i + 2).astype(int)).to(self.device), signs).unsqueeze(dim=0)))

            if modelInput == None:
                modelInput = levelStateInput.unsqueeze(0)
            else:
                modelInput = torch.cat((modelInput, levelStateInput.unsqueeze(0)))

        unetOutput = self.unet(modelInput.float())

        """
        finalInput = None
        for i, batchEl in enumerate(unetOutput):
            if finalInput == None:
                finalInput = torch.cat((batchEl, modelInput[i]), dim=0).unsqueeze(0)
            else:
                finalInput = torch.cat((finalInput, torch.cat((batchEl, modelInput[i]), dim=0).unsqueeze(0)))

        finalOutput = self.finalEncoder(finalInput)[0].detach().cpu().tolist()
        """

        result = None

        for i, newState in enumerate(levelStates):
            validMoves = MinMaxAI.getValidMoves(newState, currPlayer)
            #priority = np.argsort(finalOutput[i])

            if len(validMoves) == 0:
                if result == None:
                    if outputMove:
                        result = [None]
                    else:
                        result = torch.zeros(1).to(self.device)
                else:
                    if outputMove:
                        result.append(None)
                    else:
                        result = torch.cat((result, torch.zeros(1).to(self.device)))

                continue

            #for piece in reversed(priority):
                #maxPriority = -float("inf")
                #maxInd = -1

            maxPriority = -float("inf")
            maxSecondaryPriority = -float("inf")
            maxInd = -1

            for j, move in enumerate(validMoves):
                #if (piece + 6) * currPlayer != newState[move[0][0]][move[0][1]]:
                #    continue

                if unetOutput[i][0][move[1][0]][move[1][1]] > maxPriority:
                    maxPriority = unetOutput[i][0][move[1][0]][move[1][1]]
                    maxSecondaryPriority = unetOutput[i][0][move[0][0]][move[0][1]]
                    maxInd = j
                elif unetOutput[i][0][move[1][0]][move[1][1]] == maxPriority:
                    if unetOutput[i][0][move[0][0]][move[0][1]] > maxSecondaryPriority:
                        maxInd = j
                        maxSecondaryPriority = unetOutput[i][0][move[0][0]][move[0][1]]

            if maxInd >= 0:
                if outputMove:
                    if result == None:
                        result = [validMoves[maxInd]]
                    else:
                        result.append(validMoves[maxInd])
                else:
                    if result == None:
                        result = torch.Tensor(unetOutput[i][0][validMoves[maxInd][1][0]][validMoves[maxInd][1][1]]).unsqueeze(0).to(self.device)
                    else:
                        result = torch.cat((result, torch.Tensor(unetOutput[i][0][validMoves[maxInd][1][0]][validMoves[maxInd][1][1]]).unsqueeze(0))).to(self.device)

        return result

class ResUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.encoder = ResNetEncoder(inlayers=5, block=BasicBlock, layers=[2, 2, 2, 2], num_classes=2)
        self.decoder = ResDecoder(outlayers=1)

    def forward(self, x):
        _, conv5, conv4, conv3, conv2, conv1 = self.encoder(x)

        outSeg = self.decoder(x, conv5, conv4, conv3, conv2, conv1)

        return outSeg

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.InstanceNorm2d(out_channels),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.InstanceNorm2d(in_channels),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Dropout(0.25)
    ) 

class ResDecoder(nn.Module):
    def __init__(self, outlayers, block=double_conv):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up4 = block(64 + 64, 32)
        self.dconv_up3 = block(64 + 32, 32)
        self.dconv_up2 = block(64 + 32, 32)
        self.dconv_up1 = block(32 + 32, 32)
        self.conv_last = nn.Conv2d(32, outlayers, 1)

        self.sigm = nn.Sigmoid()
        
        
    def forward(self, input, conv5, conv4, conv3, conv2, conv1):
        x = torch.cat([conv5, conv4], dim=1)

        x = self.dconv_up4(x)     
        x = torch.cat([x, conv3], dim=1)       

        x = self.dconv_up3(x)  
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1) 

        x = self.dconv_up1(x)

        x = self.upsample(x)

        out = self.conv_last(x)
        
        return out
    
#From here to line 532, all code was taken from the torchvision ResNet implementation at https://pytorch.org/vision/main/_modules/torchvision/models/resnet.html
#The only changes made are modifying the number of input channels from 3 to 1 for CT slices and returning the output from all blocks in the forward function
def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNetEncoder(nn.Module):
    def __init__(
        self,
        inlayers,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 32
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(inlayers, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_stagnant_layer(block, 64, layers[1], dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_stagnant_layer(block, 64, layers[2], dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_stagnant_layer(block, 64, layers[3], dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Linear(64 * block.expansion, num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]
    
    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)
    
    def _make_stagnant_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, self.inplanes, stride),
                norm_layer(planes),
            )

        layers = []
        layers.append(
            block(
                planes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )

        for _ in range(1, blocks):
            layers.append(
                block(
                    planes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor, useSigm=True, getConv=True) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        out1 = self.relu(x)
        x = self.maxpool(out1)

        out2 = self.layer1(x)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)

        x = self.avgpool(out5)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        if useSigm:
            sigm = nn.Sigmoid()
            x = sigm(x)

        if getConv:
            return x, out5, out4, out3, out2, out1

        return x