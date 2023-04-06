#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import numpy as np
import torch
from nnunet.network_architecture.custom_modules.conv_blocks import BasicResidualBlock, ResidualLayer
from nnunet.network_architecture.generic_UNet import Upsample
from nnunet.network_architecture.generic_modular_UNet import PlainConvUNetDecoder, get_default_network_config
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss
from torch import nn
from torch.optim import SGD
from torch.backends import cudnn
from torchvision import models


class ResNetEncoder(nn.Module):
    base_num_features = 3
    stage_output_features = [3, 64, 256, 512, 1024, 2048,]
    stage_pool_kernel_size = [[1,1], [4,4], [1,1], [2,2], [2,2], [2,2]]
    stage_conv_op_kernel_size = [[1,1], [7,7], [3,3], [3,3], [3,3], [3,3]]

    num_blocks_per_stage = [1, 1, 3, 4, 6, 3]  # decoder may need this

    def __init__(self, input_channels, default_return_skips=True, pretrained=False):
        """
        Following UNet building blocks can be added by utilizing the properties this class exposes (TODO)

        this one includes the bottleneck layer!

        :param input_channels:
        :param base_num_features:
        :param num_blocks_per_stage:
        :param feat_map_mul_on_downscale:
        :param pool_op_kernel_sizes:
        :param conv_kernel_sizes:
        :param props:
        """
        super(ResNetEncoder, self).__init__()

#         print('USING ResNet50 WEIGHTS FROM ImageNet')
        resnet = models.resnet50(pretrained=pretrained)

        self.channel_adapter = nn.Conv2d(input_channels, resnet.conv1.in_channels, [1,1])

        self.stages = [self.channel_adapter,
                       nn.Sequential(resnet.conv1,
                                     resnet.bn1,
                                     resnet.relu,
                                     resnet.maxpool,),]
        self.stages += [resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,]

        self.default_return_skips = default_return_skips


        self.stages = nn.ModuleList(self.stages)

    def forward(self, x, return_skips=None):
        """

        :param x:
        :param return_skips: if none then self.default_return_skips is used
        :return:
        """
        skips = []

        if return_skips is None:
            return_skips = self.default_return_skips

        for s in self.stages:
            x = s(x)
            if return_skips:
                skips.append(x)

        if return_skips:
            return skips
        else:
            return x

    @classmethod
    def compute_approx_vram_consumption(cls, patch_size, num_modalities,
                                        feat_map_mul_on_downscale, batch_size):
        npool = len(cls.stage_pool_kernel_size)

        current_shape = np.array(patch_size)

        tmp = (cls.num_blocks_per_stage[0] * 2 + 1) * np.prod(current_shape) * cls.base_num_features \
              + num_modalities * np.prod(current_shape)

        num_feat = cls.base_num_features
        for p in range(npool):
            current_shape = current_shape / np.array(cls.stage_pool_kernel_size[p])
            num_feat = cls.stage_output_features[p]
            num_convs = cls.num_blocks_per_stage[p] * 2 + 1  # + 1 for conv in skip in first block
            print(p, num_feat, num_convs, current_shape)
            tmp += num_convs * np.prod(current_shape) * num_feat
        return tmp * batch_size

# class ResNetDecoder(nn.Module):

class ResUNet(SegmentationNetwork):
    use_this_for_2D_configuration = 87404544.0
    default_blocks_per_stage_decoder = (1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
    default_min_batch_size = 2 # this is what works with the numbers above

    def __init__(self, input_channels, props, num_classes, num_blocks_per_stage_decoder,
                 deep_supervision=False, upscale_logits=False, initializer=None,
                 props_decoder=None, pretrained_resnet=False):
        super().__init__()
        
        assert props['conv_op'] == nn.Conv2d, 'only 2D ResUNet available'

        self.conv_op = props['conv_op']
        self.num_classes = num_classes

        self.encoder = ResNetEncoder(input_channels, default_return_skips=True, pretrained=pretrained_resnet)
        props['dropout_op_kwargs']['p'] = 0
        if props_decoder is None:
            props_decoder = props
        self.decoder = PlainConvUNetDecoder(self.encoder, num_classes, num_blocks_per_stage_decoder, props_decoder,
                                            deep_supervision, upscale_logits)
        if initializer is not None:
#             print('IGNORING INITIALIZER')
            self.apply(initializer)

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)
    
    @staticmethod
    def compute_approx_vram_consumption(patch_size, num_modalities, num_classes,
                                        num_conv_per_stage_decoder, feat_map_mul_on_downscale,
                                        batch_size):

        enc = ResNetEncoder.compute_approx_vram_consumption(patch_size, num_modalities, feat_map_mul_on_downscale,
                                                                  batch_size)
        dec = PlainConvUNetDecoder.compute_approx_vram_consumption(patch_size, ResNetEncoder.base_num_features,
                                                                   max(ResNetEncoder.stage_output_features),
                                                                   num_classes, ResNetEncoder.stage_pool_kernel_size,
                                                                   num_conv_per_stage_decoder,
                                                                   feat_map_mul_on_downscale, batch_size)

        return enc + dec


# def find_3d_configuration():
#     # lets compute a reference for 3D
#     # we select hyperparameters here so that we get approximately the same patch size as we would get with the
#     # regular unet. This is just my choice. You can do whatever you want
#     # These default hyperparemeters will then be used by the experiment planner

#     # since this is more parameter intensive than the UNet, we will test a configuration that has a lot of parameters
#     # herefore we copy the UNet configuration for Task005_Prostate
#     cudnn.deterministic = False
#     cudnn.benchmark = True

#     patch_size = (20, 320, 256)
#     max_num_features = 320
#     num_modalities = 2
#     num_classes = 3
#     batch_size = 2

#     # now we fiddle with the network specific hyperparameters until everything just barely fits into a titanx
#     blocks_per_stage_encoder = FabiansUNet.default_blocks_per_stage_encoder
#     blocks_per_stage_decoder = FabiansUNet.default_blocks_per_stage_decoder
#     initial_num_features = 32

#     # we neeed to add a [1, 1, 1] for the res unet because in this implementation all stages of the encoder can have a stride
#     pool_op_kernel_sizes = [[1, 1, 1],
#                             [1, 2, 2],
#                             [1, 2, 2],
#                             [2, 2, 2],
#                             [2, 2, 2],
#                             [1, 2, 2],
#                             [1, 2, 2]]

#     conv_op_kernel_sizes = [[1, 3, 3],
#                             [1, 3, 3],
#                             [3, 3, 3],
#                             [3, 3, 3],
#                             [3, 3, 3],
#                             [3, 3, 3],
#                             [3, 3, 3]]

#     unet = FabiansUNet(num_modalities, initial_num_features, blocks_per_stage_encoder[:len(conv_op_kernel_sizes)], 2,
#                        pool_op_kernel_sizes, conv_op_kernel_sizes,
#                        get_default_network_config(3, dropout_p=None), num_classes,
#                        blocks_per_stage_decoder[:len(conv_op_kernel_sizes)-1], False, False,
#                        max_features=max_num_features).cuda()

#     optimizer = SGD(unet.parameters(), lr=0.1, momentum=0.95)
#     loss = DC_and_CE_loss({'batch_dice': True, 'smooth': 1e-5, 'do_bg': False}, {})

#     dummy_input = torch.rand((batch_size, num_modalities, *patch_size)).cuda()
#     dummy_gt = (torch.rand((batch_size, 1, *patch_size)) * num_classes).round().clamp_(0, 2).cuda().long()

#     for _ in range(20):
#         optimizer.zero_grad()
#         skips = unet.encoder(dummy_input)
#         print([i.shape for i in skips])
#         output = unet.decoder(skips)

#         l = loss(output, dummy_gt)
#         l.backward()

#         optimizer.step()
#         if _ == 0:
#             torch.cuda.empty_cache()

#     # that should do. Now take the network hyperparameters and insert them in FabiansUNet.compute_approx_vram_consumption
#     # whatever number this spits out, save it to FabiansUNet.use_this_for_batch_size_computation_3D
#     print(FabiansUNet.compute_approx_vram_consumption(patch_size, initial_num_features, max_num_features, num_modalities,
#                                                 num_classes, pool_op_kernel_sizes,
#                                                 blocks_per_stage_encoder[:len(conv_op_kernel_sizes)],
#                                                 blocks_per_stage_decoder[:len(conv_op_kernel_sizes)-1], 2, batch_size))
#     # the output is 1230348800.0 for me
#     # I increment that number by 1 to allow this configuration be be chosen

def find_2d_configuration():
    # lets compute a reference for 2D
    # we select hyperparameters here so that we get approximately the same patch size as we would get with the
    # regular unet. This is just my choice. You can do whatever you want
    # These default hyperparemeters will then be used by the experiment planner

    # since this is more parameter intensive than the UNet, we will test a configuration that has a lot of parameters
    # herefore we copy the UNet configuration for BraTS2020 (kind of)
    cudnn.deterministic = False
    cudnn.benchmark = True

    patch_size = (96, 96)
    num_modalities = 4
    num_classes = 3
    batch_size = 32
    blocks_per_stage_encoder = ResNetEncoder.num_blocks_per_stage

    # now we fiddle with the network specific hyperparameters until everything just barely fits into a titanx
    # blocks_per_stage_decoder = ResUNet.default_blocks_per_stage_decoder
    n = 8
    blocks_per_stage_decoder = (n,n,n,n,n,n,n,n,n,n,n)

    unet = ResUNet(num_modalities, get_default_network_config(2, dropout_p=None), num_classes,
                   blocks_per_stage_decoder[:len(ResNetEncoder.stage_conv_op_kernel_size)-1],
                   deep_supervision=False, upscale_logits=False, initializer=None,
                   props_decoder=None, pretrained_resnet=False).cuda()

    optimizer = SGD(unet.parameters(), lr=0.1, momentum=0.95)
    loss = DC_and_CE_loss({'batch_dice': True, 'smooth': 1e-5, 'do_bg': False}, {})

    dummy_input = torch.rand((batch_size, num_modalities, *patch_size)).cuda()
    dummy_gt = (torch.rand((batch_size, 1, *patch_size)) * num_classes).round().clamp_(0, 2).cuda().long()

    for _ in range(20):
        optimizer.zero_grad()
        skips = unet.encoder(dummy_input)
        print([i.shape for i in skips])
        output = unet.decoder(skips)

        l = loss(output, dummy_gt)
        l.backward()

        optimizer.step()
        if _ == 0:
            torch.cuda.empty_cache()

    # that should do. Now take the network hyperparameters and insert them in FabiansUNet.compute_approx_vram_consumption
    # whatever number this spits out, save it to FabiansUNet.use_this_for_batch_size_computation_2D
    print(ResUNet.compute_approx_vram_consumption(patch_size, num_modalities, num_classes,
                                                  blocks_per_stage_decoder[:len(ResNetEncoder.stage_conv_op_kernel_size)-1],
                                                  2, batch_size))
    # the output is 87404544.0 for me
    # I increment that number by 1 to allow this configuration be be chosen
    # This will not fit with 32 filters, but so will the regular U-net. We still use 32 filters in training.
    # This does not matter because we are using mixed precision training now, so a rough memory approximation is OK

if __name__ == "__main__":
    find_2d_configuration()
