import torch
import torch.nn as nn

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

# from .utils import load_state_dict_from_url

from ib_layers import *

# __all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
#            'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
#            'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

resnet_alpha = 1.0



def conv3x3(in_planes, out_planes, wib, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    # resnet_wib = False
    resnet_wib = True
    resnet_alpha = 1E-3
    if not wib:
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

    else:
        return WibConv2d(alpha=resnet_alpha,
                       in_channels=in_planes, out_channels=out_planes, kernel_size=3, stride=stride,
                       padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, wib, stride=1):
    """1x1 convolution"""
    # resnet_wib = False
    resnet_wib = True
    resnet_alpha = 1E-3
    if not wib:
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
    else:
        return WibConv2d(alpha=resnet_alpha,
                       in_channels=in_planes, out_channels=out_planes, kernel_size=1, stride=stride, bias=False)


cfg = {

#####    括号里第一个数是卷积核个数；第二个数是卷积结构，1表示正常卷基层，2表示resnet的2层block结构，3表示resnet
#####    的三层Bottleneck结构；第三个数表示如果是resnet结构，第一个卷积层的stride


    #resnet18 (2,2,2,2)
    'G5': [(64, 1, 1, 1.0/32),                    ## InformationBottleneck
           'M',
           (64, 2, 1, 1.0/32),                    ## InformationBottleneck
           (64, 2, 1, 1.0/32),                    ## InformationBottleneck
           (128, 2, 2, 1.0/16),                   ## InformationBottleneck
           (128, 2, 1, 1.0/16),                   ## InformationBottleneck
           (256, 2, 2, 1.0/8),                    ## InformationBottleneck
           (256, 2, 1, 1.0/8),                    ## InformationBottleneck
           (512, 2, 2, 1.0/4),                    ## InformationBottleneck
           (512, 2, 1, 1.0/4),                    ## InformationBottleneck
           'A'],

    #resnet34 (3,4,6,3)
    'G1': [(64, 1, 1, 1.0/32),                    ## InformationBottleneck
           'M',
           (64, 2, 1, 1.0/32),                    ## InformationBottleneck
           (64, 2, 1, 1.0/32),                    ## InformationBottleneck
           (64, 2, 1, 1.0/32),                    ## InformationBottleneck
           (128, 2, 2, 1.0/16),                   ## InformationBottleneck
           (128, 2, 1, 1.0/16),                   ## InformationBottleneck
           (128, 2, 1, 1.0/16),                   ## InformationBottleneck
           (128, 2, 1, 1.0/16),                   ## InformationBottleneck
           (256, 2, 2, 1.0/8),                    ## InformationBottleneck
           (256, 2, 1, 1.0/8),                    ## InformationBottleneck
           (256, 2, 1, 1.0/8),                    ## InformationBottleneck
           (256, 2, 1, 1.0/8),                    ## InformationBottleneck
           (256, 2, 1, 1.0/8),                    ## InformationBottleneck
           (256, 2, 1, 1.0/8),                    ## InformationBottleneck
           (512, 2, 2, 1.0/4),                    ## InformationBottleneck
           (512, 2, 1, 1.0/4),                    ## InformationBottleneck
           (512, 2, 1, 1.0/4),                    ## InformationBottleneck
           'A'],

    # resnet50 (3,4,6,3)
    'G2': [(64, 1, 1, 1.0 / 32),  ## InformationBottleneck
           'M',
           (64, 3, 1, 1.0 / 32),  ## InformationBottleneck
           (64, 3, 1, 1.0 / 32),  ## InformationBottleneck
           (64, 3, 1, 1.0 / 32),  ## InformationBottleneck
           (128, 3, 2, 1.0 / 16),  ## InformationBottleneck
           (128, 3, 1, 1.0 / 16),  ## InformationBottleneck
           (128, 3, 1, 1.0 / 16),  ## InformationBottleneck
           (128, 3, 1, 1.0 / 16),  ## InformationBottleneck
           (256, 3, 2, 1.0 / 8),  ## InformationBottleneck
           (256, 3, 1, 1.0 / 8),  ## InformationBottleneck
           (256, 3, 1, 1.0 / 8),  ## InformationBottleneck
           (256, 3, 1, 1.0 / 8),  ## InformationBottleneck
           (256, 3, 1, 1.0 / 8),  ## InformationBottleneck
           (256, 3, 1, 1.0 / 8),  ## InformationBottleneck
           (512, 3, 2, 1.0 / 4),  ## InformationBottleneck
           (512, 3, 1, 1.0 / 4),  ## InformationBottleneck
           (512, 3, 1, 1.0 / 4),  ## InformationBottleneck
           'A']

}

def reparameterize(mu, logalpha):
    std = logalpha.mul(0.5).exp_()
    eps = torch.FloatTensor(std.size(0)).cuda(mu.get_device()).normal_()
    eps = Variable(eps)

    # phi = std * eps - std * std / 2
    # return phi

    phi = (std * eps - std * std / 2).exp_()
    return phi * mu

    # std = logalpha.mul(0.5).exp_()
    # eps = torch.FloatTensor(std.size(0)).cuda(mu.get_device()).normal_()
    # eps = Variable(eps)
    # return mu + eps * std


class WeightIB(nn.Module):
    def __init__(self, out_channels, init_mag=9, init_var=0.01):
        super(WeightIB, self).__init__()
        self.dim = out_channels
        print(self.dim)
        # self.phi = Parameter(torch.Tensor(self.dim))
        self.logalpha = Parameter(torch.Tensor(self.dim))
        self.mu = Parameter(torch.Tensor(self.dim))
        self.epsilon = 1e-8
        self.offset = 0.00

        self.mu.data.normal_(1, init_var)
        self.logalpha.data.normal_(-init_mag, init_var)

    def forward(self, x, training=False):
        if self.training:
            # z_scale = reparameterize(self.mu, self.logalpha)
            # z_scale_exp = z_scale.exp_()

            # hard_mask, _ = self.get_mask_hard(self.epsilon)
            # z_scale = z_scale_exp * Variable(hard_mask)

            z_scale = reparameterize(self.mu, self.logalpha)

            hard_mask, _ = self.get_mask_hard(self.epsilon)
            z_scale *= Variable(hard_mask)

            # print('self.mu: ', self.mu)
            # print('z_scale1: ', z_scale)
            # print('z_scale1: ', z_scale)
        else:
            # z_scale = reparameterize(self.mu, self.logalpha)
            # z_scale_exp = z_scale.exp_()

            z_scale = reparameterize(self.mu, self.logalpha)

            hard_mask, _ = self.get_mask_hard(self.epsilon)
            z_scale *= Variable(hard_mask)

            # z_scale = Variable(self.get_mask_weighted(self.epsilon))
            # print('z_scale2: ', z_scale)

        # new_shape = self.adapt_shape(z_scale_exp.size(), x.size())
        # return x * z_scale_exp.view(new_shape)

        new_shape = self.adapt_shape(z_scale.size(), x.size())
        return x * z_scale.view(new_shape)


    def adapt_shape(self, src_shape, x_shape):
        if len(src_shape) == 2:
            new_shape = src_shape
            # print('new_shape1: ',new_shape)
        else:
            new_shape = (src_shape[0], 1)
            # print('new_shape2: ', new_shape)
        if len(x_shape)>2:
            new_shape = list(new_shape)
            new_shape += [1 for i in range(len(x_shape)-2)]
            # print('new_shape3: ', new_shape)
        return new_shape

    def get_mask_hard(self, threshold=0):
        hard_mask = (self.mu.abs() > threshold).float()
        prune = self.mu.abs().cpu() > threshold  # e.g. [True, False, True, True, False]
        mask = np.where(prune)[0]  # e.g. [0, 2, 3]
        return hard_mask, len(mask)

    def get_mask_weighted(self, threshold=0):
        mask = (self.mu.abs() > threshold).float() * self.mu.data.float()
        return mask

    def compute_Wib_upbound(self, logalpha):
        return - 0.5 * logalpha.sum()


class WibConv2d(nn.Conv2d):
    def __init__(self, alpha, **kwargs):
        super(WibConv2d, self).__init__(**kwargs)
        self.alpha = alpha
        self.weight_ib = WeightIB(self.out_channels)
        self.W = torch.empty(self.weight.data.size())
        torch.nn.init.xavier_normal(self.W, gain=1)

    def forward(self, x):
        if self.training:
            # kernel_in = self.weight.data

            # self.W.data = self.weight_ib(self.weight, training=self.training)
            # y = nn.functional.conv2d(x, self.W, self.bias, self.stride, self.padding, self.dilation, self.groups)

            new_weight = self.weight_ib(self.weight, training=self.training)
            y = nn.functional.conv2d(x, new_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

            # y = nn.functional.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

            # self.W.data= self.W
        else:

            # y = nn.functional.conv2d(x, self.W, self.bias, self.stride, self.padding, self.dilation, self.groups)

            new_weight = self.weight_ib(self.weight, training=self.training)
            y = nn.functional.conv2d(x, new_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

            # y = nn.functional.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

            # self.weight.data = self.W.data
            # print('self.weight2: ', self.weight)


        # new_weight = self.weight_ib(self.weight, training=self.training)

        # y = nn.functional.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        # y = nn.functional.conv2d(x, new_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return y

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, wib=0, kl_mult=1):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        # self.wib=wib
        self.conv1 = conv3x3(inplanes, planes, wib, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.ib1 = InformationBottleneck(planes, kl_mult=kl_mult)
        self.conv2 = conv3x3(planes, planes, wib)
        self.bn2 = norm_layer(planes)
        self.ib2 = InformationBottleneck(planes, kl_mult=kl_mult)
        self.downsample = downsample
        self.stride = stride


    def compute_Wib_upbound(self, ):
        wib_upbound =0
        wib_upbound += self.conv1.weight_ib.compute_Wib_upbound(self.conv1.weight_ib.logalpha)
        # 之前版本错了
        # wib_upbound += self.conv2.weight_ib.compute_Wib_upbound(self.conv1.weight_ib.logalpha)
        # 正确版本
        wib_upbound += self.conv2.weight_ib.compute_Wib_upbound(self.conv2.weight_ib.logalpha)
        return wib_upbound


    def compute_compression_ratio(self, threshold, pre_mask, n=0):
        # applicable for structures with global pooling before fc
        total_params, pruned_params, remain_params = 0, 0, 0
        fmap_size=32
        out_channels1 = self.conv1.out_channels
        out_channels2 = self.conv2.out_channels
        in_channels1=self.conv1.in_channels
        in_channels2 = self.conv2.in_channels

        total_params = in_channels1 * out_channels1 * 9
        total_params += in_channels2 * out_channels2 * 9
        hard_mask1 = self.conv1.get_mask_hard(threshold)
        hard_mask2 = self.conv2.get_mask_hard(threshold)
        remain_params = pre_mask * hard_mask1 * 9
        remain_params += hard_mask1 *hard_mask2 * 9
        pruned_params = total_params - remain_params
        flops = (fmap_size ** 2) * remain_params

        # print('in_channels1: {}, in_channels2: {}, out_channels1:{},  out_channels2: {},'
        #       .format(in_channels1, in_channels2, out_channels1, out_channels2))
        # print('pre_mask: {}, hard_mask1: {}, hard_mask2:{},'
        #       .format(pre_mask, hard_mask1, hard_mask2))
        # print('total parameters: {}, pruned parameters: {}, remaining params:{},  remaining flops: {},'
        #       .format(total_params, pruned_params, remain_params, flops))

        return total_params, pruned_params, remain_params, flops


    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.ib1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ib2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, wib=0, kl_mult=1):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width, wib)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, wib, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.ib2 = InformationBottleneck(width, kl_mult=kl_mult)
        self.conv3 = conv1x1(width, planes * self.expansion, wib)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def compute_Wib_upbound(self, ):
        wib_upbound =0
        wib_upbound += self.conv1.weight_ib.compute_Wib_upbound(self.conv1.weight_ib.logalpha)
        wib_upbound += self.conv2.weight_ib.compute_Wib_upbound(self.conv2.weight_ib.logalpha)
        wib_upbound += self.conv3.weight_ib.compute_Wib_upbound(self.conv3.weight_ib.logalpha)
        return wib_upbound

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.ib2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



class RESNET_IB(nn.Module):
    def __init__(self, block, config=None, mag=9, batch_norm=False, threshold=0,
                init_var=0.01, sample_in_training=True, sample_in_testing=False, n_cls=10, no_ib=False, a=0.5, b=0.5,

                 ###resnet 初始参数
                 zero_init_residual=False, wib=1,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None
                 ):
        super(RESNET_IB, self).__init__()

        self.expansion = block.expansion

        ### resnet 初始化
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        # self.layers = layers

        self.wib = wib

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group



        self.init_mag = mag
        self.threshold = threshold
        self.config = config
        self.init_var = init_var
        self.sample_in_training = sample_in_training
        self.sample_in_testing = sample_in_testing
        self.no_ib = no_ib
        self.a = a
        self.b = b

        print('Using structure1 {}'.format(cfg[config]))
        self.conv_layers, conv_kl_list = self.make_conv_layers(cfg[config], batch_norm, block)
        print('Using structure {}'.format(cfg[config]))

        # print('conv_layers {}'.format(self.conv_layers))

        print('conv_layers {}'.format(self.conv_layers))

        print('conv_kl_list {}'.format(conv_kl_list))

        # self.compute_Wib_upbound()

        fc_ib1 = InformationBottleneck(512*block.expansion, mask_thresh=threshold, init_mag=self.init_mag, init_var=self.init_var,
                    sample_in_training=sample_in_training, sample_in_testing=sample_in_testing,a=self.a,b=self.b)
        fc_ib2 = InformationBottleneck(512*block.expansion, mask_thresh=threshold, init_mag=self.init_mag, init_var=self.init_var,
                    sample_in_training=sample_in_training, sample_in_testing=sample_in_testing,a=self.a,b=self.b)
        self.n_cls = n_cls
        # self.n = 2048
        # self.n = 4096
        self.n = 1024
        if self.config in ['G1', 'D6']:
            # t3p3 t4p2
            self.fc_layers = nn.Sequential(nn.Linear(512*block.expansion, self.n_cls))
            self.kl_list = conv_kl_list

            #resnet32
            init_kl_list = [64, 64, 64, 64, 64, 64, 64,
                            128, 128, 128, 128, 128, 128, 128, 128,
                            256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256,
                            512, 512, 512, 512, 512, 512]

            self.init_kl_list = [x / self.n for x in init_kl_list]

            # resnet32
            kl_mult_temp = [64, 64, 64, 64, 64, 64, 64,
                            128, 128, 128, 128, 128, 128, 128, 128,
                            256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256,
                            512, 512, 512, 512, 512, 512]


            self.kl_mult_temp = [x / self.n for x in kl_mult_temp]
            self.ratio = [1, 1, 1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1, 1]
            _,self.last_prune_stat = self.get_masks(hard_mask=True, threshold=threshold)
            _,self.pruned_structure = self.get_masks(hard_mask=True, threshold=threshold)
        elif self.config == 'G2':
            # t3p3 t4p2
            self.fc_layers = nn.Sequential(nn.Linear(512 * block.expansion, self.n_cls))
            self.kl_list = conv_kl_list

            # resnet50
            init_kl_list = [64, 64, 64, 64,
                            128, 128, 128, 128,
                            256, 256, 256, 256, 256, 256,
                            512, 512, 512]

            # init_kl_list = [256, 256, 256, 256,
            #                 256, 256, 256, 256,
            #                 256, 256, 256, 256, 256, 256,
            #                 256, 256, 256]

            self.init_kl_list = [x / self.n for x in init_kl_list]

            # resnet50
            kl_mult_temp =  [64, 64, 64, 64,
                            128, 128, 128, 128,
                            256, 256, 256, 256, 256, 256,
                            512, 512, 512]

            # kl_mult_temp = [256, 256, 256, 256,
            #                 256, 256, 256, 256,
            #                 256, 256, 256, 256, 256, 256,
            #                 256, 256, 256]

            self.kl_mult_temp = [x / self.n for x in kl_mult_temp]
            self.ratio = [1, 1, 1, 1,
                          1, 1, 1, 1,
                          1, 1, 1, 1, 1, 1,
                          1, 1, 1]
            _, self.last_prune_stat = self.get_masks(hard_mask=True, threshold=threshold)
            _, self.pruned_structure = self.get_masks(hard_mask=True, threshold=threshold)
        elif self.config == 'G5':
            # t3p3 t4p2
            self.fc_layers = nn.Sequential(nn.Linear(512*block.expansion, self.n_cls))
            self.kl_list = conv_kl_list
            init_kl_list = [64,
                            64, 64, 64, 64,
                            128, 128, 128, 128,
                            256, 256, 256, 256,
                            512, 512, 512, 512]
            self.init_kl_list = [x / self.n for x in init_kl_list]
            kl_mult_temp = [64,
                            64, 64, 64, 64,
                            128, 128, 128, 128,
                            256, 256, 256, 256,
                            512, 512, 512, 512]
            self.kl_mult_temp = [x / self.n for x in kl_mult_temp]
            self.ratio = [1,
                          1, 1, 1, 1,
                          1, 1, 1, 1,
                          1, 1, 1, 1,
                          1, 1, 1, 1]
            _,self.last_prune_stat = self.get_masks(hard_mask=True, threshold=threshold)
            _,self.pruned_structure = self.get_masks(hard_mask=True, threshold=threshold)
        else:
            # D4 t3p1
            fc_layer_list = [nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, self.n_cls)] if no_ib else \
                    [nn.Linear(512, 512), nn.ReLU(), fc_ib1, nn.Linear(512, 512), nn.ReLU(), fc_ib2, nn.Linear(512, self.n_cls)]
            self.fc_layers = nn.Sequential(*fc_layer_list)
            self.kl_list = conv_kl_list + [fc_ib1, fc_ib2]
            self.init_kl_list = [1/32, 1/32, 1/16, 1/16, 1/8, 1/8, 1/8, 1/4, 1/4, 1/4, 1/2, 1/2, 1/2, 1, 1]
            self.kl_mult_temp = [1/32, 1/32, 1/16, 1/16, 1/8, 1/8, 1/8, 1/4, 1/4, 1/4, 1/2, 1/2, 1/2, 1, 1]
            _,self.last_prune_stat = self.get_masks(hard_mask=True, threshold=threshold)
            _,self.pruned_structure = self.get_masks(hard_mask=True, threshold=threshold)

        print(self.kl_mult_temp)
        print(self.init_kl_list)


        ### resnet 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                print('ok1')
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            # elif isinstance(m, WibConv2d):
            #     print('ok2')
            #     nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    ### resnet的make_layer函数
    def _make_layer(self, block, planes, blocks=1, stride=1, dilate=False, kl_mult=1 ):
        norm_layer = self._norm_layer
        wib = self.wib
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, wib, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, wib, kl_mult))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, wib = wib, kl_mult = kl_mult))

        return nn.Sequential(*layers)



    def make_conv_layers(self, config, batch_norm, block, blocks=1, dilate=False):
        layers, kl_list = [], []
        in_channels = 3
        for v in config:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
            elif v == 'A':
                # layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
                layers += [nn.AdaptiveAvgPool2d((1, 1))]
            else:

                ##判断是第一个卷积层，还是block模块
                if v[1]==1:#第一个卷积层，按照vgg类似操作构建信息瓶颈层
                    # conv2d = nn.Conv2d(in_channels, v[0], kernel_size=7, stride=2, padding=3, bias=False)
                    # conv2d = nn.Conv2d(in_channels, v[0], kernel_size=3, stride=1, padding=1, bias=False)
                    conv2d  = conv3x3(3, v[0], stride=1, wib=self.wib)
                    ib = InformationBottleneck(v[0], mask_thresh=self.threshold, init_mag=self.init_mag, init_var=self.init_var,
                        kl_mult=v[3], sample_in_training=self.sample_in_training, sample_in_testing=self.sample_in_testing,a=self.a,b=self.b)
                    if batch_norm:
                        layers += [conv2d, nn.BatchNorm2d(v[0]), nn.ReLU(inplace=True)]
                    else:
                        layers += [conv2d, nn.ReLU(inplace=True)]

                    if not self.no_ib:
                        layers.append(ib)
                        kl_list.append(ib)

                if v[1]==2:#属于resnet的BasicBlock模块，调用resnet的make_layer函数
                    resblock = self._make_layer(block, v[0], stride=v[2], kl_mult=v[3])
                    layers += [resblock]
                    kl_list.append(resblock[0].ib1)
                    kl_list.append(resblock[0].ib2)
                    ib = InformationBottleneck(v[0]*block.expansion, mask_thresh=self.threshold, init_mag=self.init_mag, init_var=self.init_var,
                        kl_mult=v[3], sample_in_training=self.sample_in_training, sample_in_testing=self.sample_in_testing,a=self.a,b=self.b)

                    # if not self.no_ib:
                    #     layers.append(ib)
                    #     kl_list.append(ib)

                if v[1]==3:#属于resnet的Bottleneck模块，调用resnet的make_layer函数
                    resblock = self._make_layer(block, v[0], stride=v[2], kl_mult=v[3])
                    layers += [resblock]
                    kl_list.append(resblock[0].ib2)
                    # ib = InformationBottleneck(v[0]*block.expansion, mask_thresh=self.threshold, init_mag=self.init_mag, init_var=self.init_var,
                    #     kl_mult=v[3], sample_in_training=self.sample_in_training, sample_in_testing=self.sample_in_testing,a=self.a,b=self.b)

                in_channels = v[0]

                # if not self.no_ib:
                #     layers.append(ib)
                #     kl_list.append(ib)

        return nn.Sequential(*layers), kl_list

    def auto_kl_mult(self):
        # _, prune_stat = self.get_masks(hard_mask=True, threshold=threshold)
        # conv_shapes = [v[0] for v in cfg[self.config] if type(v) is not str]

        if self.config in ['G', 'D6']:
            conv_shapes = [v[0] for v in cfg[self.config] if type(v) is not str]
            fc_shapes = [512]
        elif self.config in ['G5', 'G1']:
            conv_shapes = [v[0] for v in cfg[self.config] if type(v) is not str]
            conv_shapes_temp=[]
            conv_shapes_temp += [conv_shapes[0]]
            for i in range(len(conv_shapes)-1):
                conv_shapes_temp += [conv_shapes[i + 1]]
                conv_shapes_temp += [conv_shapes[i + 1]]
            conv_shapes = conv_shapes_temp
            fc_shapes = []
        elif self.config in ['G2']:
            conv_shapes = [v[0] for v in cfg[self.config] if type(v) is not str]
            # conv_shapes[0]=conv_shapes[0]/self.expansion
            fc_shapes = []
        else:
            fc_shapes = [512, 512]

        # print('prune_stat: {}, last_prune_stat:{}'.format(prune_stat, self.last_prune_stat))

        remain_stat = [out_channels - self.last_prune_stat[idx] for idx, out_channels in enumerate(conv_shapes + fc_shapes)]
        init_stat = [out_channels for idx, out_channels in enumerate(conv_shapes + fc_shapes)]

        sum = 0
        # a=32
        for i in range(len(init_stat)):
            a = init_stat[i]/2
            self.ratio[i] = remain_stat[i] / init_stat[i]
            # sum = sum + self.ratio[i]
            sum = sum + math.tan(math.pi*(a-1)/a/2*self.ratio[i])

        # offset = 1 / len(self.init_kl_list)
        b = 1.2
        c= 0.01

        # conv_kl_mult = 4

        for i in range(len(self.init_kl_list)):

            a=init_stat[i]/2
            temp1 = len(self.init_kl_list)/2 - abs(i-len(self.init_kl_list)/2)
            max1 =  len(self.init_kl_list)/2
            temp2 = remain_stat[i]
            max2 = max(remain_stat)
            # print('i:')
            # print('(a-1)/a/2:',(a-1)/a/2)
            # print('self.ratio[i]:', self.ratio[i])
            # print('math.pi*(a-1)/a/2*self.ratio[i]:', math.pi*(a-1)/a/2*self.ratio[i])

            # self.kl_list[i].kl_mult = self.init_kl_list[i] * (
            #             1 + b* math.log(temp2,2)/math.log(max2,2)*
            #             (math.log(1 + temp1, 2) / math.log(1 + max1, 2)) *
            #             (math.tan(math.pi*(a-1)/a/2*self.ratio[i]) / sum) * len(self.init_kl_list))

            if temp2==0:
                self.kl_list[i].kl_mult=0
                self.kl_mult_temp[i]=0
            else:
                self.kl_list[i].kl_mult = self.init_kl_list[i] * (
                            2* b* math.log(2+temp2,2)/math.log(max2,2)*
                            (math.log(1 + temp1, 2) / math.log(2 + max1, 2)) *
                            (math.tan(math.pi*(a-1)/a/2*self.ratio[i]) / sum) * len(self.init_kl_list))

                self.kl_mult_temp[i] = self.kl_list[i].kl_mult / self.init_kl_list[i]

        # print('conv_kl_mult:',conv_kl_mult)
        print(b)
        print(self.ratio)
        print(self.init_kl_list)
        print(self.kl_mult_temp)

    def adapt_dropout(self, p):
        # conv_shapes = [v[0] for v in cfg[self.config] if type(v) is not str]

        if self.config in ['G', 'D6']:
            conv_shapes = [v[0] for v in cfg[self.config] if type(v) is not str]
            fc_shapes = [512]
        elif self.config in ['G5', 'G1']:
            conv_shapes = [v[0] for v in cfg[self.config] if type(v) is not str]
            conv_shapes_temp = []
            conv_shapes_temp += [conv_shapes[0]]
            for i in range(len(conv_shapes) - 1):
                conv_shapes_temp += [conv_shapes[i + 1]]
                conv_shapes_temp += [conv_shapes[i + 1]]
            conv_shapes = conv_shapes_temp
            fc_shapes = []
        elif self.config in ['G2']:
            conv_shapes = [v[0] for v in cfg[self.config] if type(v) is not str]
            # conv_shapes[0] = conv_shapes[0] / self.expansion
            fc_shapes = []
        else:
            fc_shapes = [512, 512]

        remain_stat = [out_channels - self.last_prune_stat[idx] for idx, out_channels in enumerate(conv_shapes + fc_shapes)]

        for i in range(len(self.init_kl_list)):
            if remain_stat[i] < 150:
            # if remain_stat[i] < 200:
            # if remain_stat[i] < 120:
                # self.kl_list[i].p=1
                self.kl_list[i].p = 1
            else:
                # self.kl_list[i].p = 1.0-1.0/remain_stat[i]

                # 原设置
                # self.kl_list[i].p = 0.99
                self.kl_list[i].p = 1

            print(i,self.kl_list[i].p)



    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv_layers(x).view(batch_size, -1)
        x = self.fc_layers(x)

        if self.training:
            if self.no_ib:
                # return x

                if not self.wib:
                    return x
                else:
                    Wib_upbound = self.compute_Wib_upbound()
                    return x, Wib_upbound

            else:
                if not self.wib:
                    ib_kld = self.kl_list[0].kld
                    for ib in self.kl_list[1:]:
                        ib_kld += ib.kld
                    return x, ib_kld.float()
                else:
                    ib_kld = self.kl_list[0].kld
                    for ib in self.kl_list[1:]:
                        ib_kld += ib.kld
                    Wib_upbound = self.compute_Wib_upbound()
                    return x, ib_kld.float(), Wib_upbound
        else:
            return x

    def get_masks(self, hard_mask=True, threshold=0):
        masks = []
        if hard_mask:
            masks = [ib_layer.get_mask_hard(threshold) for ib_layer in self.kl_list]
            return masks, [np.sum(mask.cpu().numpy()==0) for mask in masks]
        else:
            masks = [ib_layer.get_mask_weighted(threshold) for ib_layer in self.kl_list]
            return masks

    def compute_Wib_upbound(self,):
        Wib_upbound = 0
        offset=0
        interval=0
        Wib_upbound += self.conv_layers[0].weight_ib.compute_Wib_upbound(self.conv_layers[0].weight_ib.logalpha)
        # print('conv_layers: {}'.format(self.conv_layers))
        if not self.no_ib:
            offset=5
            interval=0
        else:
            offset=4
            interval=1
        for i in range(8):
            # print('self.conv_layers[5+i*2]: {}'.format(self.conv_layers[5+i*2]))
            block=self.conv_layers[offset+i*(2-interval)]
            # print('block: {}'.format(block[0]))
            Wib_upbound += block[0].compute_Wib_upbound()

        return Wib_upbound

    def print_params(self,):
        mu = []
        logalpha = []
        weight = []
        weight += [self.conv_layers[0].weight]

        offset = 0
        interval = 0
        if not self.no_ib:
            offset=5
            interval=0
        else:
            offset=4
            interval=1

        # print('weight: {}'.format(weight))
        if self.wib:
            mu += [self.conv_layers[0].weight_ib.mu]
            logalpha += [self.conv_layers[0].weight_ib.logalpha]
            mask_w,_= self.conv_layers[0].weight_ib.get_mask_hard(self.conv_layers[0].weight_ib.epsilon)
            if not self.no_ib:
                mask_a = self.kl_list[0].get_mask_hard()
                mask_dert = mask_w - mask_a
            print('mask_w: {}'.format(mask_w))
            if not self.no_ib:
                print('mask_a: {}'.format(mask_a))
                print('mask_dert: {}'.format(mask_dert))
            print('mu: {}, logalpha: {}'.format(mu, logalpha))
            for i in range(8):
                # print('self.conv_layers[5+i*2]: {}'.format(self.conv_layers[5+i*2]))

                block = self.conv_layers[offset + i * (2 - interval)]
                # block=self.conv_layers[5+i*2]

                # print('block: {}'.format(block[0]))
                mu += [block[0].conv1.weight_ib.mu]
                mu += [block[0].conv2.weight_ib.mu]
                logalpha += [block[0].conv1.weight_ib.logalpha]
                logalpha += [block[0].conv2.weight_ib.logalpha]

        # mu = [ib_layer.post_z_mu for ib_layer in self.kl_list]
        # logalpha = [ib_layer.post_z_logD for ib_layer in self.kl_list]
        # print('mu: {}, logalpha: {}'.format(mu, logalpha))


    def print_compression_ratio(self, threshold, writer=None, epoch=-1):
        # applicable for structures with global pooling before fc
        _, prune_stat = self.get_masks(hard_mask=True, threshold=threshold)


        # conv_shapes = [v[0] for v in cfg[self.config] if type(v) is not str]

        if self.config in ['G', 'D6']:
            conv_shapes = [v[0] for v in cfg[self.config] if type(v) is not str]
            fc_shapes = [512]
        elif self.config in ['G5', 'G1']:
            conv_shapes = [v[0] for v in cfg[self.config] if type(v) is not str]
            conv_shapes_temp = []
            conv_shapes_temp += [conv_shapes[0]]
            for i in range(len(conv_shapes) - 1):
                conv_shapes_temp += [conv_shapes[i + 1]]
                conv_shapes_temp += [conv_shapes[i + 1]]
            conv_shapes = conv_shapes_temp
            fc_shapes = []
        elif self.config in ['G2']:
            conv_shapes = [v[0]  for v in cfg[self.config] if type(v) is not str]
            # conv_shapes[0] = conv_shapes[0] / self.expansion
            fc_shapes = []
        else:
            fc_shapes = [512, 512]

        # print('prune_stat: {}, last_prune_stat:{}'.format(prune_stat, self.last_prune_stat))

        self.pruned_structure = [prune_stat[idx] - self.last_prune_stat[idx] for idx, out_channels in enumerate(conv_shapes + fc_shapes)]
        self.last_prune_stat = [prune_stat[idx] for idx, out_channels in enumerate(conv_shapes + fc_shapes)]
        net_shape = [ out_channels-prune_stat[idx] for idx, out_channels in enumerate(conv_shapes+fc_shapes)]
        #conv_shape_with_pool = [v[0] if v != 'M' else 'M' for v in cfg[self.config]]
        current_n, hdim, last_channels, flops, fmap_size = 0, 64, 3, 0, 32
        for n, pruned_channels in enumerate(prune_stat):
            if n < len(conv_shapes):
                # current_channels = cfg[self.config][current_n][0] - pruned_channels
                current_channels = conv_shapes[current_n] - pruned_channels
                flops += (fmap_size**2) * 9 * last_channels * current_channels
                last_channels = current_channels
                current_n += 1

                if self.config in ['G1']:
                    if current_n==1 or current_n==8 or current_n==16 or current_n==28 or current_n==33:
                        fmap_size /= 2
                        hdim *= 2
                if self.config in ['G5']:
                    if current_n==1 or current_n==6 or current_n==10 or current_n==14 or current_n==17:
                        fmap_size /= 2
                        hdim *= 2
                # if type(cfg[self.config][current_n]) is str:
                #     current_n += 1
                #     fmap_size /= 2
                #     hdim *= 2

            else:
                current_channels = 512 - pruned_channels
                flops += last_channels * current_channels
                last_channels = current_channels
        flops += last_channels * self.n_cls

        total_params, pruned_params, remain_params = 0, 0, 0
        # total number of conv params
        in_channels, in_pruned = 3, 0
        for n, n_out in enumerate(conv_shapes):
            n_params = in_channels * n_out * 9
            total_params += n_params
            n_remain = (in_channels - in_pruned) * (n_out - prune_stat[n]) * 9
            remain_params += n_remain
            pruned_params += n_params - n_remain
            in_channels = n_out
            in_pruned = prune_stat[n]
            # print('n_params: {}, n_remain: {}, in_channels:{}, in_pruned:{}, n_out: {}, prune_stat: {},'.format(n_params, n_remain, in_channels, in_pruned, n_out, prune_stat))
        # fc layers
        offset = len(prune_stat) - len(fc_shapes)
        for n, n_out in enumerate(fc_shapes):
            n_params = in_channels * n_out
            total_params += n_params
            n_remain = (in_channels - in_pruned) * (n_out - prune_stat[n+offset])
            remain_params += n_remain
            pruned_params += n_params - n_remain
            in_channels = n_out
            in_pruned = prune_stat[n+offset]
            # print('n_params: {}, n_remain: {}, in_channels:{}, in_pruned:{}, n_out: {}, prune_stat: {},'.format(n_params, n_remain, in_channels, in_pruned, n_out, prune_stat))
        total_params += in_channels * self.n_cls
        remain_params += (in_channels - in_pruned) * self.n_cls
        pruned_params += in_pruned * self.n_cls

        self.print_params()
        print('total parameters: {}, pruned parameters: {}, remaining params:{}, remain/total params:{}, remaining flops: {}, remaining flops/params: {},'
              'each layer pruned: {}, this epoch each layer pruned: {}, remaining structure:{}'.format(total_params, pruned_params, remain_params,
                    float(total_params-pruned_params)/total_params, flops, 0.0000000001 * flops/(float(total_params-pruned_params)/total_params), prune_stat, self.pruned_structure, net_shape))

        if writer is not None:
            writer.add_scalar('flops', flops, epoch)
            writer.add_scalar('remain/total params', float(total_params-pruned_params)/total_params, epoch)
            writer.add_scalar('flops/remaining params', 0.0000000001 * flops/(float(total_params-pruned_params)/total_params), epoch)





