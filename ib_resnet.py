import torch
from torch import nn
import numpy as np

from ib_layers import *

# model configuration, (out_channels, kl_multiplier), 'M': Mean pooling, 'A': Average pooling
cfg = {
    'D6': [(64, 1.0/32), (64, 1.0/32), 'M', (128, 1.0/16), (128, 1.0/16), 'M', (256, 1.0/8), (256, 1.0/8), (256, 1.0/8),
        'M', (512, 1.0/4), (512, 1.0/4), (512, 1.0/4), 'M', (512, 1.0/2), (512, 1.0/2), (512, 1.0/2), 'M'],
    'D5': [(64, 1.0/32**2), (64, 1.0/32**2), 'M', (128, 1.0/16**2), (128, 1.0/16**2), 'M', (256, 1.0/8**2), (256, 1.0/8**2), (256, 1.0/8**2), 
        'M', (512, 1.0/4**2), (512, 1.0/4**2), (512, 1.0/4**2), 'M', (512, 1.0/2**2), (512, 1.0/2**2), (512, 1.0/2**2), 'M'],
    'D4': [(64, 1.0/32), (64, 1.0/32), 'M', (128, 1.0/16), (128, 1.0/16), 'M', (256, 1.0/8), (256, 1.0/8), (256, 1.0/8),
        'M', (512, 1.0/4), (512, 1.0/4), (512, 1.0/4), 'M', (512, 1.0/2), (512, 1.0/2), (512, 1.0/2), 'M'],
    'D3': [(64, 0.1), (64, 0.1), 'M', (128, 0.5), (128, 0.5), 'M', (256, 1), (256, 1), (256, 1), 
        'M', (512, 1), (512, 1), (512, 1), 'M', (512, 1), (512, 1), (512, 1), 'M'],
    'D2': [(64, 0.01), (64, 0.01), 'M', (128, 1), (128, 1), 'M', (256, 1), (256, 1), (256, 1), 
        'M', (512, 1), (512, 1), (512, 1), 'M', (512, 1), (512, 1), (512, 1), 'M'],
    'D1': [(64, 0.1), (64, 0.1), 'M', (128, 1), (128, 1), 'M', (256, 1), (256, 1), (256, 1), 
        'M', (512, 1), (512, 1), (512, 1), 'M', (512, 1), (512, 1), (512, 1), 'M'],
    'D0': [(64, 1), (64, 1), 'M', (128, 1), (128, 1), 'M', (256, 1), (256, 1), (256, 1), 
        'M', (512, 1), (512, 1), (512, 1), 'M', (512, 1), (512, 1), (512, 1), 'M'],
    'G':[(64, 1.0/32), (64, 1.0/32), 'M', (128, 1.0/16), (128, 1.0/16), 'M', (256, 1.0/8), (256, 1.0/8), (256, 1.0/8), 
        'M', (512, 1.0/4), (512, 1.0/4), (512, 1.0/4), 'M', (512, 1.0/2), (512, 1.0/2), (512, 1.0/2), 'M'], # VGG 16 with one fewer FC
    'G5': [(64, 1.0/32), (64, 1.0/32), 'M', (128, 1.0/16), (128, 1.0/16), 'M', (256, 1.0/8), (256, 1.0/8), (256, 1.0/8), (256, 1.0/8),
        'M', (512, 1.0/4), (512, 1.0/4), (512, 1.0/4), (512, 1.0/4), 'M', (512, 1.0/2), (512, 1.0/2), (512, 1.0/2), (512, 1.0/2), 'A']
    # kl_mult2
    # 'D6': [(64, 1.0/32), (64, 1.0/32), 'M', (128, 1.0/8), (128, 1.0/8), 'M', (256, 1.0/4), (256, 1.0/4), (256, 1.0/4),
    #     'M', (512, 1.0/4), (512, 1.0/4), (512, 1.0/4), 'M', (512, 1.0/2), (512, 1.0/2), (512, 1.0/2), 'M'],
    # 'D4': [(64, 1.0/32), (64, 1.0/32), 'M', (128, 1.0/8), (128, 1.0/8), 'M', (256, 1.0/4), (256, 1.0/4), (256, 1.0/4),
    #     'M', (512, 1.0/4), (512, 1.0/4), (512, 1.0/4), 'M', (512, 1.0/2), (512, 1.0/2), (512, 1.0/2), 'M'],
    # 'G5': [(64, 1.0/32), (64, 1.0/32), 'M', (128, 1.0/8), (128, 1.0/8), 'M', (256, 1.0/4), (256, 1.0/4), (256, 1.0/4), (256, 1.0/4),
    #     'M', (512, 1.0/4), (512, 1.0/4), (512, 1.0/4), (512, 1.0/4), 'M', (512, 1.0/2), (512, 1.0/2), (512, 1.0/2), (512, 1.0/2), 'A']
}

class VGG_IB(nn.Module):
    def __init__(self, config=None, mag=9, batch_norm=False, threshold=0, 
                init_var=0.01, sample_in_training=True, sample_in_testing=False, n_cls=10, no_ib=False, a=0.5, b=0.5):
        super(VGG_IB, self).__init__()

        self.init_mag = mag
        self.threshold = threshold
        self.config = config
        self.init_var = init_var
        self.sample_in_training = sample_in_training
        self.sample_in_testing = sample_in_testing
        self.no_ib = no_ib
        self.a = a
        self.b = b

        self.conv_layers, conv_kl_list = self.make_conv_layers(cfg[config], batch_norm)
        print('Using structure {}'.format(cfg[config]))

        print('conv_layers {}'.format(self.conv_layers))

        print('conv_kl_list {}'.format(conv_kl_list))

        fc_ib1 = InformationBottleneck(512, mask_thresh=threshold, init_mag=self.init_mag, init_var=self.init_var, 
                    sample_in_training=sample_in_training, sample_in_testing=sample_in_testing,a=self.a,b=self.b)
        fc_ib2 = InformationBottleneck(512, mask_thresh=threshold, init_mag=self.init_mag, init_var=self.init_var, 
                    sample_in_training=sample_in_training, sample_in_testing=sample_in_testing,a=self.a,b=self.b)
        self.n_cls = n_cls
        self.n = 2048
        if self.config in ['G', 'D6']:
            # t3p2 t4p1
            fc_layer_list = [nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, self.n_cls)] if no_ib else \
                            [nn.Linear(512, 512), nn.ReLU(), fc_ib1, nn.Linear(512, self.n_cls)] 
            self.fc_layers = nn.Sequential(*fc_layer_list)
            self.kl_list = conv_kl_list + [fc_ib1]
            self.init_kl_list = [1/32, 1/32, 1/16, 1/16, 1/8, 1/8, 1/8, 1/4, 1/4, 1/4, 1/2, 1/2, 1/2, 1]
            self.kl_mult_temp = [1/32, 1/32, 1/16, 1/16, 1/8, 1/8, 1/8, 1/4, 1/4, 1/4, 1/2, 1/2, 1/2, 1]
            # self.init_kl_list = [1/16, 1/16, 1/8, 1/8, 1/4, 1/4, 1/4, 1/2, 1/2, 1/2, 1/2, 1/2, 1/2, 1, 1]
            # self.kl_mult_temp = [1/16, 1/16, 1/8, 1/8, 1/4, 1/4, 1/4, 1/2, 1/2, 1/2, 1/2, 1/2, 1/2, 1, 1]
            _,self.last_prune_stat = self.get_masks(hard_mask=True, threshold=threshold)
            _,self.pruned_structure = self.get_masks(hard_mask=True, threshold=threshold)
        elif self.config == 'G5':
            # t3p3 t4p2
            self.fc_layers = nn.Sequential(nn.Linear(512, self.n_cls))
            self.kl_list = conv_kl_list
            # self.init_kl_list = [1/32, 1/32, 1/16, 1/16, 1/8, 1/8, 1/8, 1/8, 1/4, 1/4, 1/4, 1/4, 1/2, 1/2, 1/2, 1/2]
            # self.kl_mult_temp = [1/32, 1/32, 1/16, 1/16, 1/8, 1/8, 1/8, 1/8, 1/4, 1/4, 1/4, 1/4, 1/2, 1/2, 1/2, 1/2]

            init_kl_list = [64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512]
            self.init_kl_list = [x/self.n for x in init_kl_list]
            kl_mult_temp = [64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512]
            self.kl_mult_temp = [x/self.n for x in kl_mult_temp]

            # for i in range(len(self.init_kl_list)):
            #     self.init_kl_list[i] = self.init_kl_list[i] * math.sqrt(math.log(i+2,2))

            # for i in range(len(self.init_kl_list)):
            #     temp = len(self.init_kl_list)/2 - abs(i-len(self.init_kl_list)/2)
            #     max =  len(self.init_kl_list)/2
            #     self.init_kl_list[i] = self.init_kl_list[i] * (1 +  math.log(1+temp, 2)/math.log(1+max, 2))

            self.ratio = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]


            # self.init_kl_list = [1/16, 1/16, 1/8, 1/8, 1/4, 1/4, 1/4, 1/2, 1/2, 1/2, 1/2, 1/2, 1/2, 1, 1]
            # self.kl_mult_temp = [1/16, 1/16, 1/8, 1/8, 1/4, 1/4, 1/4, 1/2, 1/2, 1/2, 1/2, 1/2, 1/2, 1, 1]
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
            # self.init_kl_list = [1/16, 1/16, 1/8, 1/8, 1/4, 1/4, 1/4, 1/2, 1/2, 1/2, 1/2, 1/2, 1/2, 1, 1]
            # self.kl_mult_temp = [1/16, 1/16, 1/8, 1/8, 1/4, 1/4, 1/4, 1/2, 1/2, 1/2, 1/2, 1/2, 1/2, 1, 1]
            _,self.last_prune_stat = self.get_masks(hard_mask=True, threshold=threshold)
            _,self.pruned_structure = self.get_masks(hard_mask=True, threshold=threshold)

        print(self.kl_mult_temp)
        print(self.init_kl_list)



    def make_conv_layers(self, config, batch_norm):
        layers, kl_list = [], []
        in_channels = 3
        for v in config:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif v == 'A':
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v[0], kernel_size=3, padding=1)
                in_channels = v[0]
                ib = InformationBottleneck(v[0], mask_thresh=self.threshold, init_mag=self.init_mag, init_var=self.init_var, 
                    kl_mult=v[1], sample_in_training=self.sample_in_training, sample_in_testing=self.sample_in_testing,a=self.a,b=self.b)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v[0]), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                if not self.no_ib:
                    layers.append(ib)
                    kl_list.append(ib)
        return nn.Sequential(*layers), kl_list

    def auto_kl_mult(self):
        # _, prune_stat = self.get_masks(hard_mask=True, threshold=threshold)
        conv_shapes = [v[0] for v in cfg[self.config] if type(v) is not str]

        if self.config in ['G', 'D6']:
            fc_shapes = [512]
        elif self.config == 'G5':
            fc_shapes = []
        else:
            fc_shapes = [512, 512]

        remain_stat = [out_channels - self.last_prune_stat[idx] for idx, out_channels in enumerate(conv_shapes + fc_shapes)]
        init_stat = [out_channels for idx, out_channels in enumerate(conv_shapes + fc_shapes)]

        sum = 0
        # a=32
        for i in range(len(init_stat)):
            a = init_stat[i]/2
            self.ratio[i] = remain_stat[i] / init_stat[i]
            # sum = sum + self.ratio[i]
            sum = sum + math.tan(math.pi*(a-1)/a/2*self.ratio[i])
            # sum = sum + math.log(1+self.ratio[i], 2)
            # sum = sum + self.ratio[i] * self.ratio[i]
            # sum = sum + self.ratio[i] * self.ratio[i] * self.ratio[i]

        offset = 1 / len(self.init_kl_list)
        # conv_kl_mult = 1

        # conv_kl_mult = 2
        # conv_kl_mult = 3
        # conv_kl_mult = 5

        conv_kl_mult = 4

        # conv_kl_mult = 8
        # conv_kl_mult = len(self.init_kl_list)

        # for i in range(len(self.init_kl_list)):
        #     self.kl_list[i].kl_mult = self.init_kl_list[i] * (
        #                 1 + conv_kl_mult * (self.ratio[i] * self.ratio[i] * self.ratio[i] / sum - offset ))
        #     self.kl_mult_temp[i] = self.kl_list[i].kl_mult / self.init_kl_list[i]

        # max = math.ceil(len(self.init_kl_list) / 2)-1
        # max = int(len(self.init_kl_list) / 2)



        # b=0.9
        # b = 0.9
        b=1.2
        # b=1.5
        # b=1.8


        for i in range(len(self.init_kl_list)):
            # temp = len(self.init_kl_list)/2 - abs(i-len(self.init_kl_list)/2)+2
            # max =  len(self.init_kl_list)/2 + 2
            # self.kl_list[i].kl_mult = self.init_kl_list[i] * (1 +  math.log(temp, 2)/math.log(max, 2) * (self.ratio[i]/ sum )* len(self.init_kl_list) )
            # self.kl_mult_temp[i] = self.kl_list[i].kl_mult / self.init_kl_list[i]

            # if i< len(self.init_kl_list)/2:
            #     temp = i
            # else: temp = len(self.init_kl_list) - i

            a=init_stat[i]/2

            # (math.log(remain_stat[i], 2) / math.log(max(remain_stat), 2)) *

            temp1 = len(self.init_kl_list)/2 - abs(i-len(self.init_kl_list)/2)
            max1 =  len(self.init_kl_list)/2
            temp2 = remain_stat[i]
            max2 = max(remain_stat)
            self.kl_list[i].kl_mult = self.init_kl_list[i] * (
                        1 + b* math.log(temp2,2)/math.log(max2,2)*
                        (math.log(1 + temp1, 2) / math.log(1 + max1, 2)) *
                        (math.tan(math.pi*(a-1)/a/2*self.ratio[i]) / sum) * len(self.init_kl_list))
            self.kl_mult_temp[i] = self.kl_list[i].kl_mult / self.init_kl_list[i]


            # self.kl_list[i].kl_mult = self.init_kl_list[i] * (1 +  conv_kl_mult*(self.ratio[i] / sum ) )
            # self.kl_mult_temp[i] = self.kl_list[i].kl_mult / self.init_kl_list[i]

            # temp = len(self.init_kl_list) / 2 - abs(i - len(self.init_kl_list) / 2)
            # max = len(self.init_kl_list) / 2
            # self.kl_list[i].kl_mult = self.init_kl_list[i] * (1 +  temp/max * (self.ratio[i] * self.ratio[i]/ sum )* len(self.init_kl_list) )
            # self.kl_mult_temp[i] = self.kl_list[i].kl_mult / self.init_kl_list[i]

            # temp = len(self.init_kl_list) / 2 - abs(i - len(self.init_kl_list) / 2)
            # max = len(self.init_kl_list) / 2
            # self.kl_list[i].kl_mult = self.init_kl_list[i] * (
            #             1 + temp / max * (math.log(1+self.ratio[i], 2) / sum) * len(self.init_kl_list))
            # self.kl_mult_temp[i] = self.kl_list[i].kl_mult / self.init_kl_list[i]


        # for i in range(len(self.init_kl_list)):
        #     self.kl_list[i].kl_mult = self.init_kl_list[i] * (
        #                 1 + remain_stat[i]/max(remain_stat) * (self.ratio[i]  / sum )*len(self.init_kl_list) )
        #     self.kl_mult_temp[i] = self.kl_list[i].kl_mult / self.init_kl_list[i]


        # for i in range(len(self.kl_mult_temp)):
        #     self.kl_mult_temp[i] = self.kl_mult_temp[i] * math.sqrt(math.log(i + 2, 2))

        # for i in range(len(self.kl_mult_temp)):
        #         temp = len(self.init_kl_list)/2 - abs(i-len(self.init_kl_list)/2)
        #         max =  len(self.init_kl_list)/2
        #         self.kl_mult_temp[i] = self.kl_mult_temp[i] * (1 +  math.log(1+temp, 2)/math.log(1+max, 2))

        print('conv_kl_mult:',conv_kl_mult)
        print(b)
        print(self.ratio)
        print(self.init_kl_list)
        print(self.kl_mult_temp)



        # sum=0
        # for i in range(len(init_stat)):
        #     sump=remain_stat[i]/init_stat[i]
        #     sum=sum+sump
        #     # print('2 ', remain_stat)
        #     # print('3 ', init_stat)
        #
        # # conv1_kl_mult = 0.30
        # conv2_kl_mult = 0.60
        # conv3_kl_mult = 1.00
        # conv4_kl_mult = 0.60
        # conv5_kl_mult = 0.45
        # # conv6_kl_mult = 0.45
        #
        # for i in range(len(self.init_kl_list)):
        #     #调试每一层的权重
        #     #第八层容易产生突变，需要限制裁剪数量
        #
        #     dropout_penal = 1
        #
        #     if i==0 or i==1:
        #         # self.kl_list[i].kl_mult = self.init_kl_list[i] * ( 1+ conv1_kl_mult * remain_stat[i]/init_stat[i]/ sum * len(self.init_kl_list))
        #         # # self.kl_list[i].kl_mult = self.init_kl_list[i] * (0.5 * (remain_stat[i]/init_stat[i])*(remain_stat[i]/init_stat[i]) / sum * len(self.init_kl_list))
        #         # self.kl_mult_temp[i] = self.kl_list[i].kl_mult/self.init_kl_list[i]
        #         temp=1
        #     else:
        #         if i==2 or i==3:
        #             self.kl_list[i].kl_mult = self.init_kl_list[i] * (
        #                         1 + conv2_kl_mult * remain_stat[i] / init_stat[i] / sum * len(self.init_kl_list))
        #             # self.kl_list[i].kl_mult = self.init_kl_list[i] * (0.5 * (remain_stat[i]/init_stat[i])*(remain_stat[i]/init_stat[i]) / sum * len(self.init_kl_list))
        #             self.kl_mult_temp[i] = self.kl_list[i].kl_mult / self.init_kl_list[i]
        #         else:
        #             if i == 4 or i == 5 or i == 6 or i == 7:
        #                 self.kl_list[i].kl_mult = self.init_kl_list[i] * (
        #                         1 + conv3_kl_mult * remain_stat[i] / init_stat[i] / sum * len(self.init_kl_list))
        #                 # self.kl_list[i].kl_mult = self.init_kl_list[i] * (0.5 * (remain_stat[i]/init_stat[i])*(remain_stat[i]/init_stat[i]) / sum * len(self.init_kl_list))
        #                 self.kl_mult_temp[i] = self.kl_list[i].kl_mult / self.init_kl_list[i]
        #             else:
        #                 if i == 8 or i == 9 or i == 10 or i == 11:
        #                     if i==8:
        #                         # if remain_stat[i] / init_stat[i] < 0.25:
        #                         #     dropout_penal = 0.8
        #                         # if remain_stat[i] / init_stat[i] < 0.2:
        #                         #     dropout_penal = 0
        #                         self.kl_list[i].kl_mult = self.init_kl_list[i] * dropout_penal * (1 + conv4_kl_mult * remain_stat[i] / init_stat[i] / sum * len(self.init_kl_list))
        #                         self.kl_mult_temp[i] = self.kl_list[i].kl_mult / self.init_kl_list[i]
        #                     else:
        #                         self.kl_list[i].kl_mult = self.init_kl_list[i] * (1 + conv4_kl_mult * remain_stat[i] / init_stat[i] / sum * len(self.init_kl_list))
        #                         self.kl_mult_temp[i] = self.kl_list[i].kl_mult / self.init_kl_list[i]
        #                 else:
        #                     if i == 12 or i == 13 :
        #                     # if i == 12 or i == 13 or i == 14 or i == 15:
        #                         self.kl_list[i].kl_mult = self.init_kl_list[i] * (
        #                                 1 + conv5_kl_mult * remain_stat[i] / init_stat[i] / sum * len(self.init_kl_list))
        #                         # self.kl_list[i].kl_mult = self.init_kl_list[i] * (0.5 * (remain_stat[i]/init_stat[i])*(remain_stat[i]/init_stat[i]) / sum * len(self.init_kl_list))
        #                         self.kl_mult_temp[i] = self.kl_list[i].kl_mult / self.init_kl_list[i]
        #                     # else:
        #                     #     if i == 13 or i == 14:
        #                     #         self.kl_list[i].kl_mult = self.init_kl_list[i] * (
        #                     #                 1 + conv6_kl_mult * remain_stat[i] / init_stat[i] / sum * len(self.init_kl_list))
        #                     #         # self.kl_list[i].kl_mult = self.init_kl_list[i] * (0.5 * (remain_stat[i]/init_stat[i])*(remain_stat[i]/init_stat[i]) / sum * len(self.init_kl_list))
        #                     #         self.kl_mult_temp[i] = self.kl_list[i].kl_mult / self.init_kl_list[i]
        # print(self.kl_mult_temp)

    def adapt_dropout(self, p):
        conv_shapes = [v[0] for v in cfg[self.config] if type(v) is not str]

        if self.config in ['G', 'D6']:
            fc_shapes = [512]
        elif self.config == 'G5':
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

        if self.training and (not self.no_ib):
            ib_kld = self.kl_list[0].kld
            for ib in self.kl_list[1:]:
                ib_kld += ib.kld
                
                #modified by lxh
                #
                #v1 failed
                #A = np.array([ib.kld])
                #B = torch.autograd.Variable(torch.from_numpy(A))
                #ib_kld += B.double()
                #v2 failed
                #
                #ib.kld = ib.kld.type(torch.DoubleTensor)

            # print('x: ', x)
            # print('ib_kld: ', ib_kld)
            return x, ib_kld.float()
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

    def print_compression_ratio(self, threshold, writer=None, epoch=-1):
        # applicable for structures with global pooling before fc
        _, prune_stat = self.get_masks(hard_mask=True, threshold=threshold)
        conv_shapes = [v[0] for v in cfg[self.config] if type(v) is not str]

        if self.config in ['G', 'D6']:
            fc_shapes = [512]
        elif self.config == 'G5':
            fc_shapes = []
        else:
            fc_shapes = [512, 512]

        # self.pruned_structure = [self.last_prune_stat[i] - prune_stat[i] for i in range(len(prune_stat))]
        # self.last_prune_stat = prune_stat

        # print('prune_stat_size: ', prune_stat.size())
        # for i in range(len(prune_stat)):
        #     print('total parameters: ', prune_stat[i].size())

        self.pruned_structure = [prune_stat[idx] - self.last_prune_stat[idx] for idx, out_channels in enumerate(conv_shapes + fc_shapes)]
        self.last_prune_stat = [prune_stat[idx] for idx, out_channels in enumerate(conv_shapes + fc_shapes)]
        net_shape = [ out_channels-prune_stat[idx] for idx, out_channels in enumerate(conv_shapes+fc_shapes)]
        #conv_shape_with_pool = [v[0] if v != 'M' else 'M' for v in cfg[self.config]]
        current_n, hdim, last_channels, flops, fmap_size = 0, 64, 3, 0, 32
        for n, pruned_channels in enumerate(prune_stat):
            if n < len(conv_shapes):
                current_channels = cfg[self.config][current_n][0] - pruned_channels
                flops += (fmap_size**2) * 9 * last_channels * current_channels
                last_channels = current_channels
                current_n += 1
                if type(cfg[self.config][current_n]) is str:
                    current_n += 1
                    fmap_size /= 2
                    hdim *= 2
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

        print('total parameters: {}, pruned parameters: {}, remaining params:{}, remain/total params:{}, remaining flops: {}, remaining flops/params: {},'
              'each layer pruned: {}, this epoch each layer pruned: {}, remaining structure:{}'.format(total_params, pruned_params, remain_params,
                    float(total_params-pruned_params)/total_params, flops, 0.0000000001 * flops/(float(total_params-pruned_params)/total_params), prune_stat, self.pruned_structure, net_shape))

        if writer is not None:
            writer.add_scalar('flops', flops, epoch)
            writer.add_scalar('remain/total params', float(total_params-pruned_params)/total_params, epoch)
            writer.add_scalar('flops/remaining params', 0.0000000001 * flops/(float(total_params-pruned_params)/total_params), epoch)
