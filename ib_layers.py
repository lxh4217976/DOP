import math

import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch import nn
from torch.nn.modules import Module
from torch.autograd import Variable
from torch.nn.modules import utils
import numpy as np
import pdb

def reparameterize(mu, logvar, batch_size, cuda=False, sampling=True,a=0.5,b=0.5):
    # output dim: batch_size * dim   128*64
    if sampling:
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(batch_size, std.size(0)).cuda(mu.get_device()).normal_()
        eps = Variable(eps)

        # rand_data = np.random.laplace(0, 1, batch_size * std.size(0))
        # rand_data = rand_data.reshape(batch_size, -1)
        # # dert = Variable(torch.from_numpy(rand_data).cuda(mu.get_device()).double())
        # dert = Variable(torch.from_numpy(rand_data).cuda(mu.get_device()))

        # return (mu.view(1, -1).double() + (0.5 * dert.double() + 0.5 * eps.double()) * std.view(1, -1).double()).float()

        # return (mu.view(1, -1).double() + (0.67 * dert.double() + 0.33 * eps.double()) * std.view(1, -1).double()).float()

        # return (mu.view(1, -1).double() + (0.8 * dert.double() + 0.2 * eps.double()) * std.view(1, -1).double()).float()

        # return (mu.view(1, -1).double() + (0.33 * dert.double() + 0.67 * eps.double()) * std.view(1, -1).double()).float()

        # print('mu: ', mu)
        # print('mu: ', mu)

        return mu.view(1, -1) +  eps * std.view(1, -1)

    else:
        return mu.view(1, -1)

class InformationBottleneck(Module):
    def __init__(self, dim, mask_thresh=0, init_mag=9, init_var=0.01,
                kl_mult=1, divide_w=False, sample_in_training=True, sample_in_testing=False, masking=False,a=0.5,b=0.5):
        super(InformationBottleneck, self).__init__()
        self.prior_z_logD = Parameter(torch.Tensor(dim))
        self.post_z_mu = Parameter(torch.Tensor(dim))
        self.post_z_logD = Parameter(torch.Tensor(dim))

        self.epsilon = 1e-8
        self.dim = dim
        self.sample_in_training = sample_in_training
        self.sample_in_testing = sample_in_testing
        # if masking=True, apply mask directly
        self.masking = masking
        self.offset=0.00
        # self.p=0.99
        self.p = 1
        self.dropout_threshold = 0.01

        # activations for kl
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()

        # initialization

        # stdv = 1. / math.sqrt(dim)
        # self.post_z_mu.data.normal_(1, init_var)
        # self.prior_z_logD.data.normal_(-init_mag, init_var)
        # self.post_z_logD.data.normal_(-init_mag, init_var)



        # rand_data1 = np.random.laplace(1, math.sqrt(init_var / 2), dim)


        # self.a = 0.5
        # self.b = 0.5

        # self.a = 0.33
        # self.b = 0.67

        # self.a = 0.2
        # self.b = 0.8

        self.a = 0.67
        self.b = 0.33

        # 拉普拉斯和高斯分布之和
        self.post_z_mu.data.normal_(1, init_var)
        self.post_z_logD.data.normal_(-init_mag, init_var)

        # 拉普拉斯分布

        # rand_data1 = np.random.laplace(1, init_var, dim)
        #
        # self.post_z_mu.data = self.a * self.post_z_mu.data.cuda().double() + self.b * torch.from_numpy(rand_data1).cuda()
        #
        # # rand_data2 = np.random.laplace(-init_mag, math.sqrt(init_var/2), dim)
        #
        # rand_data2 = np.random.laplace(-init_mag, init_var, dim)
        #
        # self.post_z_logD.data = self.a * self.post_z_logD.data.cuda().double() + self.b * torch.from_numpy(rand_data2).cuda()


        self.need_update_z = True # flag for updating z during testing
        self.mask_thresh = mask_thresh
        self.kl_mult=kl_mult
        self.divide_w=divide_w




    def adapt_shape(self, src_shape, x_shape):
        if len(src_shape) == 2:
            new_shape = src_shape
            # print('new_shape1: ',new_shape)
        else:
            new_shape = (1, src_shape[0])
            # print('new_shape2: ', new_shape)
        if len(x_shape)>2:
            new_shape = list(new_shape)
            new_shape += [1 for i in range(len(x_shape)-2)]
            # print('new_shape3: ', new_shape)
        return new_shape



    def get_logalpha(self):
        # return self.post_z_logD.data - torch.log(self.post_z_mu.data.pow(2) + self.epsilon)

        log_lambda = 0.5 * torch.log(torch.exp(self.post_z_logD)/2)
        z = torch.log(self.post_z_mu.data.abs() + self.epsilon)

        # print('post_z_logD: ', self.post_z_logD.data)
        #
        # print('post_z_mu: ', torch.log(self.post_z_mu.data.pow(2) + self.epsilon))

        # return  self.b * (log_lambda - z) + self.a * (self.post_z_logD.data - torch.log(self.post_z_mu.data.pow(2) + self.epsilon) )
        return  self.post_z_logD.data - torch.log(self.post_z_mu.data.pow(2) + self.epsilon)

    def get_dp(self):
        logalpha = self.get_logalpha()
        alpha = torch.exp(logalpha)
        return alpha / (1+alpha)

    def get_mask_hard(self, threshold=0):
        logalpha = self.get_logalpha()
        hard_mask = (logalpha < threshold).float()
        # print('logalpha: ', logalpha)
        # print('hard_mask: ', hard_mask)
        return hard_mask

    def get_mask_weighted(self, threshold=0):
        logalpha = self.get_logalpha()
        # mask = (logalpha < threshold).float()*self.post_z_mu.data

        mask = (logalpha < threshold).float() * self.post_z_mu.data.float()
        return mask

    # def get_mask_dropout(self, threshold=0, p=0.95):
    #     logalpha = self.get_logalpha()
    #     mask = (logalpha < threshold).float() * self.post_z_mu.data.float()
    #     return mask

    def adapt_dropout(self, p):
        self.p=p


    def forward(self, x):
        # 4 modes: sampling, hard mask, weighted mask, use mean value
        if self.masking:
            mask = self.get_mask_hard(self.mask_thresh + self.offset)
            new_shape = self.adapt_shape(mask.size(), x.size())



            return x * Variable(mask.view(new_shape))

        bsize = x.size(0)
        if (self.training and self.sample_in_training) or (not self.training and self.sample_in_testing):
            z_scale = reparameterize(self.post_z_mu, self.post_z_logD, bsize, cuda=True, sampling=True,a=self.a,b=self.b)

            if self.p<1:
                rand_data3 = np.random.binomial(1, self.p, size=z_scale.size())

                dropout = torch.from_numpy(rand_data3).cuda().float()

                z_scale *= Variable(self.get_mask_hard(self.mask_thresh + self.offset))

                z_scale *= dropout
            else:
                z_scale *= Variable(self.get_mask_hard(self.mask_thresh + self.offset))

            if not self.training:
                z_scale *= Variable(self.get_mask_hard(self.mask_thresh + self.offset))

                #这个没有输出
        else:
            z_scale = Variable(self.get_mask_weighted(self.mask_thresh + self.offset))


        self.kld = self.kl_closed_form(x)

        new_shape = self.adapt_shape(z_scale.size(), x.size())

        return x * z_scale.view(new_shape)
        # return x


    def kl_closed_form(self, x):
        new_shape = self.adapt_shape(self.post_z_mu.size(), x.size())

        #origial code
        h_D = torch.exp(self.post_z_logD.view(new_shape))
        h_mu = self.post_z_mu.view(new_shape)

        # h_std = torch.exp(self.post_z_logD.mul(0.5).view(new_shape))
        #
        # h_lambda = torch.exp(self.post_z_logD.mul(0.5).view(new_shape))* math.sqrt(1/2)
        # # h_lambda = torch.exp(self.post_z_logD.mul(0.5).view(new_shape))


        KLD = torch.sum( torch.log(1 + h_mu.pow(2) / (h_D + self.epsilon)) ) * x.size(1) / h_D.size(1)

        # KLD = self.a * KLD1 + 2.0 * self.b * torch.sum( torch.log(1 + h_mu.abs() / (h_lambda + self.epsilon)) ) * x.size(1) / h_D.size(1)

        if x.dim() > 2:
            if self.divide_w:
                # divide it by the width
                KLD *= x.size()[2]
               
            else:
                KLD *= int(np.prod(x.size()[2:]))
                
            '''   
            else:
                try:
                    KLD = KLD.item()
                    KLD *= np.prod(x.size()[2:])
                except:
                    from IPython import embed; embed()
            '''

        return KLD * 0.5 * self.kl_mult


