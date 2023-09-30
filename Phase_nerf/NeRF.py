# DECAF training and predicting model with parallelization
# Created by Renhao Liu and Yu Sun, CIG, WUSTL, 2021

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import skimage
from skimage.metrics import peak_signal_noise_ratio
import cv2
import math
import time
import gc
from absl import flags
import logging

FLAGS = flags.FLAGS

NUM_Z = "nz"
INPUT_CHANNEL = "ic"
OUTPUT_CHANNEL = "oc"
MODEL_SCOPE = "infer_y"
NET_SCOPE = "MLP"
DNCNN_SCOPE = "DnCNN"

# get total number of visible gpus
NUM_GPUS = torch.cuda.device_count()


########################################
###       Tensorboard & Helper       ###
########################################

def record_summary(writer, name, value, step):
    writer.add_scalar(name, value, step)
    writer.flush()


def reshape_image(image):
    if len(image.shape) == 2:
        image_reshaped = image.unsqueeze(0).unsqueeze(-1)
    elif len(image.shape) == 3:
        image_reshaped = image.unsqueeze(-1)
    else:
        image_reshaped = image
    return image_reshaped


def reshape_image_2(image):
    image_reshaped = image.unsqueeze(0).unsqueeze(-1)
    return image_reshaped


def reshape_image_3(image):
    image_reshaped = image.unsqueeze(-1)
    return image_reshaped


def reshape_image_5(image):
    shape = image.shape
    image_reshaped = image.view(-1, shape[2], shape[3], 1)
    return image_reshaped


#################################################
# ***      CLASS OF NEURAL REPRESENTATION     ****
#################################################


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class NPRF(nn.Module):
    def __init__(self, FLAGS, net_kargs=None, name="model_summary"):
        super(DeCAF, self).__init__()
        # Setup parameters
        self.name = name
        self.tf_summary_dir = "{}/{}".format(FLAGS.tf_summary_dir, name)
        if net_kargs is None:
            self.net_kargs = {
                "skip_layers": FLAGS.mlp_skip_layer,
                "mlp_layer_num": FLAGS.mlp_layer_num,
                "kernel_size": FLAGS.mlp_kernel_size,
                "L_xy": FLAGS.xy_encoding_num,
                "L_z": FLAGS.z_encoding_num,
            }
        else:
            self.net_kargs = net_kargs
        print('FLAGS.mlp_skip_layer', FLAGS.mlp_skip_layer)
        input_dim = FLAGS.xy_encoding_num * 24 + FLAGS.z_encoding_num * 2
        # input_dim = 200
        output_dim = 2
        # FLAGS.mlp_kernel_size=200
        # FLAGS.mlp_kernel_size=208
        self.inputlayer = nn.Linear(input_dim, FLAGS.mlp_kernel_size)
        self.skiplayer = nn.Linear(FLAGS.mlp_kernel_size + input_dim, FLAGS.mlp_kernel_size)
        # FLAGS.mlp_layer_num=6
        self.lineares = nn.ModuleList(
            [nn.Linear(FLAGS.mlp_kernel_size, FLAGS.mlp_kernel_size) for i in range(FLAGS.mlp_layer_num)])
        self.outputlayer = nn.Linear(FLAGS.mlp_kernel_size, output_dim)
        self.le_relu = nn.LeakyReLU(negative_slope=0.2, inplace=False)
        self.sigmoid = nn.Sigmoid()

    ###########################
    ###     Neural Nets     ###
    ###########################

    def forward(self, input, padding=None, mask=None, reuse=False, training=True, epochs=0):
        self.training = training
        self.epochs = epochs
        # MLP network

        coordinates=input[:,:,0:3]
        light_source=input[:,:,3:]

        if self.training:
            # time1 = time.time()
            reflactive_index = self.__neural_repres(coordinates)
            intensity = self.rendering(light_source, reflactive_index)
            # time2 = time.time()
            # print(f"Time| camera: {time2-time1:.5f}")
            return reflactive_index, intensity
        else:
            #synthesis images
            pass
        return Hxhat, xhat

    def __neural_repres(self, in_node, skip_layers=[], mlp_layer_num=10, kernel_size=256, L_xy=6, L_z=5):
        # positional encoding
        if FLAGS.positional_encoding_type == "exp_diag":
            s = torch.sin(torch.arange(0, 180, FLAGS.dia_digree) * np.pi / 180)[:, None]
            c = torch.cos(torch.arange(0, 180, FLAGS.dia_digree) * np.pi / 180)[:, None]
            fourier_mapping = torch.cat((s, c), dim=1).T
            fourier_mapping = fourier_mapping.to('cuda').cuda()
            xy_freq = torch.matmul(in_node[:, :2], fourier_mapping)

            for l in range(L_xy):
                cur_freq = torch.cat(
                    [
                        torch.sin(2 ** l * np.pi * xy_freq),
                        torch.cos(2 ** l * np.pi * xy_freq),
                    ],
                    dim=-1,
                )
                if l == 0:
                    tot_freq = cur_freq
                else:
                    tot_freq = torch.cat([tot_freq, cur_freq], dim=-1)
            # print('z', in_node[:, 2].max(), in_node[:, 2].min(), in_node[:, 2].mean())
            for l in range(L_z):
                cur_freq = torch.cat(
                    [
                        torch.sin(2 ** l * np.pi * in_node[:, 2].unsqueeze(-1)),
                        torch.cos(2 ** l * np.pi * in_node[:, 2].unsqueeze(-1)),
                    ],
                    dim=-1,
                )

                tot_freq = torch.cat([tot_freq, cur_freq], dim=-1)

        else:
            raise NotImplementedError(FLAGS.positional_encoding_type)

        # input to MLP

        in_node = tot_freq
        x = self.inputlayer(in_node)
        x = self.le_relu(x)

        layer_cout = 1
        for f in self.lineares:
            layer_cout += 1
            if layer_cout in skip_layers:
                x = torch.cat([x, tot_freq], -1)
                x = self.skiplayer(x)
                x = self.le_relu(x)
            x = f(x)
            # print('x min a:',x.min())
            x = self.le_relu(x)
            # print('x min b:',x.min())
        x = self.outputlayer(x)
        output = self.le_relu(x)

        output = output / FLAGS.output_scale


        return output

    def rendering(self, light_source, reflactive_index):
        if self.training:
            padding = tuple(padding)

            padded_field = F.pad(x, (0, 0, padding[3][0], padding[3][0], padding[3][0], padding[3][0]))
            padded_phase = padded_field[:, :, :, :, 0]
            padded_absorption = padded_field[:, :, :, :, 1]

            transferred_field = torch.fft.ifft2(
                torch.mul(Hreal, torch.fft.fft2(padded_phase.expand(Hreal.shape).to(torch.complex64)))
                + torch.mul(Himag, torch.fft.fft2(padded_absorption.expand(Himag.shape).to(torch.complex64)))
            )

            Hxhat = torch.sum(torch.real(transferred_field), dim=(1, 2))

            return Hxhat

        else:
            return x

    def save(self, directory, epoch=None, train_provider=None):
        if epoch is not None:
            directory = os.path.join(directory, "{}_model/".format(epoch))
        else:
            directory = os.path.join(directory, "latest/".format(epoch))
        if not os.path.exists(directory):
            os.makedirs(directory)
        path = os.path.join(directory, "model")
        if train_provider is not None:
            train_provider.save(directory)
        torch.save(self.state_dict(), path)
        print("saved to {}".format(path))
        return path

    def restore(self, model_path):

        param = torch.load(model_path)
        # param_model=self.state_dict()
        # new_dict={}
        # for k,v in param.items():
        #     if k in param_model:
        #         print(k)
        #         print(v)
        self.load_state_dict(param, strict=False)

        # self.load_state_dict(torch.load(model_path),strict=False)






