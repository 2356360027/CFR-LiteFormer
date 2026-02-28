import itertools
import pdb
import random
import torch
import torch.nn as nn
from tqdm.autonotebook import tqdm
from model.BrownianBridge.loss_style import *
from model.BrownianBridge.BrownianBridgeModel import BrownianBridgeModel
from model.BrownianBridge.base.modules.encoders.modules import SpatialRescaler
from model.VQGAN.vqgan import VQModel
from model.BrownianBridge.enet import *
from thop import profile
from thop import clever_format

from model.mutli_AE.Combine_AE import Combine_AE

from model.mutli_AE.Latent_Decoder import  Latent_Decoder
def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class LatentBrownianBridgeModel(BrownianBridgeModel):
    def __init__(self, model_config):
        super().__init__(model_config)

        self.vqgan = VQModel(**vars(model_config.VQGAN.params)).eval()

        # self.unshuffle_1=nn.PixelUnshuffle(downscale_factor=2)
        # self.unshuffle_2 = nn.PixelUnshuffle(downscale_factor=2)
        # self.down_conv1=nn.Conv2d(12,3,kernel_size=1,stride=1,padding=0)
        # self.down_conv2=nn.Conv2d(12,3,kernel_size=1,stride=1,padding=0)
        #
        #
        #
        #
        # self.shuffle1=nn.PixelShuffle(upscale_factor=2)
        # self.up_conv1 = nn.Conv2d(3, 12, kernel_size=1, stride=1, padding=0)
        # self.up_conv2 = nn.Conv2d(3, 12, kernel_size=1, stride=1, padding=0)
        # self.shuffle2 = nn.PixelShuffle(upscale_factor=2)

        # self.conv_down = nn.Conv2d(
        #     in_channels=3,
        #     out_channels=3,
        #     kernel_size=4,
        #     stride=4,
        #     bias=False,
        # )
        #
        # self.conv_up = self.conv_up = nn.ConvTranspose2d(
        #     in_channels=3,
        #     out_channels=3,
        #     kernel_size=4,
        #     stride=4,
        #     bias=False,
        # )
        # self.EncoderUnet=EncoderNet(
        #     in_channel=3,
        #     out_channel=3,
        #     inner_channel=32,
        #     channel_mults=[12, 12],
        #     norm_groups=32,
        #     res_blocks=2,
        #     dropout=0.1,
        #     image_size=256
        # )
        self.vqgan.train = disabled_train
        # self.combine_ae = Combine_AE(img_size=256, img_channels=3)
        # self.combine_ae.load_state_dict(torch.load(
        #     '/home/nwu-kiki/mydisk/pycharmprojects/BBDM+Trasformer+light/BBDM-main/model/mutli_AE/lg_ae/outputs/0628/350.pt'))  # './sketch_ae/outputs/1028.pt'
        # self.combine_ae.eval()
        # self.latent_decoder = Latent_Decoder(latent_size=64, latent_channels=3)

        # Num_Param_decoder = sum(p.numel() for p in self.latent_decoder.parameters() if p.requires_grad)
        for param in self.vqgan.parameters():
            param.requires_grad = False
        print(f"load vqgan from {model_config.VQGAN.params.ckpt_path}")
        blocks = [0, 1, 2]
        weights = [1, 0.8, 0.6]
        self.loss_per = PerceptualLoss(blocks, weights)
        self.loss_text = TextureLoss(blocks, weights)

        # Condition Stage Model
        if self.condition_key == 'nocond':
            self.cond_stage_model = None
        elif self.condition_key == 'first_stage':
            self.cond_stage_model = self.vqgan
        elif self.condition_key == 'SpatialRescaler':
            self.cond_stage_model = SpatialRescaler(**vars(model_config.CondStageParams))
        else:
            raise NotImplementedError

    def get_ema_net(self):
        return self

    def get_parameters(self):
        if self.condition_key == 'SpatialRescaler':
            print("get parameters to optimize: SpatialRescaler, UNet")
            params = itertools.chain(self.denoise_fn.parameters(), self.cond_stage_model.parameters())
        else:
            print("get parameters to optimize: UNet")
            params = self.denoise_fn.parameters()
        return params

    def apply(self, weights_init):
        super().apply(weights_init)
        if self.cond_stage_model is not None:
            self.cond_stage_model.apply(weights_init)
        return self

    def forward(self, x, x_cond,step, context=None):

        # x_latent = self.unshuffle_1(x)
        # x_latent = self.down_conv1(x_latent)
        # x_latent = self.unshuffle_2(x_latent)
        # x_latent = self.down_conv2(x_latent)
        #
        # x_cond_latent = self.unshuffle_1(x_cond)
        # x_cond_latent = self.down_conv1(x_cond_latent)
        # x_cond_latent = self.unshuffle_2(x_cond_latent)
        # x_cond_latent = self.down_conv2(x_cond_latent)
        # print(x_cond_latent.shape)

        # x_cond_latent = self.up_conv1(x_cond)
        #
        # x_cond_latent = self.shuffle1(x_cond_latent)
        # x_cond_latent = self.up_conv2(x_cond_latent)
        # x_cond_latent = self.shuffle2(x_cond_latent)



        with torch.no_grad():
            x_latent = self.encode(x, cond=False)
            x_cond_latent = self.encode(x_cond, cond=True)
        context = self.get_cond_stage_context(x_cond)
        # map = self.latent_decoder(self.combine_ae.encode(x_cond))
        # print("map", map.shape)
        # # print(x_latent.shape, x_cond_latent.shape)
        # print("-----------")
        return super().forward(x_latent.detach(), x_cond_latent.detach(), context,step)

    def get_cond_stage_context(self, x_cond):
        if self.cond_stage_model is not None:
            context = self.cond_stage_model(x_cond)
            if self.condition_key == 'first_stage':
                context = context.detach()
        else:
            context = None
        return context

    @torch.no_grad()
    def encode(self, x, cond=True, normalize=None):
        normalize = self.model_config.normalize_latent if normalize is None else normalize
        model = self.vqgan
        x_latent = model.encoder(x)
        if not self.model_config.latent_before_quant_conv:
            x_latent = model.quant_conv(x_latent)
        if normalize:
            if cond:
                x_latent = (x_latent - self.cond_latent_mean) / self.cond_latent_std
            else:
                x_latent = (x_latent - self.ori_latent_mean) / self.ori_latent_std
        return x_latent

    @torch.no_grad()
    def decode(self, x_latent, cond=True, normalize=None):
        normalize = self.model_config.normalize_latent if normalize is None else normalize
        if normalize:
            if cond:
                x_latent = x_latent * self.cond_latent_std + self.cond_latent_mean
            else:
                x_latent = x_latent * self.ori_latent_std + self.ori_latent_mean
        model = self.vqgan
        if self.model_config.latent_before_quant_conv:
            x_latent = model.quant_conv(x_latent)
        x_latent_quant, loss, _ = model.quantize(x_latent)

        out = model.decode(x_latent_quant)
        return out

    @torch.no_grad()
    def sample(self, x_cond, clip_denoised=False, sample_mid_step=False):
        x_cond_latent = self.encode(x_cond, cond=True)
        if sample_mid_step:
            temp, one_step_temp = self.p_sample_loop(y=x_cond_latent,
                                                     context=self.get_cond_stage_context(x_cond),
                                                     clip_denoised=clip_denoised,
                                                     sample_mid_step=sample_mid_step)
            out_samples = []
            for i in tqdm(range(len(temp)), initial=0, desc="save output sample mid steps", dynamic_ncols=True,
                          smoothing=0.01):
                with torch.no_grad():
                    out = self.decode(temp[i].detach(), cond=False)
                out_samples.append(out.to('cpu'))

            one_step_samples = []
            for i in tqdm(range(len(one_step_temp)), initial=0, desc="save one step sample mid steps",
                          dynamic_ncols=True,
                          smoothing=0.01):
                with torch.no_grad():
                    out = self.decode(one_step_temp[i].detach(), cond=False)
                one_step_samples.append(out.to('cpu'))
            return out_samples, one_step_samples
        else:
            temp = self.p_sample_loop(y=x_cond_latent,
                                      context=self.get_cond_stage_context(x_cond),
                                      clip_denoised=clip_denoised,
                                      sample_mid_step=sample_mid_step)
            x_latent = temp
            out=self.decode(x_latent)
            return out

    @torch.no_grad()
    def sample_vqgan(self, x):
        x_rec, _ = self.vqgan(x)
        return x_rec

    # @torch.no_grad()
    # def reverse_sample(self, x, skip=False):
    #     x_ori_latent = self.vqgan.encoder(x)
    #     temp, _ = self.brownianbridge.reverse_p_sample_loop(x_ori_latent, x, skip=skip, clip_denoised=False)
    #     x_latent = temp[-1]
    #     x_latent = self.vqgan.quant_conv(x_latent)
    #     x_latent_quant, _, _ = self.vqgan.quantize(x_latent)
    #     out = self.vqgan.decode(x_latent_quant)
    #     return out
