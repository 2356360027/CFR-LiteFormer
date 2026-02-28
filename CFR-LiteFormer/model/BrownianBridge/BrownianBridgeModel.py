import math
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


from torchvision.utils import make_grid, save_image
from matplotlib import pyplot as plt
from tqdm.autonotebook import tqdm
import numpy as np
from model.BrownianBridge.base.modules.diffusionmodules.ditmodel import DiT_models
from model.BrownianBridge.loss_style import *
from model.utils import extract, default
from model.BrownianBridge.loss_style import *
from thop import profile
from thop import clever_format
from model.BrownianBridge.base.modules.diffusionmodules.openaimodel import UNetModel
from model.BrownianBridge.base.modules.encoders.modules import SpatialRescaler
from model.BrownianBridge.base.modules.efficient_vit.model.build import EfficientViT_M4
import model.BrownianBridge.base.modules.repvitmodel
from model.BrownianBridge.base.modules.diffusionmodules.ncsnpp_generator_adagn import WaveletNCSNpp
class BrownianBridgeModel(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.model_config = model_config
        # model hyperparameters
        model_params = model_config.BB.params
        self.num_timesteps = model_params.num_timesteps
        self.mt_type = model_params.mt_type
        self.max_var = model_params.max_var if model_params.__contains__("max_var") else 1
        self.eta = model_params.eta if model_params.__contains__("eta") else 1
        self.skip_sample = model_params.skip_sample
        self.sample_type = model_params.sample_type
        self.sample_step = model_params.sample_step
        self.steps = None
        self.register_schedule()

        # loss and objective
        self.loss_type = model_params.loss_type
        self.objective = model_params.objective

        # UNet
        self.image_size =64
        self.channels = model_params.UNetParams.in_channels
        self.condition_key = "nocond"

        # self.denoise_fn = UNetModel(**vars(model_params.UNetParams))
        # self.denoise_fn =  EfficientViT_M4()
        # self.denoise_fn = WaveletNCSNpp(model_params.UNetParams)
        # blocks = [0, 1, 2]
        # weights = [1, 0.8, 0.6]
        # self.loss_per = PerceptualLoss(blocks, weights)
        # self.loss_text = TextureLoss(blocks, weights)

        self.denoise_fn =DiT_models[model_params.DitParams.ditmodel](
        input_size=64,
        )


        # self.denoise_fn = repvit_m1_5( )
        print(self.denoise_fn)
        # flops = FlopCountAnalysis(self.denoise_fn, torch.rand(1, 3, 64, 64))
        # print("总flops："+flop_count_table(flops))
        self.lamda_1=0.1
        self.lamda_2=1
        self.lamda_3=100

    def register_schedule(self):
        T = self.num_timesteps

        if self.mt_type == "linear":
            m_min, m_max = 0.001, 0.999
            m_t = np.linspace(m_min, m_max, T)
        elif self.mt_type == "sin":
            m_t = 1.0075 ** np.linspace(0, T, T)
            m_t = m_t / m_t[-1]
            m_t[-1] = 0.999
        else:
            raise NotImplementedError
        m_tminus = np.append(0, m_t[:-1])

        variance_t = 2. * (m_t - m_t ** 2) * self.max_var
        variance_tminus = np.append(0., variance_t[:-1])
        variance_t_tminus = variance_t - variance_tminus * ((1. - m_t) / (1. - m_tminus)) ** 2
        posterior_variance_t = variance_t_tminus * variance_tminus / variance_t

        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.register_buffer('m_t', to_torch(m_t))
        self.register_buffer('m_tminus', to_torch(m_tminus))
        self.register_buffer('variance_t', to_torch(variance_t))
        self.register_buffer('variance_tminus', to_torch(variance_tminus))
        self.register_buffer('variance_t_tminus', to_torch(variance_t_tminus))
        self.register_buffer('posterior_variance_t', to_torch(posterior_variance_t))

        if self.skip_sample:
            if self.sample_type == 'linear':
                midsteps = torch.arange(self.num_timesteps - 1, 1,
                                        step=-((self.num_timesteps - 1) / (self.sample_step - 2))).long()
                self.steps = torch.cat((midsteps, torch.Tensor([1, 0]).long()), dim=0)
            elif self.sample_type == 'cosine':
                steps = np.linspace(start=0, stop=self.num_timesteps, num=self.sample_step + 1)
                steps = (np.cos(steps / self.num_timesteps * np.pi) + 1.) / 2. * self.num_timesteps
                self.steps = torch.from_numpy(steps)
        else:
            self.steps = torch.arange(self.num_timesteps-1, -1, -1)

    def apply(self, weight_init):
        self.denoise_fn.apply(weight_init)
        return self

    def get_parameters(self):
        return self.denoise_fn.parameters()

    def forward(self, x, y, context=None,step=None):
        if self.condition_key == "nocond":
            context = None
        else:
            context = y if context is None else context
        b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(x, y, context, t,step)

    def p_losses(self, x0, y, context, t, step,noise=None):
        """
        model loss
        :param x0: encoded x_ori, E(x_ori) = x0
        :param y: encoded y_ori, E(y_ori) = y
        :param y_ori: original source domain image
        :param t: timestep
        :param noise: Standard Gaussian Noise
        :return: loss
        """
        b, c, h, w = x0.shape
        noise = default(noise, lambda: torch.randn_like(x0))
        path="../data_mid.png"
        path_1="../data_objective.png"
        # print(step)


        # print("map",map.shape)

        # print(y.shape)
        # print(x0.shape)

        x_t, objective = self.q_sample(x0, y, t, noise)
        # print(x_t.shape)

        # objective_recon = self.denoise_fn(x_t, t)
        # latent_z = torch.randn(1, 100).to(x_t.device)


        save_image(x_t,path,format="PNG")
        save_image(objective, path_1, format="PNG")
        # # #
        # flops, params = profile(self.denoise_fn, inputs=(x_t, t))
        # flops, params = clever_format([flops, params], "%.3f")
        # print(flops, params)



        objective_recon = self.denoise_fn(x_t,t)
        # objective_recon=torch.tensor(objective_recon)
        # print(objective.shape)#([4, 4, 32, 32])
        # print(objective_recon.shape)  # ([4, 8, 32, 32])


        if self.loss_type == 'l1':

            # alpha=1/(1+self.lamda_1*step)
            # beta=t/(1+self.lamda_2*step)
            # gamma=1-alpha-beta
            loss_1= (objective - objective_recon).abs().mean()
            loss_per = self.loss_per(objective, objective_recon)
            # loss_tex = self.loss_text(objective, objective_recon)



            # if step!=0:
            #
            #
            #    # loss = 0.05*step * loss_per + (step * 10) * loss_tex + (1-(step/200))*loss_1
            #    loss=loss_per+loss_tex+(1-0.1*np.log(0.05*(step)+1))*loss_1
            #    print(f'step {step} loss_l1  { (1-0.1*np.log(0.05*(step)+1))*loss_1} loss_per {loss_per } loss_tex {loss_tex}')
            # else:
            #     loss = (math.exp(0.003 * step)) * loss_per + ((100 * math.exp(0.04 * step))) * loss_tex + (
            #         math.exp(-0.004 * step)) * loss_1
            #     print(
            #         f'step {step} loss_l1  {(math.exp(-0.004 * step) * loss_1)} loss_per {(math.exp(0.003 * step)) * loss_per} loss_tex {((100 * math.exp(0.04 * step))) * loss_tex}')





            # print(f'step {step} loss_all {loss.item()}  loss_l1 {loss_1}loss_per {loss_per} loss_tex {loss_tex}')
            # if step!=0  and (step ) % 800==0:
            #
            #     print(f'step {step} loss_l1  {0.6+(1/step)}loss_per { 0.1*step * loss_per} loss_tex {(step * step) * loss_tex}')

            recloss=loss_1+loss_per

            # print(recloss)

        elif self.loss_type == 'l2':
            recloss = F.mse_loss(objective, objective_recon)
        else:
            raise NotImplementedError()

        x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon)
        log_dict = {
            "loss": recloss,
            "x0_recon": x0_recon
        }
        return recloss, log_dict

    # 定义一个方法来可视化张量
    def visualize_tensor(tensor):
        tensor = tensor.squeeze().cpu().detach().numpy()  # 转换为 numpy 数组，并移除可能的单维度
        plt.imshow(tensor, cmap='gray')  # 使用灰度色彩映射显示图像
        plt.axis('off')  # 不显示坐标轴
        plt.show()

    def q_sample(self, x0, y, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x0))
        m_t = extract(self.m_t, t, x0.shape)
        var_t = extract(self.variance_t, t, x0.shape)
        sigma_t = torch.sqrt(var_t)

        if self.objective == 'grad':
            objective = m_t * (y - x0) + sigma_t * noise
        elif self.objective == 'noise':
            objective = noise
        elif self.objective == 'ysubx':
            objective = y - x0
        else:
            raise NotImplementedError()

        return (
            (1. - m_t) * x0 + m_t * y + sigma_t * noise,
            objective
        )

    def predict_x0_from_objective(self, x_t, y, t, objective_recon):
        if self.objective == 'grad':
            x0_recon = x_t - objective_recon
        elif self.objective == 'noise':
            m_t = extract(self.m_t, t, x_t.shape)
            var_t = extract(self.variance_t, t, x_t.shape)
            sigma_t = torch.sqrt(var_t)
            x0_recon = (x_t - m_t * y - sigma_t * objective_recon) / (1. - m_t)
        elif self.objective == 'ysubx':
            x0_recon = y - objective_recon
        else:
            raise NotImplementedError
        return x0_recon

    @torch.no_grad()
    def q_sample_loop(self, x0, y):
        imgs = [x0]
        for i in tqdm(range(self.num_timesteps), desc='q sampling loop', total=self.num_timesteps):
            t = torch.full((y.shape[0],), i, device=x0.device, dtype=torch.long)
            img, _ = self.q_sample(x0, y, t)
            imgs.append(img)
        return imgs

    @torch.no_grad()
    def p_sample(self, x_t, y, context, i, clip_denoised=False):
        b, *_, device = *x_t.shape, x_t.device
        if self.steps[i] == 0:
            t = torch.full((x_t.shape[0],), self.steps[i], device=x_t.device, dtype=torch.long)
            # objective_recon = self.denoise_fn(x_t, timesteps=t, context=context)
            # objective_recon = self.denoise_fn(x_t, t)
            # latent_z = torch.randn(1, 100).to(x_t.device)
            objective_recon = self.denoise_fn(x_t,t)
            x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon=objective_recon)
            if clip_denoised:
                x0_recon.clamp_(-1., 1.)
            return x0_recon, x0_recon
        else:
            t = torch.full((x_t.shape[0],), self.steps[i], device=x_t.device, dtype=torch.long)
            n_t = torch.full((x_t.shape[0],), self.steps[i+1], device=x_t.device, dtype=torch.long)
            #
            # objective_recon = self.denoise_fn(x_t, timesteps=t, context=context)

            # objective_recon = self.denoise_fn(x_t, t)
            # latent_z = torch.randn(1, 100).to(x_t.device)
            objective_recon = self.denoise_fn(x_t,t)
            x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon=objective_recon)
            if clip_denoised:
                x0_recon.clamp_(-1., 1.)

            m_t = extract(self.m_t, t, x_t.shape)
            m_nt = extract(self.m_t, n_t, x_t.shape)
            var_t = extract(self.variance_t, t, x_t.shape)
            var_nt = extract(self.variance_t, n_t, x_t.shape)
            sigma2_t = (var_t - var_nt * (1. - m_t) ** 2 / (1. - m_nt) ** 2) * var_nt / var_t
            sigma_t = torch.sqrt(sigma2_t) * self.eta

            noise = torch.randn_like(x_t)
            x_tminus_mean = (1. - m_nt) * x0_recon + m_nt * y + torch.sqrt((var_nt - sigma2_t) / var_t) * \
                            (x_t - (1. - m_t) * x0_recon - m_t * y)

            return x_tminus_mean + sigma_t * noise, x0_recon

    @torch.no_grad()
    def p_sample_loop(self, y, context=None, clip_denoised=True, sample_mid_step=False):
        if self.condition_key == "nocond":
            context = None
        else:
            context = y if context is None else context

        if sample_mid_step:
            imgs, one_step_imgs = [y], []
            for i in tqdm(range(len(self.steps)), desc=f'sampling loop time step', total=len(self.steps)):
                img, x0_recon = self.p_sample(x_t=imgs[-1], y=y, context=context, i=i, clip_denoised=clip_denoised)
                imgs.append(img)
                one_step_imgs.append(x0_recon)
            return imgs, one_step_imgs
        else:
            img = y
            for i in tqdm(range(len(self.steps)), desc=f'sampling loop time step', total=len(self.steps)):
                img, _ = self.p_sample(x_t=img, y=y, context=context, i=i, clip_denoised=clip_denoised)
            return img

    @torch.no_grad()
    def sample(self, y, context=None, clip_denoised=True, sample_mid_step=False):
        return self.p_sample_loop(y, context, clip_denoised, sample_mid_step)