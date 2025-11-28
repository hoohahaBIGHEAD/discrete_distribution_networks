# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/
import boxx

"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import torch
from torch_utils import persistence

# ----------------------------------------------------------------------------
# Loss function corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".


@persistence.persistent_class
class VPLoss:
    def __init__(self, beta_d=19.9, beta_min=0.1, epsilon_t=1e-5):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma**2
        y, augment_labels = (
            augment_pipe(images) if augment_pipe is not None else (images, None)
        )
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss, {}

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t**2) + self.beta_min * t).exp() - 1).sqrt()


# ----------------------------------------------------------------------------
# Loss function corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".


@persistence.persistent_class
class VELoss:
    def __init__(self, sigma_min=0.02, sigma_max=100):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
        weight = 1 / sigma**2
        y, augment_labels = (
            augment_pipe(images) if augment_pipe is not None else (images, None)
        )
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss, {}


# ----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).


@persistence.persistent_class
class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = (
            augment_pipe(images) if augment_pipe is not None else (images, None)
        )
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        boxx.cf.debug and boxx.g()
        return loss, {}

    """
# analysis about sigma 
tree-sigma
└── /: (batch, 1, 1, 1) of torch.cuda.FloatTensor @ cuda:0

rnd_normal = torch.randn([10000, 1, 1, 1], device=images.device)
sigma = (rnd_normal * self.P_std + self.P_mean).exp()

loga-(rnd_normal * self.P_std + self.P_mean)
shape:(10000, 1, 1, 1) type:(float32 of torch.Tensor) max: 3.1941, min: -5.652, mean: -1.1937

loga-(rnd_normal * self.P_std + self.P_mean).exp()
shape:(10000, 1, 1, 1) type:(float32 of torch.Tensor) max: 24.389, min: 0.0035106, mean: 0.62631
plot(sigma,1)
    """


# ----------------------------------------------------------------------------


@persistence.persistent_class
class DDNLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

        import sddn

        self.diverge_shaping_manager = sddn.DivergeShapingManager()
        self.diverge_shaping = boxx.cf.get("kwargs", {}).get("diverge_shaping", 0)

    def __call__(self, net, images, labels=None, augment_pipe=None):
        y, augment_labels = (
            augment_pipe(images) if augment_pipe is not None else (images, None)
        )
        di = dict(target=y, class_labels=labels, augment_labels=augment_labels)
        with self.diverge_shaping_manager(di, self.diverge_shaping):
            d = net(di)
        self.diverge_shaping_manager.set_total_output_level(d)

        if "pixel.weight.loss" and 0:  # TODO 也许没用
            pixeln_per_ddo = [
                predict.shape[-1] * predict.shape[-2] for predict in d["predicts"]
            ]
            loss = sum(
                [loss * pixeln for loss, pixeln in zip(d["losses"], pixeln_per_ddo)]
            ) / sum(pixeln_per_ddo)
        else:
            loss = sum(d["losses"]) / len(d["losses"])  # mean Hierarchical
        loss = loss * y.numel()  # EDM code is using .sum() instead of .mean()
        boxx.cf.debug and boxx.g()
        return loss, d