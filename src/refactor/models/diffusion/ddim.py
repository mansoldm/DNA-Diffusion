from functools import partial

import torch
import torch.nn.functional as F
import tqdm
from models.diffusion.diffusion import DiffusionModel
from torch import nn
from refactor.models.networks.unet_lucas import UNetLucas
from utils.misc import extract, extract_data_from_batch, mean_flat
from utils.schedules import (
    alpha_cosine_log_snr,
    beta_linear_log_snr,
    linear_beta_schedule,
)


class DDIM(DiffusionModel):
    def __init__(
        self,
        *,
        unet: nn.Module,
        image_size,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
        criterion: nn.Module,
        is_conditional: bool=True,
        use_fp16: bool=False,
        timesteps=50,
        noise_schedule="cosine",
        use_ema: bool = True,
        ema_decay: float = 0.9999,
        lr_warmup=0,
        use_p2_weighting: bool = False,
        p2_gamma: float = 0.5,
        p2_k: float = 1,
        p_uncond: float = 0.1,
        snr: float=1.
    ):
        super().__init__(
            unet,
            image_size,
            timesteps,
            noise_schedule,
            use_fp16,
            is_conditional,
            optimizer,
            lr_scheduler,
            criterion,
            use_ema,
            ema_decay,
            lr_warmup,
            use_p2_weighting=use_p2_weighting,
            p2_gamma=p2_gamma,
            p2_k=p2_k,
            p_uncond=p_uncond,
            snr=snr
        )

    @torch.no_grad()
    def p_ddim_sample(self, model, x, t, t_index, eta=0, temp=1.0):
        alpha_t = extract(self.alphas_cumprod, t, x.shape)
        alpha_prev_t = extract(self.alphas_cumprod_prev, t, x.shape)
        sigma = eta * ((1 - alpha_prev_t) / (1 - alpha_t) * (1 - alpha_t / alpha_prev_t)) ** 0.5
        sqrt_one_minus_alphas_cumprod = extract(sqrt_one_minus_alphas_cumprod, t, x.shape)
        pred_x0 = (x - self.sqrt_one_minus_alphas_cumprod * model(x, time=t)) / (alpha_t**0.5)
        dir_xt = (1.0 - alpha_prev_t - sigma**2).sqrt() * model(x, time=t)
        if sigma == 0.0:
            noise = 0.0
        else:
            noise = torch.randn((1, x.shape[1:]))
        noise *= temp

        x_prev = (alpha_prev_t**0.5) * pred_x0 + dir_xt + sigma * noise

        return x_prev


    # Algorithm 2 but save all images:
    @torch.no_grad()
    def p_sample_loop(self, classes, shape, cond_weight):
        device = next(self.model.parameters()).device

        b = shape[0]
        # start from pure noise (for each example in the batch)
        image = torch.randn(shape, device=device)
        images = []

        if classes is not None:
            n_sample = classes.shape[0]
            context_mask = torch.ones_like(classes).to(device)
            # make 0 index unconditional
            # double the batch
            classes = classes.repeat(2)
            context_mask = context_mask.repeat(2)
            context_mask[n_sample:] = 0.0  # makes second half of batch context free
            sampling_fn = partial(
                self.p_sample_guided,
                classes=classes,
                cond_weight=cond_weight,
                context_mask=context_mask,
            )
        else:
            sampling_fn = partial(self.p_sample)

        for i in tqdm(
            reversed(range(0, self.timesteps)),
            desc="sampling loop time step",
            total=self.timesteps,
        ):
            image = sampling_fn(
                self.model,
                x=image,
                t=torch.full((b,), i, device=device, dtype=torch.long),
                t_index=i,
            )
            images.append(image.cpu().numpy())
        return images

    @torch.no_grad()
    def sample(self, image_size, classes=None, batch_size=16, channels=3, cond_weight=0):
        return self.p_sample_loop(
            self.model,
            classes=classes,
            shape=(batch_size, channels, 4, image_size),
            cond_weight=cond_weight,
        )
