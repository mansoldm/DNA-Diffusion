from functools import partial

import torch
import torch.nn.functional as F
import tqdm
from refactor.models.diffusion.diffusion import DiffusionModel
from torch import nn
from utils.misc import extract, extract_data_from_batch, mean_flat
from utils.schedules import (
    alpha_cosine_log_snr,
    beta_linear_log_snr,
    linear_beta_schedule,
)


class DDPM(DiffusionModel):
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
        noise_schedule="linear",
        use_ema: bool = True,
        ema_decay: float = 0.9999,
        lr_warmup=0,
        use_p2_weighting: bool = False,
        p2_gamma: float = 0.5,
        p2_k: float = 1,
        p_uncond: float = 0.1,
        snr: float=1.,
        beta_start=0.0001,
        beta_end=0.9999
        
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
            snr=snr,
            beta_start=beta_start,
            beta_end=beta_end
        )


    @torch.no_grad()
    def p_sample(self, x, t, t_index):
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        # print (x.shape, 'x_shape')
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (x - betas_t * self.score_f(x, time=t) / sqrt_one_minus_alphas_cumprod_t)

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            return model_mean + torch.sqrt(posterior_variance_t) * noise
        
    def score_f(self, *args, **kwargs): 
        return self.model(*args, **kwargs)

    @torch.no_grad()
    def p_sample_guided(self, x, classes, t, t_index, context_mask, cond_weight=0.0):
        # adapted from: https://openreview.net/pdf?id=qw8AKxfYbI
        # print (classes[0])
        batch_size = x.shape[0]
        # double to do guidance with
        t_double = t.repeat(2)
        x_double = x.repeat(2, 1, 1, 1)
        betas_t = extract(self.betas, t_double, x_double.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t_double, x_double.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t_double, x_double.shape)

        # classifier free sampling interpolates between guided and non guided using `cond_weight`
        classes_masked = classes * context_mask
        classes_masked = classes_masked.type(torch.long)
        # print ('class masked', classes_masked)
        preds = self.score_f(x_double, time=t_double, classes=classes_masked)
        eps1 = (1 + cond_weight) * preds[:batch_size]
        eps2 = cond_weight * preds[batch_size:]
        x_t = eps1 - eps2

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t[:batch_size] * (
            x - betas_t[:batch_size] * x_t / sqrt_one_minus_alphas_cumprod_t[:batch_size]
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            return model_mean + torch.sqrt(posterior_variance_t) * noise

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
