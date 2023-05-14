from functools import partial
import random

from einops import rearrange, repeat
import numpy as np

import torch
import torch.nn.functional as F
import tqdm
from refactor.models.diffusion.diffusion import DDPM
from torch import nn
from refactor.utils.misc import extract, extract_data_from_batch, mean_flat
from refactor.utils.schedules import (
    alpha_cosine_log_snr,
    beta_linear_log_snr,
    linear_beta_schedule,
)

def interpolate_embeddings(logits, nucleotide_embeddings_tensor): 
    softmax = torch.softmax(logits, dim=-2)
    return torch.einsum("bwij, id -> bwjd", softmax, nucleotide_embeddings_tensor)


def predict_logits(model, x_emb_norm, x_noisy, prev_embeds, t, classes):
    model_input = torch.cat([x_emb_norm, x_noisy, prev_embeds], dim=-2)
    logits = model(model_input, t, classes)
    return logits


def cdcd_forward(model, x_emb_norm, x_noisy, prev_embeds, t, classes, use_clamp: bool = False): 
    logits = predict_logits(model, x_emb_norm, x_noisy, prev_embeds, t, classes)
    nucleotide_emb_norm = F.normalize(model.nucleotide_embeddings.weight, p=2, dim=-1)  # normalize across emb_dim
    emb = interpolate_embeddings(logits, nucleotide_emb_norm)
    if use_clamp: 
        emb = torch.clamp(emb, -1., 1.)
    return logits, emb



class CDCDModel(DDPM):
    def __init__(
        self,
        *,
        image_size,
        timesteps=50,
        noise_schedule="cosine",
        time_difference=0.0,
        cdcd_transformer: nn.Module,
        is_conditional: bool,
        p_uncond: float = 0.1,
        use_fp16: bool,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
        criterion: nn.Module,
        use_ema: bool = True,
        ema_decay: float = 0.9999,
        lr_warmup=0,
        use_p2_weigthing: bool = False,
        p2_gamma: float = 0.5,
        p2_k: float = 1,
        use_clamp: bool = False,
        use_time_warping: bool = True,
        n_steps_threshold_time_warping: int = 10,
        last_n_timesteps_time_warping: int = 10,
        mse_coef: float=0.
    ):
        super().__init__(
            cdcd_transformer,
            image_size,
            optimizer,
            lr_scheduler,
            criterion,
            is_conditional,
            use_fp16,
            timesteps,
            noise_schedule,
            use_ema,
            ema_decay,
            lr_warmup,
            use_p2_weigthing,
            p2_gamma, 
            p2_k, 
            p_uncond
        )

        self.use_clamp = use_clamp
        self.use_time_warping = use_time_warping
        self.n_steps_time_warping = n_steps_threshold_time_warping
        self.last_n_timesteps_time_warping = last_n_timesteps_time_warping
        self.mse_coef = mse_coef

        # for self conditioning
        self.prev_embeds = None

        # for time warping
        self.epoch_losses = []
        self.batch_t = []

        
    def predict_embeddings(self, x_noisy, prev_embeds, t, classes):
        _, emb = cdcd_forward(self.model, x_noisy, x_noisy, prev_embeds, t, classes, use_clamp=self.use_clamp)
        return emb


    def score_f(self, image, time, classes=None): 
        if self.prev_embeds is None: 
            self.prev_embeds = torch.zeros_like(image, device=image.device)
            if classes is not None: 
                self.prev_embeds = self.prev_embeds.repeat(2, 1, 1, 1)

        # score interpolation function
        self.prev_embeds = self.predict_embeddings(image, self.prev_embeds, time, classes)
        time = time[:, None, None, None]
        return (self.prev_embeds - image) * (time ** 2)


    @torch.no_grad()
    def p_sample_loop(
        self,
        classes,
        shape,
        cond_weight=0
    ):        
        # sample from score interpolation CDCD model
        device = next(self.model.parameters()).device
        model = self.model
        timesteps = self.timesteps

        b = shape[0]
        image = torch.randn(shape, device=device) 
        images = [] # accumulate all steps
        
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
            reversed(range(0, timesteps)),
            desc="sampling loop time step",
            total=timesteps,
        ):
            
            time_next = torch.full((b,), i, device=device, dtype=torch.long)
            image = sampling_fn(self.score_f, image, t=time_next, t_index=i)

            # save intermediate distributions
            out = model.linear_out(image)
            out = rearrange(out, "b w s nucl -> b w nucl s")
            out = torch.softmax(out, dim=-2)
            images.append(out.cpu().numpy())
        
        return images


    def cdcd_loss(
        self,
        x_start, 
        t, 
        classes, 
        noise=None, 
        p_uncond=0.1, 
        use_reparameterization_trick=True, 
        noise_normalized_emb=True, 
        normalize_predicted_embs=False,
        mse_coef: float = 0.
    ):
        """
        Calculate the loss conditioned and noise injected.
        """
        model = self.model

        x_idx = torch.argmax(x_start, dim=2)  # onehot -> index
        x_emb = model.nucleotide_embeddings(x_idx)  # x_emb.shape = b,c,d,s
        x_emb_normalized = F.normalize(x_emb, p=2, dim=-2)  # L2 normalize across emb_dim

        
        device = x_start.device
        if noise is None:
            if use_reparameterization_trick:
                mean = torch.mean(x_emb_normalized, dim=0, keepdim=True)
                std = torch.mean(x_emb_normalized, dim=0, keepdim=True)
                epsilon = torch.randn_like(mean)
                noise = mean + std * epsilon

            else:
                noise = torch.randn_like(x_emb_normalized) #  gauss noise 

        if noise_normalized_emb: 
            x_noisy_emb = self.q_sample(x_start=x_emb_normalized, t=t, noise=noise) #this is the auto generated noise given t and Noise
        else: 
            x_noisy_emb = self.q_sample(x_start=x_emb, t=t, noise=noise)

        context_mask = torch.bernoulli(torch.zeros(classes.shape[0]) + (1-p_uncond)).to(device)

        # mask for unconditional guidance
        classes = classes * context_mask
        classes = classes.type(torch.long)
        
        fwd = partial(
            cdcd_forward, 
            model=model, 
            x_emb_normalized=x_emb_normalized,
            x_noisy_emb=x_noisy_emb,
            t=t,
            classes=classes
        )

        # self conditioning 
        prev_embeds = torch.zeros_like(x_emb_normalized)
        if random.random() > 0.5: 
            with torch.no_grad():
                prev_embeds = fwd(prev_embeds=prev_embeds)[1].detach()

        predicted_logits, predicted_embs = fwd(prev_embeds=prev_embeds)

        if normalize_predicted_embs: 
            predicted_embs = F.normalize(predicted_embs, dim=-2)

        mse_loss = F.mse_loss(predicted_embs, x_emb_normalized, reduction='mean')
        predicted_logits = rearrange(predicted_logits, 'b c nucl s -> b nucl c s')
        ce_loss = self.criterion(predicted_logits, x_idx)

        return dict(
            loss=ce_loss + mse_loss * mse_coef, 
            ce_loss=ce_loss,
            mse_loss=mse_loss
        )
    
    def sample_timestep(self, batch_size, device, step):
        t = torch.randint(0, self.timesteps, (batch_size,), device=device).long() # sampling a t to generate t and t+1

        if self.use_time_warping and step >= self.n_steps_time_warping:
            # sort the epoch losses so that one can take the t-s with the biggest losses
            sort_val = np.argsort(self.epoch_losses)
            sorted_t=[self.batch_t[i] for i in sort_val]

            # take the t-s for the 5 biggest losses (5 was taken as example, no extensive optimization)
            last_n_t = sorted_t[-self.last_n_timesteps_time_warping:]
            unnested_last_n_t = [item for sublist in last_n_t for item in sublist]
            rand_choice = np.random.choice(unnested_last_n_t, size=x.shape[0])

            # take x.shape[0] number of t-s for the 5 biggest losses
            t_not_random = torch.tensor(rand_choice, device="cpu")
            # pick between t generated above and t_not_random (to increase exploration, and not to get stuck
            # in the same t-s)
            t = np.random.choice([t.cpu().detach(), t_not_random.cpu().detach()])
            t = t.to(device)

        return t
        

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        x, y = batch

        batch_size = batch.shape[0]
        device = batch.device
        t = self.sample_timestep(batch_size, device, batch_idx)

        losses = self.cdcd_loss(x, t, y, p_uncond=self.p_uncond, mse_coef=self.mse_coef)
        self.log_dict(losses, batch_size=batch_size)

        if self.use_time_warping:
            # accumulate for time warping
            loss = losses['loss']
            self.epoch_losses.append(loss.item())
            self.batch_t.append(list(t.cpu().detach().numpy()))

        return losses

    def on_train_epoch_end(self):
        self.epoch_losses = []
        self.batch_t = []

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        return self.inference_step(batch, batch_idx, "validation")

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        return self.inference_step(batch, batch_idx, "test")

    def inference_step(self, batch: torch.Tensor, batch_idx: int, phase="validation", noise=None):
        x_start, condition = extract_data_from_batch(batch)
        device = x_start.device
        batch_size = batch.shape[0]

        t = torch.randint(0, self.timesteps, (batch_size,), device=device).long()  # sampling a t to generate t and t+1

        if noise is None:
            noise = torch.randn_like(x_start)  #  gauss noise
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)  # this is the auto generated noise given t and Noise

        context_mask = torch.bernoulli(torch.zeros(classes.shape[0]) + (1 - self.p_uncond)).to(device)

        # mask for unconditinal guidance
        classes = classes * context_mask
        classes = classes.type(torch.long)

        predictions = self.model(x_noisy, t, condition)

        loss = self.criterion(predictions, batch)

        self.log("validation_loss", loss) if phase == "validation" else self.log("test_loss", loss)

        """
            Log multiple losses at validation/test time according to internal discussions.
        """

        return predictions
