import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from torch import nn
import torch.nn.functional as F

from refactor.utils.misc import extract, extract_data_from_batch, mean_flat
from refactor.utils.schedules import cosine_beta_schedule, linear_beta_schedule, quadratic_beta_schedule, sigmoid_beta_schedule
from utils.ema import EMA



class DiffusionModel(pl.LightningModule):
    def __init__(
        self,
        unet: nn.Module,
        image_size: int,
        timesteps: int,
        noise_schedule: str,
        use_fp16: bool,
        is_conditional: bool,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
        criterion: nn.Module,
        use_ema: bool = True,
        ema_decay: float = 0.9999,
        lr_warmup=0,
        use_p2_weighting: bool = False,
        p2_gamma: float = 0.5,
        p2_k: float = 1,
        p_uncond: float=0.1,
        snr: float=1.,
        beta_start: float=0.0001,
        beta_end: float=0.9999
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["criterion"])

        # create Unet
        # attempt using hydra.utils.instantiate to instantiate both unet, lr scheduler and optimizer
        self.model = unet
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.timesteps = timesteps
        self.noise_schedule = noise_schedule
        # training parameters
        self.use_ema = use_ema
        if self.use_ema:
            self.eps_model_ema = EMA(self.model, beta=ema_decay)
        self.use_fp16 = use_fp16
        self.is_conditional = is_conditional
        self.image_size = image_size
        self.optimizer = optimizer
        self.lr_warmup = lr_warmup
        self.criterion = criterion


        self.use_p2_weighting = use_p2_weighting
        self.p2_gamma = p2_gamma
        self.p2_k = p2_k

        self.p_uncond=p_uncond
        self.snr = snr

        self.beta_start = beta_start
        self.beta_end = beta_end

        # setup 
        self.set_scheduler(self.noise_schedule)
        self.set_noise_schedule(self.timesteps)


    def set_scheduler(self, noise_schedule):
        if noise_schedule == "linear":
            self.noise_scheduler = linear_beta_schedule
        elif noise_schedule == "cosine":
            self.noise_scheduler = cosine_beta_schedule
        elif noise_schedule == 'quadratic':
            self.noise_scheduler = quadratic_beta_schedule
        elif noise_schedule == 'sigmoid':
            self.noise_scheduler = sigmoid_beta_schedule
        else:
            raise ValueError(f"invalid noise schedule {noise_schedule}")

    def set_noise_schedule(self, timesteps):
        # define beta schedule
        self.betas = self.noise_scheduler(timesteps=timesteps, beta_end=0.05)

        # define alphas
        alphas = 1.0 - self.betas

        self.alphas_cumprod = torch.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)

        # sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        """
        Forward pass with noise.
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p2_weighting(self, x_t, ts, target, prediction):
        """
        From Perception Prioritized Training of Diffusion Models: https://arxiv.org/abs/2204.00227.
        """
        weight = (1 / (self.p2_k + self.snr) ** self.p2_gamma, ts, x_t.shape)
        loss_batch = mean_flat(weight * (target - prediction) ** 2)
        loss = torch.mean(loss_batch)
        return loss


    def training_step(self, batch: torch.Tensor, batch_idx: int):
        x_start, condition = extract_data_from_batch(batch)

        if noise is None:
            noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=self.timesteps, noise=noise)

        # calculating generic loss function, we'll add it to the class constructor once we have the code
        # we should log more metrics at train and validation e.g. l1, l2 and other suggestions
        if self.use_fp16:
            with torch.cuda.amp.autocast():
                if self.is_conditional:
                    predicted_noise = self.model(x_noisy, self.timesteps, condition)
                else:
                    predicted_noise = self.model(x_noisy, self.timesteps)
        else:
            if self.is_conditional:
                predicted_noise = self.model(x_noisy, self.timesteps, condition)
            else:
                predicted_noise = self.model(x_noisy, self.timesteps)

        if self.use_p2_weighting: 
            loss = self.p2_weighting(
                x_t=x_noisy,
                ts=self.timesteps,
                target=x_start,
                prediction=predicted_noise
            )
        else: 
            loss = self.criterion(predicted_noise, noise)

        self.log("train", loss, batch_size=batch.shape[0])

        return loss

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

    def sample(
        self, n_sample: int, condition=None, timesteps=None, *args, **kwargs  # number of samples
    ) -> torch.Tensor:
        return

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        if self.trainer.global_step < self.lr_warmup:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.lr_warmup)
            for pg in optimizer.param_groups:
                pg["learning_rate"] = lr_scale * self.optimizer_config.params.lr

        optimizer.step(closure=optimizer_closure)

    def on_before_zero_grad(self, *args, **kwargs) -> None:
        if self.use_ema:
            self.eps_model_ema.update(self.model)

    def configure_optimizers(self):
        #    optimizer = instantiate(
        #        self.optimizer)
        #   if self.lr_scheduler is not None:
        #        scheduler = instantiate(
        #            self.lr_scheduler, optimizer=optimizer)
        #        return {"optimizer": optimizer, "lr_scheduler": scheduler}
        return {"optimizer": self.optimizer, "lr_scheduler": self.lr_scheduler}
