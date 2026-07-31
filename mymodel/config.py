"""
Central configuration for the frozen, sensor-agnostic motion prior.

Every locked design decision lives here so the rest of the code reads it rather
than hardcoding. Values chosen for ETH/UCY (~2.5 Hz pedestrian data) with a
~5 s window. Nothing here is sensor-specific: the model trains on self-injected
Gaussian corruption and conditions on the noise level sigma, never on sensor id.
"""
from dataclasses import dataclass, field


@dataclass
class DataConfig:
    # ETH/UCY is ~2.5 Hz. 12 frames ~= 4.8 s window.
    fps: float = 2.5
    window_len: int = 12          # frames per training/inference window
    # Representation the prior operates on: POSITIONS ONLY, expressed relative
    # to the first frame of the window (pos - pos[0]).
    #
    # Derivative channels are OFF by default. Differencing amplifies noise:
    # velocity from a noisy position is noisier than the position, and
    # acceleration (second difference) noisier still, so at high sigma those
    # channels carry mostly noise. Attention over the window can learn finite
    # differences itself (they are linear ops). Flip these to True to ablate.
    n_pos_dims: int = 2
    use_velocity: bool = False
    use_acceleration: bool = False

    # Normalization: trajectories are converted to origin-relative offsets and
    # scaled to metres. Absolute scene coordinates never enter the prior (that
    # is what allows outdoor->indoor transfer).
    normalize_to_offsets: bool = True

    @property
    def n_channels(self) -> int:
        """Input channels: position (+velocity)(+acceleration)."""
        c = self.n_pos_dims
        if self.use_velocity:
            c += self.n_pos_dims
        if self.use_acceleration:
            c += self.n_pos_dims
        return c

    @property
    def n_out_dims(self) -> int:
        """Output channels: ONLY denoised position.

        Velocity/acceleration are inputs (so attention sees stops/turns) but not
        outputs, because they are deterministic functions of position. Predicting
        them as free channels would let the model output a (pos,vel,acc) triple
        that is not self-consistent. We denoise position; derivatives are context.
        """
        return self.n_pos_dims


@dataclass
class EDMConfig:
    """EDM (Karras et al. 2022) noise/preconditioning hyperparameters.

    sigma is the noise standard deviation in the SAME units as the position
    channels (metres). The training sigma range must bracket the real sensor
    noise you expect at inference on NILoc.
    """
    sigma_data: float = 0.30      # std of the clean signal (offsets, metres). Tune to data.
    sigma_min: float = 0.02       # smallest noise level trained/sampled
    sigma_max: float = 2.0        # largest noise level trained/sampled
    # Log-normal sampling of training sigma (EDM default P_mean/P_std).
    p_mean: float = -1.2
    p_std: float = 1.2
    rho: float = 7.0              # sampling schedule curvature (for the ODE step grid)


@dataclass
class ModelConfig:
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 256
    dropout: float = 0.1
    # Bidirectional attention within the window (denoising, not prediction).
    # Causal emission is handled by the delayed-freeze inference loop, not here.
    causal_attention: bool = False


@dataclass
class TrainConfig:
    batch_size: int = 256
    lr: float = 2e-4
    weight_decay: float = 0.0
    betas: tuple = (0.9, 0.999)
    ema_decay: float = 0.999      # EMA of weights; sampled model uses EMA copy
    max_steps: int = 100_000
    warmup_steps: int = 1_000
    grad_clip: float = 1.0
    seed: int = 0
    device: str = "cuda"          # falls back to cpu if unavailable
    log_every: int = 100
    ckpt_every: int = 5_000


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    edm: EDMConfig = field(default_factory=EDMConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
