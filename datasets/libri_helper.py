import torch
from torchaudio.transforms import MelSpectrogram, Resample, TimeMasking, FrequencyMasking
from torchaudio.functional import resample
from typing import Optional



class AudioProcessor:
    def __init__(
        self,
        resample_rate=16000,
        win_time=0.025,
        stride_time=0.01,
        n_fft=2048,
        mel_filter_size=80,
        freq_mask_length=27,
        time_mask_prop=0.05,
        time_mask_length=10,
        sample_rate: int = 16000,
        should_spec_aug: bool = True,
        noise_scale: float = 1e-4
    ):
        self.noise_scale = noise_scale
        self.win_length = int(resample_rate * win_time)
        self.hop_length = int(resample_rate * stride_time)
        self.resampler: Optional[Resample] = None
        if resample_rate != sample_rate:
            self.resampler = Resample(sample_rate, resample_rate)

        self.mel_sampler = MelSpectrogram(
            sample_rate=resample_rate,
            win_length=int(resample_rate * win_time),
            hop_length=int(resample_rate * stride_time),
            n_fft=n_fft,
            n_mels=mel_filter_size
        )
        self.resample_rate = resample_rate
        self.should_spec_aug = should_spec_aug
        self.freq_masking = FrequencyMasking(freq_mask_length)
        self.time_masking = TimeMasking(time_mask_length, iid_masks=True, p=time_mask_prop)

    def __call__(self, inputs: torch.Tensor, sample_rate=16000) -> torch.Tensor:
        """
        Args:
            inputs (torch.Tensor): with shape `(T)` or `(B, T)`
            sample_rate (int): input sampling rate.

        Returns:
            torch.Tensor with shape `(T, D)` or `(B, T, D)`

        """
        if sample_rate != self.resample_rate:
            inputs = resample(inputs, sample_rate, self.resample_rate)
        elif self.resampler is not None:
            inputs = self.resampler(inputs)

        # Add noise for log scaling
        noise = torch.randn(inputs.size(), device=inputs.device) * self.noise_scale
        inputs += noise

        mel_feature = self.mel_sampler(inputs)
        log_mel_feature = mel_feature.log2()

        if self.should_spec_aug:
            log_mel_feature = self.freq_masking(log_mel_feature)
            log_mel_feature = self.time_masking(log_mel_feature)

        if log_mel_feature.dim() == 2:
            log_mel_feature = log_mel_feature.transpose(0, 1)
        else:
            log_mel_feature = log_mel_feature.transpose(1, 2)

        return log_mel_feature
