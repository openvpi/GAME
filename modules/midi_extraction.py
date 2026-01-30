from torch import nn

from lib.config.schema import ModelConfig
from lib.reflection import build_object_from_class_name
from modules.commons.common_layers import CyclicRegionEmbedding, LocalDownsample


class SegmentationModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.spectrogram_projection = nn.Linear(config.in_channels, config.embedding_dim)
        self.region_embedding = CyclicRegionEmbedding(
            config.embedding_dim,
            cycle_length=config.region_cycle_len
        )
        self.use_language_embedding = config.use_languages
        if config.use_languages:
            self.language_embedding = nn.Embedding(config.num_languages + 1, config.embedding_dim, padding_idx=0)
        self.segmenter = build_object_from_class_name(
            config.segmenter.cls, nn.Module,
            config.embedding_dim, 1, True,
            **config.segmenter.kwargs
        )

    def forward(self, spectrogram, regions, language=None, mask=None):
        x = self.spectrogram_projection(spectrogram) + self.region_embedding(regions)
        if self.use_language_embedding:
            x = x + self.language_embedding(language.unsqueeze(-1))
        x, latent = self.segmenter(x, mask=mask)
        velocities = x.squeeze(-1).tanh()
        return velocities, latent


class EstimationModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.spectrogram_projection = nn.Linear(config.in_channels, config.embedding_dim)
        self.region_embedding = CyclicRegionEmbedding(
            config.embedding_dim,
            cycle_length=config.region_cycle_len
        )
        self.adaptor = build_object_from_class_name(
            config.adaptor.cls, nn.Module,
            config.embedding_dim, config.embedding_dim, True,
            **config.adaptor.kwargs
        )
        self.downsample = LocalDownsample()
        self.estimator = build_object_from_class_name(
            config.estimator.cls, nn.Module,
            config.embedding_dim, config.midi_num_bins, False,
            **config.estimator.kwargs
        )

    def forward(self, spectrogram, regions, max_n: int, t_mask=None, n_mask=None, sigmoid=True):
        x = self.spectrogram_projection(spectrogram) + self.region_embedding(regions)
        x, latent = self.adaptor(x, mask=t_mask)
        x_down = self.downsample(x, regions, max_n=max_n)
        logits = self.estimator(x_down, mask=n_mask)
        if sigmoid:
            estimations = logits.sigmoid()
            return estimations, latent
        else:
            return logits, latent
