from torch import nn

from lib.config.schema import ModelConfig
from lib.reflection import build_object_from_class_name
from modules.commons.common_layers import CyclicRegionEmbedding, LocalDownsample


class GeneratorSegmentationModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.spectrogram_projection = nn.Linear(config.in_channels, config.embedding_dim)
        self.region_embedding = CyclicRegionEmbedding(
            config.embedding_dim,
            cycle_length=config.region_cycle_len
        )
        self.use_language_embedding = config.use_languages
        if self.use_language_embedding:
            self.language_embedding = nn.Embedding(config.num_languages + 1, config.embedding_dim, padding_idx=0)
        self.segmenter = build_object_from_class_name(
            config.segmenter.cls, nn.Module,
            config.embedding_dim, 1, True,
            **config.segmenter.kwargs
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(1, config.embedding_dim * 4),
            nn.GELU(),
            nn.Linear(config.embedding_dim * 4, config.embedding_dim)
        )

    def forward(self, spectrogram, regions, times, language=None, mask=None):
        x = self.spectrogram_projection(spectrogram) + self.region_embedding(regions)
        if self.use_language_embedding:
            x = x + self.language_embedding(language.unsqueeze(-1))
        time_emb=self.time_embedding(times.unsqueeze(-1))
        x = x + time_emb
        x, latent = self.segmenter(x,times, mask=mask)
        velocities = x.squeeze(-1).tanh()
        return velocities, latent


class SegmentationModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.spectrogram_projection = nn.Linear(config.in_channels, config.embedding_dim)
        self.region_embedding = CyclicRegionEmbedding(
            config.embedding_dim,
            cycle_length=config.region_cycle_len
        )
        self.mode = config.mode
        if self.mode == "d3pm":
            self.time_embedding = nn.Sequential(
                nn.Linear(1, config.embedding_dim * 4),
                nn.GELU(),
                nn.Linear(config.embedding_dim * 4, config.embedding_dim)
            )
        self.use_language_embedding = config.use_languages
        if self.use_language_embedding:
            self.language_embedding = nn.Embedding(config.num_languages + 1, config.embedding_dim, padding_idx=0)
        self.segmenter = build_object_from_class_name(
            config.segmenter.cls, nn.Module,
            config.embedding_dim, 1, True,
            **config.segmenter.kwargs
        )

    def forward(self, spectrogram, regions, t=None, language=None, mask=None):
        x = self.spectrogram_projection(spectrogram) + self.region_embedding(regions)
        if self.mode == "d3pm":
            x = x + self.time_embedding(t[..., None, None])
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
        self.use_glu = config.use_glu
        if self.use_glu:
            adaptor_out_dim = config.embedding_dim * 2
        else:
            adaptor_out_dim = config.embedding_dim
        self.adaptor = build_object_from_class_name(
            config.adaptor.cls, nn.Module,
            config.embedding_dim, adaptor_out_dim, True,
            **config.adaptor.kwargs
        )
        if self.use_glu:
            self.glu = nn.GLU(dim=-1)
        self.downsample = LocalDownsample()
        self.estimator = build_object_from_class_name(
            config.estimator.cls, nn.Module,
            config.embedding_dim, config.estimator_out_channels, False,
            **config.estimator.kwargs
        )

    def forward(self, spectrogram, regions, max_n: int, t_mask=None, n_mask=None):
        x = self.spectrogram_projection(spectrogram) + self.region_embedding(regions)
        x, latent = self.adaptor(x, mask=t_mask)
        if self.use_glu:
            x = self.glu(x)
        x_down = self.downsample(x, regions, max_n=max_n)
        estimations = self.estimator(x_down, mask=n_mask)
        return estimations, latent
