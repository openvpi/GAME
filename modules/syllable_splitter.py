from torch import nn

from lib.config.schema import ModelConfig
from lib.reflection import build_object_from_class_name


class SyllableSplitter(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.language_embedding = nn.Embedding(config.num_languages, config.in_channels, padding_idx=0)
        self.spectrogram_projection = nn.Linear(config.in_channels, config.in_channels)
        self.backbone = build_object_from_class_name(
            config.backbone_class, nn.Module,
            config.in_channels,
            **config.backbone_kwargs
        )

    def forward(self, spectrogram, language, mask=None):
        x = self.language_embedding(language.unsqueeze(-1)) + self.spectrogram_projection(spectrogram)
        features, boundaries = self.backbone(x, mask=mask)
        return features, boundaries
