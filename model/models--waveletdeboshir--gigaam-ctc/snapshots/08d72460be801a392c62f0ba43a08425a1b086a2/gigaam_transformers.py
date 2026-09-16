from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from .encoder import ConformerEncoder
from torch import Tensor
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2Processor
from transformers.configuration_utils import PretrainedConfig
from transformers.feature_extraction_sequence_utils import \
    SequenceFeatureExtractor
from transformers.feature_extraction_utils import BatchFeature
from transformers.modeling_outputs import CausalLMOutput
from transformers.modeling_utils import PreTrainedModel


class GigaAMCTC(nn.Module):
    """
    GigaAM-CTC model
    """

    def __init__(self, config_encoder, config_head):
        super().__init__()
        self.encoder = ConformerEncoder(**config_encoder)
        self.head = CTCHead(**config_head)

    def forward(self, input_features: Tensor, input_lengths: Tensor) -> Tensor:
        encoded, encoded_lengths = self.encoder(input_features, input_lengths)
        logits = self.head(encoded)
        return logits, encoded_lengths


class CTCHead(nn.Module):
    """
    CTC Head module for Connectionist Temporal Classification.
    """

    def __init__(self, feat_in: int, num_classes: int):
        super().__init__()
        self.decoder_layers = nn.Sequential(
            nn.Conv1d(feat_in, num_classes, kernel_size=1)
        )

    def forward(self, encoder_output: Tensor) -> Tensor:
        # B x C x T
        return self.decoder_layers(encoder_output)


class GigaAMFeatureExtractor(SequenceFeatureExtractor):
    """
    Feature extractor for GigaAM.
    """
    model_input_names = ["input_features"]

    def __init__(
        self,
        feature_size=64,
        sampling_rate=16000,
        padding_value=0.0,
        chunk_length=30.0,
        **kwargs,
    ):
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            chunk_length=chunk_length,
            **kwargs,
        )
        self.hop_length = sampling_rate // 100
        self.n_samples = chunk_length * sampling_rate
        self.featurizer = torchaudio.transforms.MelSpectrogram(
                sample_rate=sampling_rate,
                n_fft=sampling_rate // 40,
                win_length=sampling_rate // 40,
                hop_length=self.hop_length,
                n_mels=feature_size,
            )

    def to_dict(self) -> Dict[str, Union[str, int, Dict]]:
        dictionary = super().to_dict()

        if "featurizer" in dictionary:
            del dictionary["featurizer"]
        dictionary["hop_length"] = self.hop_length
        dictionary["n_samples"] = self.n_samples
        return dictionary

    def out_len(self, input_lengths: Tensor) -> Tensor:
        """
        Calculates the output length after the feature extraction process.
        """
        return input_lengths.div(self.hop_length, rounding_mode="floor").add(1).long()

    def __call__(
        self,
        raw_speech: Union[np.ndarray, List[float], List[np.ndarray], List[List[float]]],
        sampling_rate: Optional[int] = None,
        padding: str = "max_length",
        **kwargs,
    ):
        is_batched_numpy = (
            isinstance(raw_speech, np.ndarray) and len(raw_speech.shape) > 1
        )
        if is_batched_numpy and len(raw_speech.shape) > 2:
            raise ValueError(
                f"Only mono-channel audio is supported for input to {self}"
            )
        is_batched = is_batched_numpy or (
            isinstance(raw_speech, (list, tuple))
            and (isinstance(raw_speech[0], (np.ndarray, tuple, list)))
        )

        if is_batched:
            raw_speech = [
                np.asarray([speech], dtype=np.float32).T for speech in raw_speech
            ]
        elif not is_batched and not isinstance(raw_speech, np.ndarray):
            raw_speech = np.asarray(raw_speech, dtype=np.float32)
        elif isinstance(raw_speech, np.ndarray) and raw_speech.dtype is np.dtype(
            np.float64
        ):
            raw_speech = raw_speech.astype(np.float32)

        # always return batch
        if not is_batched:
            raw_speech = [np.asarray([raw_speech]).T]

        input_lengths = torch.tensor([len(speech) for speech in raw_speech])

        batched_speech = BatchFeature({"input_features": raw_speech})

        padded_inputs = self.pad(
            batched_speech,
            padding=padding,
            max_length=self.n_samples,
            truncation=False,
            return_tensors="pt",
        )

        input_features = padded_inputs["input_features"].transpose(1, 2)
        input_features = self.featurizer(input_features).squeeze(1)
        input_features = torch.log(input_features.clamp_(1e-9, 1e9))
        input_lengths = self.out_len(input_lengths)

        return BatchFeature({"input_features": input_features, "input_lengths": input_lengths}, tensor_type="pt")


class GigaAMCTCTokenizer(Wav2Vec2CTCTokenizer):
    """
    Char tokenizer for GigaAM-CTC model.
    """
    def __init__(
        self,
        vocab_file,
        unk_token="[BLANK]",
        pad_token="[BLANK]",
        bos_token=None,
        eos_token=None,
        word_delimiter_token=" ",
        **kwargs,
    ):
        super().__init__(
            vocab_file=vocab_file,
            unk_token=unk_token,
            pad_token=pad_token,
            bos_token=bos_token,
            eos_token=eos_token,
            word_delimiter_token=word_delimiter_token,
            **kwargs,
        )


class GigaAMProcessor(Wav2Vec2Processor):
    feature_extractor_class = "GigaAMFeatureExtractor"
    tokenizer_class = "GigaAMCTCTokenizer"

    def __init__(self, feature_extractor, tokenizer):
        # super().__init__(feature_extractor, tokenizer)
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.current_processor = self.feature_extractor
        self._in_target_context_manager = False

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        feature_extractor = GigaAMFeatureExtractor.from_pretrained(pretrained_model_name_or_path, **kwargs)
        tokenizer = GigaAMCTCTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)

        return cls(feature_extractor=feature_extractor, tokenizer=tokenizer)


class GigaAMConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class GigaAMCTCHF(PreTrainedModel):
    """
    GigaAM-CTC model for transformers
    """
    config_class = GigaAMConfig
    base_model_prefix = "gigaamctc"
    main_input_name = "input_features"

    def __init__(self, config: GigaAMConfig):
        super().__init__(config)
        self.model = GigaAMCTC(config.encoder, config.head)

    def forward(self, input_features, input_lengths, labels=None, **kwargs):

        # B x C x T
        logits, encoded_lengths = self.model(input_features, input_lengths)
        # B x C x T -> B x T x C -> T x B x C
        log_probs = torch.log_softmax(
            logits.transpose(1, 2), dim=-1, dtype=torch.float32
        ).transpose(0, 1)

        loss = None
        if labels is not None:
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            loss = nn.functional.ctc_loss(
                log_probs,
                flattened_targets,
                encoded_lengths,
                target_lengths,
                blank=self.config.blank_id,
                zero_infinity=True,
            )

        return CausalLMOutput(loss=loss, logits=logits.transpose(1, 2))
