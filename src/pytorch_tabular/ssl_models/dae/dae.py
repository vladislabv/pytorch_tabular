# Pytorch Tabular
# Author: Manu Joseph <manujoseph@gmail.com>
# For license information, see LICENSE.TXT
# Inspired by implementation https://github.com/ryancheunggit/tabular_dae
"""DenoisingAutoEncoder Model."""

from collections import namedtuple
from typing import Dict

import torch
import torch.nn as nn
from omegaconf import DictConfig

from ..base_model import SSLBaseModel
from ..common.heads import MultiTaskHead
from ..common.layers import MixedEmbedding1dLayer
from ..common.noise_generators import SwapNoiseCorrupter
from ...models.common.layers import Embedding2dLayer


class DenoisingAutoEncoderFeaturizer(nn.Module):
    output_tuple = namedtuple("output_tuple", ["features", "mask"])
    # Fix for pickling
    # https://codefying.com/2019/05/04/dont-get-in-a-pickle-with-a-namedtuple/
    output_tuple.__qualname__ = "DenoisingAutoEncoderFeaturizer.output_tuple"

    def __init__(self, encoder, config: DictConfig, **kwargs):
        super().__init__()
        self.encoder = encoder
        self.config = config
        self.pick_keys = ["binary", "categorical", "continuous", "embedding"]
        self._build_network()

    def _get_noise_probability(self, name):
        return self.config.noise_probabilities.get(name, self.config.default_noise_probability)

    def _build_embedding_layer(self):
        if self.config.encoder_config._config_name == "CategoryEmbeddingModelConfig":
            return MixedEmbedding1dLayer(
                continuous_dim=self.config.continuous_dim,
                categorical_embedding_dims=self.config.embedding_dims,
                max_onehot_cardinality=self.config.max_onehot_cardinality,
                embedding_dropout=self.config.embedding_dropout,
                batch_norm_continuous_input=self.config.batch_norm_continuous_input,
                virtual_batch_size=self.config.virtual_batch_size,
            )
        else:
            continuous_dim = self.config.continuous_dim if self.config.encoder_config._config_name == "FTTransformerConfig" else 0
            categorical_cardinality = [card for card, _ in self.config.embedding_dims]
            return Embedding2dLayer(
                continuous_dim=continuous_dim,
                categorical_cardinality=categorical_cardinality,
                embedding_dim=self.config.encoder_config.input_embed_dim,
                shared_embedding_strategy=self.config.encoder_config.share_embedding_strategy,
                frac_shared_embed=self.config.encoder_config.shared_embedding_fraction,
                embedding_bias=self.config.encoder_config.embedding_bias,
                batch_norm_continuous_input=self.config.encoder_config.batch_norm_continuous_input,
                embedding_dropout=self.config.encoder_config.embedding_dropout,
                initialization=self.config.encoder_config.embedding_initialization,
                virtual_batch_size=self.config.virtual_batch_size,
            )

    def _build_network(self):
        is_transformer = self.config.encoder_config._config_name != "CategoryEmbeddingModelConfig"
        if not is_transformer:
            swap_probabilities = []
             # conditions based on real cardinality (excluding missing value placeholder)
            for i, (name, (cardinality, embed_dim)) in enumerate(zip(self.config.categorical_cols, self.config.embedding_dims)):
                if cardinality == 2:
                    swap_probabilities += [self._get_noise_probability(name)]
                elif cardinality <= self.config.max_onehot_cardinality:
                    swap_probabilities += [self._get_noise_probability(name)] * cardinality
                else:
                    swap_probabilities += [self._get_noise_probability(name)] * embed_dim
            for name in self.config.continuous_cols:
                swap_probabilities += [self._get_noise_probability(name)]
        else:
            embed_dim = self.config.encoder_config.input_embed_dim
            names = self.config.categorical_cols + self.config.continuous_cols if self.config.encoder_config._config_name == "FTTransformerConfig" else self.config.categorical_cols
            swap_probabilities = sum([[self._get_noise_probability(name)] * embed_dim for name in names], [])
        self._swap_probabilities = swap_probabilities
        self.swap_noise = SwapNoiseCorrupter(swap_probabilities)

    def _concatenate_features(self, x: Dict):
        x = torch.cat([x[key] for key in self.pick_keys if x[key] is not None], 1)
        return x

    def forward(self, x: Dict, perturb: bool = True, return_input: bool = False):
        # (B, N, E)
        is_transformer = self.config.encoder_config._config_name != "CategoryEmbeddingModelConfig"
        if not is_transformer:
            x = self._concatenate_features(x)
        mask = None
        if perturb:
            with torch.no_grad():
                if is_transformer:
                    b, n, h = x.shape
                    x_flat = x.reshape(b, n * h)
                    x_flat, mask = self.swap_noise(x_flat)
                    x = x_flat.reshape(b, n, h)
                else:
                    x, mask = self.swap_noise(x)
        z = self.encoder(x)
        if return_input:
            return self.output_tuple(z, mask), x
        else:
            return self.output_tuple(z, mask)


class DenoisingAutoEncoderModel(SSLBaseModel):
    output_tuple = namedtuple("output_tuple", ["original", "reconstructed"])
    loss_weight_tuple = namedtuple("loss_weight_tuple", ["binary", "categorical", "continuous", "mask"])
    # fix for pickling
    # https://codefying.com/2019/05/04/dont-get-in-a-pickle-with-a-namedtuple/
    output_tuple.__qualname__ = "DenoisingAutoEncoderModel.output_tuple"
    loss_weight_tuple.__qualname__ = "DenoisingAutoEncoderModel.loss_weight_tuple"
    ALLOWED_MODELS = ["CategoryEmbeddingModelConfig", "FTTransformerConfig", "TabTransformerConfig"]

    def __init__(self, config: DictConfig, **kwargs):
        inferred_config = kwargs.get("inferred_config")
        if config.encoder_config._config_name == "CategoryEmbeddingModelConfig":
            encoded_cat_dims = 0
            for card, embd_dim in inferred_config.embedding_dims:
                if card == 2:
                    encoded_cat_dims += 1
                elif card <= config.max_onehot_cardinality:
                    encoded_cat_dims += card
                else:
                    encoded_cat_dims += embd_dim
            config.encoder_config._backbone_input_dim = encoded_cat_dims + len(config.continuous_cols)
        else:
            embed_dim = config.encoder_config.input_embed_dim
            cat_dim = len(inferred_config.embedding_dims)
            cont_dim = len(config.continuous_cols)
            if config.encoder_config._config_name == "FTTransformerConfig":
                config.encoder_config._backbone_input_dim = (cat_dim + cont_dim) * embed_dim
            else:
                config.encoder_config._backbone_input_dim = cat_dim * embed_dim + cont_dim
        assert config.encoder_config._config_name in self.ALLOWED_MODELS, (
            "Encoder must be one of the following: " + ", ".join(self.ALLOWED_MODELS)
        )
        if config.decoder_config is not None:
            assert config.decoder_config._config_name in self.ALLOWED_MODELS, (
                "Decoder must be one of the following: " + ", ".join(self.ALLOWED_MODELS)
            )
            if config.encoder_config._config_name == "CategoryEmbeddingModelConfig":
                if "-" in config.encoder_config.layers:
                    last_layer = int(config.encoder_config.layers.split("-")[-1])
                else:
                    last_layer = int(config.encoder_config.layers)
            elif config.encoder_config._config_name == "FTTransformerConfig":
                last_layer = config.encoder_config.input_embed_dim
            else:  # TabTransformerConfig
                last_layer = config.encoder_config.input_embed_dim * len(inferred_config.embedding_dims) + len(config.continuous_cols)
            config.decoder_config._backbone_input_dim = last_layer
        super().__init__(config, **kwargs)

    def _get_noise_probability(self, name):
        return self.hparams.noise_probabilities.get(name, self.hparams.default_noise_probability)

    @property
    def embedding_layer(self):
        return self._embedding

    @property
    def featurizer(self):
        return self._featurizer

    def _build_network(self):
        self._featurizer = DenoisingAutoEncoderFeaturizer(self.encoder, self.hparams)
        self._embedding = self._featurizer._build_embedding_layer()
        is_transformer = self.hparams.encoder_config._config_name != "CategoryEmbeddingModelConfig"
        if not is_transformer:
            self.reconstruction = MultiTaskHead(
                self.decoder.output_dim,
                n_binary=len(self._embedding._binary_feat_idx),
                n_categorical=len(self._embedding._onehot_feat_idx),
                n_numerical=self._embedding.embedded_cat_dim + len(self.hparams.continuous_cols),
                cardinality=[self._embedding.categorical_embedding_dims[i][0] for i in self._embedding._onehot_feat_idx],
            )
        else:
            self.reconstruction = nn.Linear(self.decoder.output_dim, len(self._featurizer.swap_noise.probas))
        self.mask_reconstruction = nn.Linear(self.decoder.output_dim, len(self._featurizer.swap_noise.probas))

    def _setup_loss(self):
        self.losses = {
            "binary": nn.BCEWithLogitsLoss(),
            "categorical": nn.CrossEntropyLoss(),
            "continuous": nn.MSELoss(),
            "mask": nn.BCEWithLogitsLoss(),
        }
        if self.hparams.loss_type_weights is None:
            self.loss_weights = self.loss_weight_tuple(*self._init_loss_weights())
        else:
            self.loss_weights = self.loss_weight_tuple(*self.hparams.loss_type_weights, self.hparams.mask_loss_weight)

    def _init_loss_weights(self):
        if self.hparams.encoder_config._config_name == "CategoryEmbeddingModelConfig":
            n_features = self.hparams.continuous_dim + len(self.hparams.embedding_dims)
            return [
                len(self.embedding_layer._binary_feat_idx) / n_features,
                len(self.embedding_layer._onehot_feat_idx) / n_features,
                self.hparams.continuous_dim + len(self.embedding_layer._embedding_feat_idx) / n_features,
                self.hparams.mask_loss_weight,
            ]
        else:
            return [0, 0, 1.0, self.hparams.mask_loss_weight]

    def _setup_metrics(self):
        return None

    def forward(self, x: Dict):
        if self.mode == "pretrain":
            is_transformer = self.hparams.encoder_config._config_name != "CategoryEmbeddingModelConfig"
            x = self.embedding_layer(x)
            if is_transformer:
                original_input = x.clone()
            features = self.featurizer(x, perturb=True)
            z, mask = features.features, features.mask
            z_hat = self.decoder(z)
            reconstructed_in = self.reconstruction(z_hat)
            reconstructed_mask = self.mask_reconstruction(z_hat)
            output_dict = {"mask": self.output_tuple(mask, reconstructed_mask)}
            if not is_transformer:
                if "continuous" in reconstructed_in.keys():
                    output_dict["continuous"] = self.output_tuple(
                        torch.cat(
                            [
                                i
                                for i in [
                                    x.get("continuous", None),
                                    x.get("embedding", None),
                                ]
                                if i is not None
                            ],
                            1,
                        ),
                        reconstructed_in["continuous"],
                    )
                if "categorical" in reconstructed_in.keys():
                    output_dict["categorical"] = self.output_tuple(x["_categorical_orig"], reconstructed_in["categorical"])
                if "binary" in reconstructed_in.keys():
                    output_dict["binary"] = self.output_tuple(x["binary"], reconstructed_in["binary"])
            else:
                flat_original = original_input.reshape(original_input.shape[0], -1)
                output_dict["reconstruction"] = self.output_tuple(flat_original, reconstructed_in)
            return output_dict
        else:  # self.mode == "finetune"
            is_transformer = self.hparams.encoder_config._config_name != "CategoryEmbeddingModelConfig"
            z, x = self.featurizer(x, perturb=False, return_input=True)
            if self.hparams.include_input_features_inference:
                if is_transformer:
                    x = x.reshape(x.shape[0], -1)
                return torch.cat([z.features, x], 1)
            else:
                return z.features

    def calculate_loss(self, output, tag, sync_dist=False):
        total_loss = 0
        for type_, out in output.items():
            if type_ == "categorical":
                loss = 0
                for i in range(out.original.size(-1)):
                    loss += self.losses[type_](out.reconstructed[i], out.original[:, i])
            elif type_ == "binary":
                # Casting output to float for BCEWithLogitsLoss
                loss = self.losses[type_](out.reconstructed, out.original.float())
            elif type_ == "reconstruction":
                loss = self.losses["continuous"](out.reconstructed, out.original)
            else:
                loss = self.losses[type_](out.reconstructed, out.original)
            loss *= getattr(self.loss_weights, type_)
            self.log(
                f"{tag}_{type_}_loss",
                loss.item(),
                on_epoch=True,
                on_step=False,
                logger=True,
                prog_bar=False,
                sync_dist=sync_dist,
            )
            total_loss += loss
        self.log(
            f"{tag}_loss",
            total_loss,
            on_epoch=(tag == "valid") or (tag == "test"),
            on_step=(tag == "train"),
            # on_step=False,
            logger=True,
            prog_bar=True,
            sync_dist=sync_dist,
        )
        return total_loss

    def calculate_metrics(self, output, tag, sync_dist=False):
        pass

    def featurize(self, x: Dict):
        x = self.embedding_layer(x)
        return self.featurizer(x, perturb=False).features

    @property
    def output_dim(self):
        if self.mode == "finetune" and self.hparams.include_input_features_inference:
            return self._featurizer.encoder.output_dim + self.hparams.encoder_config._backbone_input_dim
        else:
            return self._featurizer.encoder.output_dim
