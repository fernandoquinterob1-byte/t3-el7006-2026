"""SEED model preserved from the legacy notebook."""

import numpy as np
import torch
import torch.nn as nn


EMBED_DIM = 64
EMBED_BLOCKS = 1
FILTER_IN = 64
BLOCK_DILATATION = 2
RESOL_FACTOR = 8

EMBEDDING_CONFIG_SEED = {
    "embed_dim": EMBED_DIM,
    "block_depth": 2,
    "blocks": EMBED_BLOCKS,
    "dropout": 0.0,
    "filters_in": FILTER_IN,
    "block_dilatation": BLOCK_DILATATION,
}
CLASSIFIER_CONFIG = {"embed_dim": 256, "hide_dim": 128}


class ConvBlockSEED(nn.Module):
    def __init__(self, in_channels, out_channels, block_depth, window):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels if index == 0 else out_channels,
                    out_channels,
                    window,
                    padding=int(window / 2),
                )
                for index in range(block_depth)
            ]
        )
        self.bns = nn.ModuleList([nn.BatchNorm1d(out_channels) for _ in range(block_depth)])

    def forward(self, x):
        for conv, batch_norm in zip(self.convs, self.bns):
            x = torch.relu(batch_norm(conv(x)))
        return x


class ConvEmbedSEED(nn.Module):
    def __init__(self, embed_dim, block_depth, blocks, dropout, window_len):
        super().__init__()
        self.bn = nn.BatchNorm1d(1)
        self.blocks = blocks
        self.conv_blocks = nn.ModuleList(
            [
                ConvBlockSEED(
                    1 if index == 0 else embed_dim // (2 ** (blocks - index)),
                    embed_dim // (2 ** (blocks - index - 1)),
                    block_depth,
                    window=window_len,
                )
                for index in range(blocks)
            ]
        )
        self.pool = nn.AvgPool1d(2, stride=2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.bn(x.unsqueeze(1))
        for conv_block in self.conv_blocks:
            x = self.pool(conv_block(x))
        return self.dropout(x)


class StageMultiDilatedConvolutions(nn.Module):
    def __init__(self, max_dil, filters_in, filters_out):
        super().__init__()
        max_exponent = int(np.round(np.log(max_dil) / np.log(2)))
        self.stage_config = [
            (int(filters_out / (2 ** (exponent + 1))), int(2**exponent))
            for exponent in range(max_exponent + 1)
        ]
        self.stage_config[-1] = (2 * self.stage_config[-1][0], self.stage_config[-1][1])
        self.branches = nn.ModuleList()
        for branch_filters, branch_dilation in self.stage_config:
            layers = []
            for layer_id in ("a", "b"):
                layers.extend(
                    [
                        nn.Conv1d(
                            filters_in if layer_id == "a" else branch_filters,
                            branch_filters,
                            kernel_size=3,
                            padding=branch_dilation,
                            dilation=branch_dilation,
                            bias=False,
                        ),
                        nn.BatchNorm1d(branch_filters),
                        nn.ReLU(),
                    ]
                )
            self.branches.append(nn.Sequential(*layers))

    def forward(self, inputs):
        return torch.cat([branch(inputs) for branch in self.branches], dim=1)


class LocalEncodingStageSEED(nn.Module):
    def __init__(
        self, embed_dim, block_depth, blocks, dropout, window_len, filters_in, block_dilatation
    ):
        super().__init__()
        self.conv_layer = ConvEmbedSEED(embed_dim, block_depth, blocks, dropout, window_len)
        self.layer_dilatations = nn.ModuleList(
            [
                StageMultiDilatedConvolutions(
                    8, filters_in * pow(2, index), filters_in * pow(2, index + 1)
                )
                for index in range(block_dilatation)
            ]
        )
        self.pool = nn.AvgPool1d(2, stride=2)

    def forward(self, inputs):
        output = self.conv_layer(inputs)
        for layer in self.layer_dilatations:
            output = self.pool(layer(output))
        return output.transpose(1, 2)


class BiLSTM(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.dropout = nn.Dropout(0.2)
        self.lstm = nn.LSTM(embed_dim, embed_dim, num_layers=2, dropout=0.5, bidirectional=True)

    def forward(self, x):
        x = self.dropout(x)
        hidden, _ = self.lstm(x.transpose(0, 1))
        return hidden.transpose(0, 1)


class REDFC(nn.Module):
    def __init__(self, embed_dim=256, hide_dim=128):
        super().__init__()
        self.dropout = nn.Dropout(0.5)
        self.hidden = nn.Linear(2 * embed_dim, hide_dim)
        nn.init.xavier_uniform_(self.hidden.weight, gain=nn.init.calculate_gain("relu"))
        self.hidden.bias.data.fill_(0.0)
        self.logits = nn.Linear(hide_dim, 1)
        nn.init.xavier_uniform_(self.logits.weight)
        p1 = 0.1
        self.logits.bias = nn.Parameter(torch.tensor([np.log(p1 / (1 - p1))], dtype=torch.float32))
        self.fc = nn.Sequential(self.hidden, nn.ReLU(), self.logits)

    def forward(self, x):
        return self.fc(self.dropout(x)).squeeze(-1)


class SEEDModel(nn.Module):
    """Binary temporal segmentation network returning logits of shape ``(batch, time)``."""

    def __init__(self):
        super().__init__()
        self.embedding = LocalEncodingStageSEED(**EMBEDDING_CONFIG_SEED, window_len=3)
        self.transformer = BiLSTM(FILTER_IN * (2**BLOCK_DILATATION))
        self.classifier = REDFC(**CLASSIFIER_CONFIG)
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x):
        output = self.classifier(self.transformer(self.embedding(x)))
        return output.repeat_interleave(RESOL_FACTOR, dim=1)
