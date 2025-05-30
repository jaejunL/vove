from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
import modules.mel_processing as mel_processing

class Res2Block(nn.Module):
    """Multi-scale residual blocks.
    """
    def __init__(self, channels: int, scale: int, kernels: int, dilation: int):
        """Initializer.
        Args:
            channels: size of the input channels.
            scale: the number of the blocks.
            kenels: size of the convolutional kernels.
            dilation: dilation factors.
        """
        super().__init__()
        assert channels % scale == 0, \
            f'size of the input channels(={channels})' \
            f' should be factorized by scale(={scale})'
        width = channels // scale
        self.scale = scale
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    width, width, kernels,
                    padding=(kernels - 1) * dilation // 2,
                    dilation=dilation),
                nn.ReLU(),
                nn.BatchNorm1d(width))
            for _ in range(scale - 1)])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Transform the inputs.
        Args:
            inputs: [torch.float32; [B, C, T]], input 1D tensor,
                where C = `channels`.
        Returns:
            [torch.float32; [B, C, T]], transformed.
        """
        # [B, W, T], (S - 1) x [B, W, T] where W = C // S
        straight, *xs = inputs.chunk(self.scale, dim=1)
        # [B, W, T]
        base = torch.zeros_like(straight)
        # S x [B, W, T]
        outs = [straight]
        for x, conv in zip(xs, self.convs):
            # [B, W, T], increasing receptive field progressively
            base = conv(x + base)
            outs.append(base)
        # [B, C, T]
        return torch.cat(outs, dim=1)
        
class SERes2Block(nn.Module):
    """Multiscale residual block with Squeeze-Excitation modules.
    """
    def __init__(self,
                 channels: int,
                 scale: int,
                 kernels: int,
                 dilation: int,
                 bottleneck: int):
        """Initializer.
        Args:
            channels: size of the input channels.
            scale: the number of the resolutions, for res2block.
            kernels: size of the convolutional kernels.
            dilation: dilation factor.
            bottleneck: size of the bottleneck layers for squeeze and excitation.
        """
        super().__init__()
        self.preblock = nn.Sequential(
            nn.Conv1d(channels, channels, 1),
            nn.ReLU(),
            nn.BatchNorm1d(channels))
        
        self.res2block = Res2Block(channels, scale, kernels, dilation)

        self.postblock = nn.Sequential(
            nn.Conv1d(channels, channels, 1),
            nn.ReLU(),
            nn.BatchNorm1d(channels))

        self.excitation = nn.Sequential(
            nn.Linear(channels, bottleneck),
            nn.ReLU(),
            nn.Linear(bottleneck, channels),
            nn.Sigmoid())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Transform the inputs.
        Args:
            inputs: [torch.float32; [B, C, T]], input tensors,
                where C = `channels`.
        Returns:
            [torch.float32; [B, C, T]], transformed.
        """
        # [B, C, T]
        x = self.preblock(inputs)
        # [B, C, T], res2net, multi-scale architecture
        x = self.res2block(x)
        # [B, C, T]
        x = self.postblock(x)
        # [B, C], squeeze and excitation
        scale = self.excitation(x.mean(dim=-1))
        # [B, C, T]
        x = x * scale[..., None]
        # residual connection
        return x + inputs


class AttentiveStatisticsPooling(nn.Module):
    """Attentive statistics pooling.
    """
    def __init__(self, channels: int, bottleneck: int):
        """Initializer.
        Args:
            channels: size of the input channels.
            bottleneck: size of the bottleneck.
        """
        super().__init__()
        # nonlinear=Tanh
        # ref: https://github.com/KrishnaDN/Attentive-Statistics-Pooling-for-Deep-Speaker-Embedding
        # ref: https://github.com/TaoRuijie/ECAPA-TDNN
        self.attention = nn.Sequential(
            nn.Conv1d(channels, bottleneck, 1),
            nn.Tanh(),
            nn.Conv1d(bottleneck, channels, 1),
            nn.Softmax(dim=-1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Pooling with weighted statistics.
        Args:
            inputs: [torch.float32; [B, C, T]], input tensors,
                where C = `channels`.
        Returns:
            [torch.float32; [B, C x 2]], weighted statistics.
        """
        # [B, C, T]
        weights = self.attention(inputs)
        # [B, C]
        mean = torch.sum(weights * inputs, dim=-1)
        var = torch.sum(weights * inputs ** 2, dim=-1) - mean ** 2
        # [B, C x 2], for numerical stability of square root
        return torch.cat([mean, (var + 1e-7).sqrt()], dim=-1)
        
class ECAPA_TDNN(nn.Module):
    """ECAPA-TDNN: Emphasized Channel Attention,
    [1] Propagation and Aggregation in TDNN Based Speaker Verification,
        Desplanques et al., 2020, arXiv:2005.07143.
    [2] Res2Net: A New Multi-scale Backbone architecture,
        Gao et al., 2019, arXiv:1904.01169.
    [3] Squeeze-and-Excitation Networks, Hu et al., 2017, arXiv:1709.01507.
    [4] Attentive Statistics Pooling for Deep Speaker Embedding,
        Okabe et al., 2018, arXiv:1803.10963.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_classes: int,
                 channels: int,
                 prekernels: int,
                 scale: int,
                 kernels: int,
                 dilations: List[int],
                 bottleneck: int,
                 hiddens: int,
                 ):
        """Initializer.
        Args:
            in_channels: size of the input channels.
            out_channels: size of the output embeddings.
            channels: size of the major states.
            prekernels: size of the convolutional kernels before feed to SERes2Block.
            scale: the number of the resolutions, for SERes2Block.
            kernels: size of the convolutional kernels, for SERes2Block.
            dilations: dilation factors.
            bottleneck: size of the bottleneck layers,
                both SERes2Block and AttentiveStatisticsPooling.
            hiddens: size of the hidden channels for attentive statistics pooling.
            latent: size of the timber latent query.
        """
        super().__init__()
        # channels=512, prekernels=5
        # ref:[1], Figure2 and Page3, "architecture with either 512 or 1024 channels"
        self.preblock = nn.Sequential(
            nn.Conv1d(in_channels, channels, prekernels, padding=prekernels // 2),
            nn.ReLU(),
            nn.BatchNorm1d(channels))
        # scale=8, kernels=3, dilations=[2, 3, 4], bottleneck=128
        self.blocks = nn.ModuleList([
            SERes2Block(channels, scale, kernels, dilation, bottleneck)
            for dilation in dilations])
        # hiddens=1536
        # TODO: hiddens=3072 on NANSY++
        self.conv1x1 = nn.Sequential(
            nn.Conv1d(len(dilations) * channels, hiddens, 1),
            nn.ReLU())
        # attentive pooling and additional projector
        # out_channels=192
        self.pool = nn.Sequential(
            AttentiveStatisticsPooling(hiddens, bottleneck),
            nn.BatchNorm1d(hiddens * 2),
            nn.Linear(hiddens * 2, out_channels),
            nn.BatchNorm1d(out_channels))
        self.fc = nn.Linear(out_channels, num_classes)
        self.bn = nn.BatchNorm1d(num_classes)
        
    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the x-vectors from the input sequence.
        Args:
            inputs: [torch.float32; [B, in_channels, T]], input sequences,
        Returns:
            [torch.float32; [B, out_channels]], global x-vectors,
            [torch.float32; [B, timber, tokens]], timber token bank.
        """
        # [B, C, T]
        x = self.preblock(inputs)
        # N x [B, C, T]
        xs = []
        for block in self.blocks:
            # [B, C, T]
            x = block(x)
            xs.append(x)
        # [B, H, T]
        mfa = self.conv1x1(torch.cat(xs, dim=1))
        # [B, O]
        global_ = F.normalize(self.pool(mfa), p=2, dim=-1)
        logit = self.bn(self.fc(global_))
        return logit
    
    def infer(self, inputs: torch.Tensor):
        # [B, C, T]
        x = self.preblock(inputs)
        # N x [B, C, T]
        xs = []
        for block in self.blocks:
            # [B, C, T]
            x = block(x)
            xs.append(x)
        # [B, H, T]
        mfa = self.conv1x1(torch.cat(xs, dim=1))
        # [B, O]
        global_ = F.normalize(self.pool(mfa), p=2, dim=-1)
        logit = self.bn(self.fc(global_))
        return logit

class Vove_Ecapa(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_classes: int,
                 channels: int,
                 prekernels: int,
                 scale: int,
                 kernels: int,
                 dilations: List[int],
                 bottleneck: int,
                 hiddens: int):
        super().__init__()
        
        # self.sample_rate = sampling_rate
        self.ecapa_tdnn = ECAPA_TDNN(in_channels, out_channels, num_classes, channels, prekernels, scale, kernels, dilations, bottleneck, hiddens) 
        # if self.sample_rate != 16000:
            # self.resampler = T.Resample(sampling_rate, 16000)
        self.relu = nn.ReLU()
        
        num_speakers = 2443
        self.fc = nn.Linear(num_classes, num_speakers)
        self.bn = nn.BatchNorm1d(num_speakers)
        self.sigmoid = nn.Sigmoid()

        self.attributes = ['adult-like', 'bright', 'calm', 'clear', 'cool', 'cute', 'dark', 'elegant',
                    'feminine', 'fluent', 'friendly', 'gender-neutral', 'halting', 'hard', 'intellectual', 'intense',
                    'kind', 'light', 'lively', 'masculine', 'mature', 'middle-aged', 'modest', 'muffled', 'nasal',
                    'old', 'powerful', 'raspy', 'reassuring', 'refreshing', 'relaxed','sexy', 'sharp', 'sincere',
                    'soft', 'strict', 'sweet', 'tensed', 'thick', 'thin', 'unique', 'weak', 'wild', 'young']

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # if self.sample_rate != 16000:
            # inputs = self.resampler(inputs)
        mel = mel_processing.mel_spectrogram_torch(inputs.squeeze(1), 1024, 80, 16000, 256, 1024, 0, 8000)
        class_emb = self.ecapa_tdnn(mel)
        speaker_emb = self.bn(self.fc(self.relu(class_emb)))
        return class_emb, speaker_emb

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        mel = mel_processing.mel_spectrogram_torch(inputs.squeeze(1), 1024, 80, 16000, 256, 1024, 0, 8000)
        class_emb = self.sigmoid(self.ecapa_tdnn(mel))
        return class_emb.squeeze()
