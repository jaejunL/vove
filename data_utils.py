import os
import glob
import random
import pickle
import numpy as np
import pandas as pd

import torch
import torch.utils.data
import torchaudio.transforms as T
from torch.nn import functional as F

import utils
from modules.mel_processing import spectrogram_torch, spec_to_mel_torch
from utils import load_filepaths_and_text, load_wav_to_torch

class Dataset_Main(torch.utils.data.Dataset):
    def __init__(self, data_root, filelist_root, label_root, typ):
        self.typ = typ
        self.data_root = data_root
        
        if typ == 'train':
            datasets = ['train-clean-100', 'train-clean-360', 'train-other-500']
        elif typ == 'valid':
            datasets = ['dev-clean', 'dev-other']
        elif typ == 'test':
            datasets = ['test-clean', 'test-other']
        elif typ == 'full':
            datasets = ['train-clean-100', 'train-clean-360', 'train-other-500',
                        'dev-clean', 'dev-other', 'test-clean', 'test-other']
        elif typ == 'full_test':
            datasets = ['test-clean', 'test-other']
            
        self.wav_dirs = []
        for dataset in datasets:
            with open(os.path.join(filelist_root, dataset + '.txt'), mode='rb') as f:
                data = pickle.load(f)
            self.wav_dirs += data

        with open(label_root, 'rb') as fr:
            self.labels = pickle.load(fr)            
        
        if typ != 'test':
            random.seed(1234)
            random.shuffle(self.wav_dirs)
        self.max_wav_value = 32768.0
        self.sampling_rate = 16000
        self.slice_sec = 4
        self.slice_len = self.slice_sec * self.sampling_rate
        
    def get_audio(self, wav_dir):
        audio, sampling_rate = load_wav_to_torch(os.path.join(self.data_root, wav_dir))
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        # spec = spectrogram_torch(audio_norm, self.filter_length,
                                    # self.sampling_rate, self.hop_length, self.win_length,
                                    # center=False)
        # spec = torch.squeeze(spec, 0)
        return audio_norm
    
    def get_label(self, speaker):
        label = np.clip(self.labels[speaker], a_min=0, a_max=1)
        return torch.from_numpy(label)
    
    def __getitem__(self, index):
        wav_dir = self.wav_dirs[index]
        audio_norm = self.get_audio(wav_dir)        
        speaker = os.path.basename(os.path.dirname(os.path.dirname(wav_dir)))
        label = self.get_label(speaker)
        
        if self.typ == 'test':
            return audio_norm, label, speaker
        else:
            return self.random_slice(audio_norm), label, speaker

    def random_slice(self, audio_norm):
        if audio_norm.shape[-1] > self.slice_len:
            start = random.randint(0, audio_norm.shape[-1]-self.slice_len)
            end = start + self.slice_len
            audio_norm = audio_norm[:, start : end]
        return audio_norm
    
    def __len__(self):
        return len(self.wav_dirs)


class Dataset_Speaker(torch.utils.data.Dataset):
    def __init__(self, data_root, filelist_root, label_root, typ):
        self.typ = typ
        self.data_root = data_root
        
        if typ == 'train':
            datasets = ['train-clean-100', 'train-clean-360', 'train-other-500']
        elif typ == 'valid':
            datasets = ['dev-clean', 'dev-other']
        elif typ == 'test':
            datasets = ['test-clean', 'test-other']
        elif typ == 'full':
            datasets = ['train-clean-100', 'train-clean-360', 'train-other-500',
                        'dev-clean', 'dev-other', 'test-clean', 'test-other']
        elif typ == 'full_test':
            datasets = ['test-clean', 'test-other']
            
        self.wav_dirs = []
        for dataset in datasets:
            with open(os.path.join(filelist_root, dataset + '.txt'), mode='rb') as f:
                data = pickle.load(f)
            self.wav_dirs += data

        speakers = []
        for wav_dir in self.wav_dirs:
            speaker = os.path.basename(os.path.dirname(os.path.dirname(wav_dir)))
            speakers.append(speaker)
        self.unique_speakers = list(set(speakers))
        self.unique_speakers.sort()
            
        with open(label_root, 'rb') as fr:
            self.labels = pickle.load(fr)
        
        if typ != 'test':
            random.seed(1234)
            random.shuffle(self.wav_dirs)
        self.max_wav_value = 32768.0
        self.sampling_rate = 16000
        self.slice_sec = 4
        self.slice_len = self.slice_sec * self.sampling_rate
        
    def get_audio(self, wav_dir):
        audio, sampling_rate = load_wav_to_torch(os.path.join(self.data_root, wav_dir))
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        # spec = spectrogram_torch(audio_norm, self.filter_length,
                                    # self.sampling_rate, self.hop_length, self.win_length,
                                    # center=False)
        # spec = torch.squeeze(spec, 0)
        return audio_norm
    
    def get_label(self, speaker):
        label = np.clip(self.labels[speaker], a_min=0, a_max=1)
        return torch.from_numpy(label)
    
    def __getitem__(self, index):
        wav_dir = self.wav_dirs[index]
        audio_norm = self.get_audio(wav_dir)        
        speaker = os.path.basename(os.path.dirname(os.path.dirname(wav_dir)))
        label = self.get_label(speaker)
        
        if self.typ == 'test':
            return audio_norm, label, speaker
        else:
            return self.random_slice(audio_norm), label, self.unique_speakers.index(speaker)

    def random_slice(self, audio_norm):
        if audio_norm.shape[-1] > self.slice_len:
            start = random.randint(0, audio_norm.shape[-1]-self.slice_len)
            end = start + self.slice_len
            audio_norm = audio_norm[:, start : end]
        return audio_norm
    
    def __len__(self):
        return len(self.wav_dirs)
    
class Collate:
    def __call__(self, batch):
        batch = [b for b in batch if b is not None]

        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[0].shape[-1] for x in batch]),
            dim=0, descending=True)

        max_wav_len = max([x[0].size(-1) for x in batch])
        
        wav_padded = torch.FloatTensor(len(batch), max_wav_len)
        labels = torch.FloatTensor(len(batch), 44)
        lengths = torch.LongTensor(len(batch))
        
        wav_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            wav = row[0]
            wav_padded[i, :wav.size(-1)] = wav
            
            label = row[1]
            labels[i] = label
            
            lengths[i] = wav.size(-1)
            
        return wav_padded, labels, lengths

class Collate_Speaker:
    def __call__(self, batch):
        batch = [b for b in batch if b is not None]

        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[0].shape[-1] for x in batch]),
            dim=0, descending=True)

        max_wav_len = max([x[0].size(-1) for x in batch])
        
        wav_padded = torch.FloatTensor(len(batch), max_wav_len)
        labels = torch.FloatTensor(len(batch), 44)
        lengths = torch.LongTensor(len(batch))
        speakers = torch.LongTensor(len(batch))
        
        wav_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            wav = row[0]
            wav_padded[i, :wav.size(-1)] = wav
            
            label = row[1]
            labels[i] = label
            
            lengths[i] = wav.size(-1)
            
            speakers[i] = row[-1]
        return wav_padded, labels, lengths, speakers
    
# data_root = "/disk2/LibriTTS-R/modified/wav16"
# filelist_root = "/home/jaejun/libri_p/filelists"

# datasets = ['train-clean-100', 'train-clean-360', 'train-other-500']
# wav_dirs = []
# for dataset in datasets:
#     with open(os.path.join(filelist_root, dataset + '.txt'), mode='rb') as f:
#         data = pickle.load(f)
#     wav_dirs += data

# train_speakers = []
# for wav_dir in wav_dirs:
#     speaker = os.path.basename(os.path.dirname(os.path.dirname(wav_dir)))
#     train_speakers.append(speaker)

# print(len(set(train_speakers)))
    
# datasets = ['dev-clean', 'dev-other']
# wav_dirs = []
# for dataset in datasets:
#     with open(os.path.join(filelist_root, dataset + '.txt'), mode='rb') as f:
#         data = pickle.load(f)
#     wav_dirs += data

# valid_speakers = []
# for wav_dir in wav_dirs:
#     speaker = os.path.basename(os.path.dirname(os.path.dirname(wav_dir)))
#     valid_speakers.append(speaker)

# print(len(set(valid_speakers)))


# datasets = ['test-clean', 'test-other']
# wav_dirs = []
# for dataset in datasets:
#     with open(os.path.join(filelist_root, dataset + '.txt'), mode='rb') as f:
#         data = pickle.load(f)
#     wav_dirs += data

# test_speakers = []
# for wav_dir in wav_dirs:
#     speaker = os.path.basename(os.path.dirname(os.path.dirname(wav_dir)))
#     test_speakers.append(speaker)

# print(len(set(test_speakers)))

