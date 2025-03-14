import os
import json
import torch
from models.vove_ecapa import Vove_Ecapa
from utils import HParams, load_checkpoint, load_filepaths_and_text, load_wav_to_torch

attributes = ['adult-like', 'bright', 'calm', 'clear', 'cool', 'cute', 'dark', 'elegant',
            'feminine', 'fluent', 'friendly', 'gender-neutral', 'halting', 'hard', 'intellectual', 'intense',
            'kind', 'light', 'lively', 'masculine', 'mature', 'middle-aged', 'modest', 'muffled', 'nasal',
            'old', 'powerful', 'raspy', 'reassuring', 'refreshing', 'relaxed','sexy', 'sharp', 'sincere',
            'soft', 'strict', 'sweet', 'tensed', 'thick', 'thin', 'unique', 'weak', 'wild', 'young']

if __name__ == "__main__":
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning) 
    warnings.simplefilter(action='ignore', category=UserWarning) 
    
    parser = argparse.ArgumentParser()
    # set parameters
    parser.add_argument('--ckpt_dir', type=str, default="ckpt/vove.pth", help='your-VOVE-model-root')
    parser.add_argument('--sample_dir', type=str, default='sample/source.wav', help='your-source-audio-root')
    parser.add_argument('--device', type=str, default='cpu', help='cpu or cuda')
    base_args = parser.parse_args()

    configs_dir = f'configs/vove_ecapa.json'
    with open(configs_dir, "r") as f:
        data = f.read()
    config = json.loads(data)
    args = HParams(**config)
    args.base_args = base_args

    ckpt_path = args.base_args.ckpt_dir
    args.base_args.device = 'cuda:0' if 'cuda' in base_args.device else 'cpu'

    vove = Vove_Ecapa(**args.model)
    vove, _, _, _ = load_checkpoint(ckpt_path, vove, None)
    vove = vove.to(device)
    vove = vove.eval()

    # Inference
    audio, sampling_rate = load_wav_to_torch(args.base_args.sample_dir)
    audio_norm = audio / 32768.0
    audio_norm = audio_norm.unsqueeze(0)

    with torch.no_grad():
        pred, _ = vove(audio_norm.to(device))

    attribute_dict = {}
    for idx, attribute in enumerate(vove.attributes):
        attribute_dict[attribute] = pred[idx].detach().cpu().item()
    print(sorted(attribute_dict.items(), key=lambda x: x[1], reverse=True))
    print(f"Prediction complted")
    
# CUDA_VISIBLE_DEVICES=0 python inference/inference.py