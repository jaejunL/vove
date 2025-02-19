import os
import glob
import numpy as np
from scipy.io.wavfile import read

import torch
from torch.nn import functional as F

f0_bin = 256
f0_max = 1100.0
f0_min = 50.0
f0_mel_min = 1127 * np.log(1 + f0_min / 700)
f0_mel_max = 1127 * np.log(1 + f0_max / 700)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')  
    
def load_wav_to_torch(full_path):
  sampling_rate, data = read(full_path)
  return torch.FloatTensor(data.astype(np.float32)), sampling_rate

def load_filepaths_and_text(filename, split="|"):
  with open(filename, encoding='utf-8') as f:
    filepaths_and_text = [line.strip().split(split) for line in f]
  return filepaths_and_text

def repeat_expand_2d(content, target_len, mode = 'left'):
    # content : [h, t]
    return repeat_expand_2d_left(content, target_len) if mode == 'left' else repeat_expand_2d_other(content, target_len, mode)

def repeat_expand_2d_left(content, target_len):
    # content : [h, t]

    src_len = content.shape[-1]
    target = torch.zeros([content.shape[0], target_len], dtype=torch.float).to(content.device)
    temp = torch.arange(src_len+1) * target_len / src_len
    current_pos = 0
    for i in range(target_len):
        if i < temp[current_pos+1]:
            target[:, i] = content[:, current_pos]
        else:
            current_pos += 1
            target[:, i] = content[:, current_pos]

    return target

# mode : 'nearest'| 'linear'| 'bilinear'| 'bicubic'| 'trilinear'| 'area'
def repeat_expand_2d_other(content, target_len, mode = 'nearest'):
    # content : [h, t]
    content = content[None,:,:]
    target = F.interpolate(content,size=target_len,mode=mode)[0]
    return target

def load_checkpoint(checkpoint_path, model, optimizer=None):
    print('Checkpoint Load starting..')
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    iteration = checkpoint_dict['iteration']
    learning_rate = checkpoint_dict['learning_rate']
    if optimizer is not None and checkpoint_dict['optimizer'] is not None:
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
    saved_state_dict = checkpoint_dict['model']
    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    new_state_dict= {}
    for k, v in state_dict.items():
        try:
            new_state_dict[k] = saved_state_dict[k]
        except:
            print("Model {} is not in the checkpoint".format(k))
            new_state_dict[k] = v
    if hasattr(model, 'module'):
        model.module.load_state_dict(new_state_dict, strict=False)
    else:
        model.load_state_dict(new_state_dict, strict=False)
    print("Loaded checkpoint '{}' (Epoch {})" .format(checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration

def save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path):
    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save({'model': state_dict,
                'iteration': iteration,
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, checkpoint_path)

def latest_checkpoint_path(dir_path, regex="G_*.pth"):
    f_list = glob.glob(os.path.join(dir_path, regex))
    f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    x = f_list[-1]
    print(x)
    return x

def f0_to_coarse(f0):
    f0_mel = 1127 * (1 + f0 / 700).log()
    a = (f0_bin - 2) / (f0_mel_max - f0_mel_min)
    b = f0_mel_min * a - 1.
    f0_mel = torch.where(f0_mel > 0, f0_mel * a - b, f0_mel)
    # torch.clip_(f0_mel, min=1., max=float(f0_bin - 1))
    f0_coarse = torch.round(f0_mel).long()
    f0_coarse = f0_coarse * (f0_coarse > 0)
    f0_coarse = f0_coarse + ((f0_coarse < 1) * 1)
    f0_coarse = f0_coarse * (f0_coarse < f0_bin)
    f0_coarse = f0_coarse + ((f0_coarse >= f0_bin) * (f0_bin - 1))
    return f0_coarse

def f0_normalize(f0, reverse=False):
    if reverse:
        return 700 * (torch.pow(10, f0 * 500 / 2595) - 1)
    else:
        return 2595. * torch.log10(1. + f0 / 700.) / 500

def e_normalize(v, norm_val=100, reverse=False):
    if reverse:
        return v * norm_val
    else:
        return v / norm_val

def get_f0_predictor(f0_predictor,hop_length,sampling_rate,**kargs):
    if f0_predictor == "pm":
        from modules.F0Predictor.PMF0Predictor import PMF0Predictor
        f0_predictor_object = PMF0Predictor(hop_length=hop_length,sampling_rate=sampling_rate)
    elif f0_predictor == "crepe":
        from modules.F0Predictor.CrepeF0Predictor import CrepeF0Predictor
        f0_predictor_object = CrepeF0Predictor(hop_length=hop_length,sampling_rate=sampling_rate,device=kargs["device"],threshold=kargs["threshold"])
    elif f0_predictor == "harvest":
        from modules.F0Predictor.HarvestF0Predictor import HarvestF0Predictor
        f0_predictor_object = HarvestF0Predictor(hop_length=hop_length,sampling_rate=sampling_rate)
    elif f0_predictor == "dio":
        from modules.F0Predictor.DioF0Predictor import DioF0Predictor
        f0_predictor_object = DioF0Predictor(hop_length=hop_length,sampling_rate=sampling_rate) 
    elif f0_predictor == "rmvpe":
        from modules.F0Predictor.RMVPEF0Predictor import RMVPEF0Predictor
        f0_predictor_object = RMVPEF0Predictor(hop_length=hop_length,sampling_rate=sampling_rate,dtype=torch.float32 ,device=kargs["device"],threshold=kargs["threshold"])
    elif f0_predictor == "fcpe":
        from modules.F0Predictor.FCPEF0Predictor import FCPEF0Predictor
        f0_predictor_object = FCPEF0Predictor(hop_length=hop_length,sampling_rate=sampling_rate,dtype=torch.float32 ,device=kargs["device"],threshold=kargs["threshold"])
    else:
        raise Exception("Unknown f0 predictor")
    return f0_predictor_object

class HParams():
  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      if type(v) == dict:
        v = HParams(**v)
      self[k] = v

  def keys(self):
    return self.__dict__.keys()

  def items(self):
    return self.__dict__.items()

  def values(self):
    return self.__dict__.values()

  def __len__(self):
    return len(self.__dict__)

  def __getitem__(self, key):
    return getattr(self, key)

  def __setitem__(self, key, value):
    return setattr(self, key, value)

  def __contains__(self, key):
    return key in self.__dict__

  def __repr__(self):
    return self.__dict__.__repr__()

  def get(self,index):
    return self.__dict__.get(index)