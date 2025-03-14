# Vo-Ve: An Explainable Voice-Vector for Speaker Identity Evaluation
This repository contains the official implementation of our paper submitted to Interspeech 2025.

> **Note**: The code has now been released! If you have any questions or encounter any issues while running it, feel free to contact us.
The released version follows the same training strategy described in our paper.

> As mentioned in the discussion section, we will be uploading a real-world robust model. This model will incorporate several data robustness modules, such as data augmentation with noise and reverberation.

# Inference
1. To perform inference:\
Save the model weights (`vove.pth`) in the `ckpt` folder.\
Model weights can be downloaded from our Google Drive: [link](https://drive.google.com/drive/folders/17JXnrx2UUoZUg7LrOCeP8XZqTFkh7fXD?usp=sharing).

2. Prepare input audio:\
Sample audio files from the VCTK dataset are available in the `sample` folder.
Alternatively, you can use your own audio files by modifying the parser inside 'predict.py'.

3. Run the following command:
```
# Run on GPU
CUDA_VISIBLE_DEVICES=0 python predict.py --ckpt_dir=ckpt/vove.pth --sample_dir=sample/vctk_female1_p276_002.wav --device=cuda

# Run on CPU
python predict.py --ckpt_dir=ckpt/vove.pth --sample_dir=sample/vctk_female1_p276_002.wav --device=cpu
```

# Voice attributes
You can check the 44 voice attributes (which represent the dimensions of the Vo-Ve model's output) by accessing `model.attributes`.\
For details, please refer to `predict.py` or our demo file.
```
['adult-like', 'bright', 'calm', 'clear', 'cool', 'cute', 'dark', 'elegant', 'feminine', 'fluent', 'friendly', 'gender-neutral', 'halting', 'hard', 'intellectual', 'intense', 'kind', 'light', 'lively', 'masculine', 'mature', 'middle-aged', 'modest', 'muffled', 'nasal', 'old', 'powerful', 'raspy', 'reassuring', 'refreshing', 'relaxed', 'sexy', 'sharp', 'sincere', 'soft', 'strict', 'sweet', 'tensed', 'thick', 'thin', 'unique', 'weak', 'wild', 'young']
```

# Demo
Please check our demo in `demo.ipynb`. Using four sample audio files from the sample folder, we compare each speech sample and predict its attributes.\
In the `.ipynb` file, we compare the top-10 attributes between speakers of the same gender. Listen to the samples and check whether they make sense.
