# srllctts
A simple utility for synthesizing English speech from the command line; uses
NVIDIA's Tacotron2 and WaveGlow models to do the work, both of which were trained
using the LJ Speech dataset.  Just a quick little thing that we thought was neat.

Some of the code is taken directly from NVIDIA's TorchHub example (see links).

Please note that this is not a maintained project.  Additionally, it also no longer represents state-of-the-art; while I haven't taken the time to investigate the following, it may be of interest:

[MelGAN Vocoder code](https://github.com/seungwonpark/melgan), and [paper](https://arxiv.org/abs/1910.06711). 

Links
-----
- [NVIDIA's WaveGlow example](https://pytorch.org/hub/nvidia_deeplearningexamples_waveglow/)
- [Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions](https://arxiv.org/abs/1712.05884)
- [WaveGlow: A Flow-based Generative Network for Speech Synthesis](https://arxiv.org/abs/1811.00002)

Samples of the output
---------------------
[Decent: knuth.wav](knuth.wav)  
[Really bad: shakespeare.wav](shakespeare.wav)

Dependencies
------------
You can `pip install -r DEPENDENCIES` to get these.

- torch
- matplotlib
- numpy
- inflect
- librosa
- scipy
- unidecode
- plac

Execution time
--------------
With a GTX 1080 Ti video card and an Intel Core i7-7700k (4.2GHz), it takes roughly a second per word or two.

Licenses
--------
The LJ Speech dataset is public domain and NVIDIA's models are covered by a BSD 3-clause license.
Imagine the court battles Hollywood is going to go through when we really get these things right.
