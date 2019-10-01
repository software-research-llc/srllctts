#!/usr/bin/env python3
"""A simple utility for synthesizing English speech from the command line; uses
NVIDIA's Tacotron2 and WaveGlow models to do the work, both of which were trained
using the LJ Speech dataset.  Just a quick little thing that we thought was neat.

Some of the code is taken directly from NVIDIA's TorchHub example:

https://pytorch.org/hub/nvidia_deeplearningexamples_waveglow/"""

import sys
import os
import uuid
import tempfile
import numpy as np
from scipy.io.wavfile import write
import torch
from torch import hub
import plac

tacotron2 = None
waveglow = None


def init():
    """Encapsulates the initialization and compensates for some minor quirks in TorchHub"""
    global tacotron2
    global waveglow
    tacotron2 = hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_tacotron2')
    tacotron2 = tacotron2.to('cuda')
    # Tell the model we're using it to evaluate, not to train (optimization)
    tacotron2.eval()
    sys.stderr.write("\n")

    waveglow = hub.load('nvidia/DeepLearningExamples:torchhub', 'nvidia_waveglow')
    waveglow = waveglow.remove_weightnorm(waveglow)
    waveglow = waveglow.to('cuda')
    # As above: disable training specific stuff
    waveglow.eval()
    sys.stderr.write("\n")

def tts(text, keep = False):
    """Perform text-to-speech on a given string"""
    sys.stderr.write("Processing '%s'...\n" % text)

    # preprocess the input
    sequence = np.array(tacotron2.text_to_sequence(text, ['english_cleaners']))[None, :]
    sequence = torch.from_numpy(sequence).to(device='cuda', dtype=torch.int64)

    # in a nutshell, torch.no_grad() speeds things up by disabling some training functions
    with torch.no_grad():
        _, mel, _, _ = tacotron2.infer(sequence)
        audio = waveglow.infer(mel)
    audio_numpy = audio[0].data.cpu().numpy()
    rate = 22050

    # Write out the binary wav
    filename = os.path.join(tempfile.gettempdir(), "%s.wav" % uuid.uuid4())
    try:
        write(filename, rate, audio_numpy)
        os.system("aplay -q %s" % filename)
    finally:
        if not keep:
            os.unlink(filename)
        else:
            return filename

def main(loop: ("Continue reading lines and speaking the contents", "flag", "l"),
         keep: ("Don't delete the temporary .wav file", "flag", "k"),
         *text: "The text to speak"):
    """Initialize things and perform the tts operation"""
    init()
    if text:
        concat = " ".join(text)
        wavfile = tts(concat, keep)
    if keep:
        print("The audio is in %s" % wavfile)
    while loop:
        try:
            if int(sys.version[0]) >= 3:
                text = input("> ")
            else:
                text = sys.stdin.readline()
        except EOFError:
            sys.exit()
        wavfile = tts(text, keep)
        if keep:
            print("The audio is in %s" % wavfile)


if __name__ == '__main__':
    # Call main() using plac (which handles the command line stuff for us)
    plac.call(main)
