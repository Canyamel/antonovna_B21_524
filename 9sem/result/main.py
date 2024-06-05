import os
import numpy as np
import matplotlib.pyplot as plt
import itertools
from scipy.io import wavfile
from scipy import signal
from tqdm import tqdm

def spectrogram(samples, sample_rate, filename):
    f, t, spec = signal.spectrogram(samples, sample_rate, scaling='spectrum', window=('hann'))

    spec = np.log10(spec + 1)
    plt.pcolormesh(t, f, spec, shading='gouraud', vmin=spec.min(), vmax=spec.max())
    plt.ylabel('Частота (Гц)')
    plt.xlabel('Время (с)')
    plt.savefig(filename)

    return f, t, spec


def denoised_savgol_filter(sample_rate, data, output_dir):
    denoised_savgol = signal.savgol_filter(data, 75, 5)
    wavfile.write(os.path.join(output_dir, 'savgol.wav'), sample_rate, denoised_savgol.astype(np.int16))
    spectrogram(denoised_savgol, sample_rate, os.path.join(output_dir, 'savgol.png'))


def get_peaks(sample_rate, data, output_dir):
    peaks = set()
    delta_t = 0.1
    delta_f = 50

    f, t, spec = spectrogram(data, sample_rate, os.path.join(output_dir, 'input.png'))

    for i in tqdm(range(len(f))):
        for j in range(len(t)):
            index_t = np.asarray(abs(t - t[j]) < delta_t).nonzero()[0]
            index_f = np.asarray(abs(f - f[i]) < delta_f).nonzero()[0]
            indexes = np.array([x for x in itertools.product(index_f, index_t)])
            flag = True
            for a, b in indexes:
                if spec[i, j] <= spec[a, b] and i != a and i != b:
                    flag = False
                    break
            if flag:
                peaks.add(t[j])

    with open(os.path.join(output_dir, 'peaks.txt'), 'w') as f:
        f.write(str(peaks))


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(current_dir, 'input')
    output_path = os.path.join(current_dir, 'output')

    sample_rate, data = wavfile.read(os.path.join(input_path, 'oshinoko.wav'))

    denoised_savgol_filter(sample_rate, data, output_path)

    get_peaks(sample_rate, data, output_path)


if __name__ == '__main__':
    main()