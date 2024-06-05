import os
import librosa
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
from scipy.io import wavfile
from scipy.ndimage import maximum_filter

def make_spectrogram(samples, sample_rate, filepath):
    freq, t, spect = signal.spectrogram(samples, sample_rate, window=('hann'))

    spect = np.log10(spect + 1)
    plt.pcolormesh(t, freq, spect, shading='gouraud', vmin=spect.min(), vmax=spect.max())
    plt.ylabel('Частота (Гц)')
    plt.xlabel('Время (с)')

    plt.savefig(filepath)

    return freq, t, spect

def get_max_tembr(filepath):
    data, sample_rate = librosa.load(filepath)
    chroma = librosa.feature.chroma_stft(y=data, sr=sample_rate)
    f = librosa.piptrack(y=data, sr=sample_rate, S=chroma)[0]
    max_f = np.argmax(f)

    return max_f

def get_peaks(freq, t, spect):
    delta_t = int(len(t) * 0.1)
    delta_f = int(50 / (freq[1] - freq[0]))
    filtered = maximum_filter(spect, size=(delta_f, delta_t))

    peaks_mask = (spect == filtered)
    peak_values = spect[peaks_mask]
    peak_frequencies = freq[peaks_mask.any(axis=1)]

    top_indices = np.argsort(peak_values)[-3:]
    top_frequencies = peak_frequencies[top_indices]

    return list(top_frequencies // 1)

def get_min_max(filepath):
    y, sr = librosa.load(filepath, sr=None)
    d = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

    frequencies = librosa.fft_frequencies(sr=sr)
    mean_spec = np.mean(d, axis=1)

    idx_min = np.argmax(mean_spec > -80)
    idx_max = len(mean_spec) - np.argmax(mean_spec[::-1] > -80) - 1

    min_freq = frequencies[idx_min]
    max_freq = frequencies[idx_max]

    return max_freq, min_freq

def main():
    input_path = "lab10/input/"
    output_path = "lab10/output/"

    files = os.listdir(input_path)

    for file in files:
        rate, samples = wavfile.read(input_path + file)
        max_freq, min_freq = get_min_max(input_path + file)
        freq, t, spect = make_spectrogram(samples, rate, output_path + file[:-3] + 'png')

        print(f'{file}:')
        print('Max частота:', max_freq)
        print('Min частота:', min_freq)
        print('Наиболее тембрально окрашенный основной тон:', get_max_tembr(input_path + file))
        print('Три самые сильные форманты:', get_peaks(freq, t, spect))
        print('\n')

if __name__ == "__main__":
    main()