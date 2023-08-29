import numpy as np
from scipy import stats

def series_filter(values, kernel_size=3):

    filter_values = np.cumsum(values, dtype=float)

    filter_values[kernel_size:] = filter_values[kernel_size:] - filter_values[:-kernel_size]
    filter_values[kernel_size:] = filter_values[kernel_size:] / kernel_size

    for i in range(1, kernel_size):
        filter_values[i] /= i + 1

    return filter_values


def extrapolate_next(values):


    last_value = values[-1]
    slope = [(last_value - v) / i for (i, v) in enumerate(values[::-1])]
    slope[0] = 0
    next_values = last_value + np.cumsum(slope)

    return next_values


def marge_series(values, extend_num=5, forward=5):

    next_value = extrapolate_next(values)[forward]
    extension = [next_value] * extend_num

    if isinstance(values, list):
        marge_values = values + extension
    else:
        marge_values = np.append(values, extension)
    return marge_values


class Silency(object):
    def __init__(self, amp_window_size, series_window_size, score_window_size):
        self.amp_window_size = amp_window_size
        self.series_window_size = series_window_size
        self.score_window_size = score_window_size

    def transform_silency_map(self, values):


        freq = np.fft.fft(values)
        mag = np.sqrt(freq.real ** 2 + freq.imag ** 2)
        spectral_residual = np.exp(np.log(mag) - series_filter(np.log(mag), self.amp_window_size))

        freq.real = freq.real * spectral_residual / mag
        freq.imag = freq.imag * spectral_residual / mag

        silency_map = np.fft.ifft(freq)
        return silency_map

    def transform_spectral_residual(self, values):
        silency_map = self.transform_silency_map(values)
        spectral_residual = np.sqrt(silency_map.real ** 2 + silency_map.imag ** 2)
        return spectral_residual

    def generate_anomaly_score(self, values, type="avg"):


        extended_series = marge_series(values, self.series_window_size, self.series_window_size)
        mag = self.transform_spectral_residual(extended_series)[: len(values)]

        if type == "avg":
            ave_filter = series_filter(mag, self.score_window_size)
            score = (mag - ave_filter) / ave_filter
        elif type == "abs":
            ave_filter = series_filter(mag, self.score_window_size)
            score = np.abs(mag - ave_filter) / ave_filter
        elif type == "chisq":
            score = stats.chi2.cdf((mag - np.mean(mag)) ** 2 / np.var(mag), df=1)
        else:
            raise ValueError("No type!")
        return score