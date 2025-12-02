import numpy as np
from scipy import signal


def find_delay(ref, test):
    correlation = signal.correlate(ref, test, mode='full')
    lags = signal.correlation_lags(ref.size, test.size, mode='full')
    lag = lags[np.argmax(correlation)]
    return lag