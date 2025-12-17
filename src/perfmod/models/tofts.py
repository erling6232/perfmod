import numpy as np
from scipy import ndimage
from . import default_parameters


def make_tofts_delay(aif_value, b0in, prmin) -> tuple[callable, dict, dict]:

    def tofts(t, *b):
        """h = tofts(t, b), where b = [ktrans ve delay]
        Model based on
        ...

        # ktrans=b[0]: ktrans
        # ve=b[1]: EES volume
        # delay=b[2]: delay in AIF arrival
        """

        ktrans, ve, delay = b
        dt = t[2] - t[1]
        aif_shifted = ndimage.shift(aif_value, delay / dt, mode='nearest')
        print('tofts: ktrans {} ve {} td {}'.format(ktrans, ve, delay))

        # kep = ktrans/ve
        h = ktrans * np.exp(-ktrans/ve * t)
        h = np.convolve(h, aif_shifted)[:t.size]
        return h

    # Set defaults
    # ktrans, ve, delay
    prm = default_parameters()
    # prm['x_scale'] = np.array([0.1, 1, 1, 1, 1, 1])  # [10 1 10 1 10];
    prm['x_scale'] = None
    prm['lower_bounds'] = np.array([0, 0, -1.5])
    prm['upper_bounds'] = np.array([np.inf, np.inf, 1.5])

    # optimization parameters, initialization
    b0 = {'ktrans': 0.2, 've': 0.2, 'delay': 0.}
    prm['parameters'] = ['ktrans', 've', 'delay']
    prm['units'] = {'ktrans': 'ml/ml/min', 've': 'ml/ml/min',
                    'delay': 'min'};

    # Apply user-provided parameters
    prm = prm | prmin
    b0 = b0 | b0in

    return tofts, b0, prm
