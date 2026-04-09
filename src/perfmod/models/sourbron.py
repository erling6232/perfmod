import numpy as np
from . import default_parameters


def make_sourbron_conv(aif_value, b0in, prmin):
    # Construction to pass aif_value to function
    # def myfun_sourbron_conv(x, vp, tp, ft, tt):
    def myfun_sourbron_conv(t, *b):
        # NB must have equal time sampling

        vp, tp, ft, tt = b

        dt = np.empty_like(t, dtype=np.float32)
        dt[1:] = t[1:] - t[:-1]
        dt[0] = dt[1]

        f2 = dt * np.exp(-t / tp)
        cp = np.convolve(aif_value, f2 / tp)[:t.size]

        f1 = dt * np.exp(-t / tt)
        ck = ft * np.convolve(f1, cp)[:t.size]
        c = vp * cp + ck
        return c

    # Set defaults
    prm = default_parameters()
    prm['x_scale'] = None
    prm['lower_bounds'] = np.array([0, 0, 0, 0])
    prm['upper_bounds'] = np.array([1, 50, 0.5, 600])
    prm['parameters'] = ['vp', 'Tp', 'Ft', 'Tt']
    prm['units'] = {'vp': None, 'Tp': 't', 'Ft': 'ml/ml/t', 'Tt': 't'}
    prm['Cp'] = True
    prm['hematocrit'] = 0.45
    # Apply user-provided parameters
    prm = prm | prmin

    # account for hematocrit
    # Sourbron 2013, eqs 37-40
    if prm['Cp']:
        aif_value = aif_value / (1 - prm['hematocrit'])

    # optimization parameters, initialization
    b0 = {'vp': 0.15, 'Tp': 4.5, 'Ft': 0.0044, 'Tt': 30} | b0in

    return myfun_sourbron_conv, b0, prm