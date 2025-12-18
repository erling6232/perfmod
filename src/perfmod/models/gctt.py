import numpy as np
from scipy import ndimage
import scipy.special as sc
from . import default_parameters


def make_gctt(aif_value, b0in, prmin) -> tuple[callable, dict, dict]:
    # def myfun_gctt(x, F, E, ve, Tc, alphainv, delay)

    def gctt(t, *b):
        """h = GCTT(t, b), where b = [F E ve Tc alphainv]
        Model based on
        Schabel, M. C. (2012). A unified impulse response model for DCE-MRI. Magnetic Resonance in Medicine, 68(5), 1632�1646. http://doi.org/10.1002/mrm.24162

        # F=b[0]: plasma flow
        # E=b[1]: Extraction fraction
        # ve=b[2]: EES volume
        # Tc=b[3]: mean of transit times (not sensitive - is multiplied by 10 to increase influence)
        # alphainv=b[4]: std.deviation of transit times
        """

        F, E, ve, Tc, alphainv = b

        alpha = (1 / alphainv)
        tau = Tc / alpha
        kep = (E * F) / ve
        if (1 / tau - kep) < 0:
            raise ValueError("gammaincc(.., 1/tau - kep) is negative")

        Rv = sc.gammaincc(alpha, t / tau)
        Rp = (E * np.exp(-kep * t)) / ((1 - kep * tau) ** alpha) * \
             (1 - sc.gammaincc(alpha, (1 / tau - kep) * t))

        R = Rv + Rp
        h = F * np.convolve(R, aif_value)[:t.size]

        # Hv = np.convolve(Hv * F, aif_value)  # vascular part of IRF
        # Hp = np.convolve(Hp * F, aif_value)  # parenchymal part of IRF

        return h

    # Set defaults
    prm = default_parameters()
    # prm['x_scale'] = np.array([0.1, 1, 1, 1, 1, 1])  # [10 1 10 1 10];
    prm['x_scale'] = None
    prm['lower_bounds'] = np.array([0, 0, 1e-9, 0, 0])
    prm['upper_bounds'] = np.array([np.inf, 1, np.inf, np.inf, 1])

    # optimization parameters, initialization
    b0 = {'F': 0.1, 'E': 0.2, 've': 0.2, 'Tc': 10./60, 'alphainv': 0.6}
    prm['parameters'] = ['F', 'E', 've', 'Tc', 'alphainv']
    prm['units'] = {'F': 'ml/ml/min', 'E': None, 've': 'ml/ml/min', 'Tc': 'min',
                    'alphainv': None}

    # Apply user-provided parameters
    prm = prm | prmin
    b0 = b0 | b0in

    return gctt, b0, prm


def make_gctt_delay(aif_value, b0in, prmin) -> tuple[callable, dict, dict]:
    # def myfun_gctt(x, F, E, ve, Tc, alphainv, delay)

    def gctt(t, *b):
        """h = GCTT(t, b), where b = [F E ve Tc alphainv delay]
        Model based on
        Schabel, M. C. (2012). A unified impulse response model for DCE-MRI. Magnetic Resonance in Medicine, 68(5), 1632�1646. http://doi.org/10.1002/mrm.24162

        # F=b[0]: plasma flow
        # E=b[1]: Extraction fraction
        # ve=b[2]: EES volume
        # Tc=b[3]: mean of transit times (not sensitive - is multiplied by 10 to increase influence)
        # alphainv=b[4]: std.deviation of transit times
        # delay=b[5]: delay in AIF arrival
        """

        F, E, ve, Tc, alphainv, delay = b
        # t = t.copy() + delay
        dt = t[2] - t[1]
        aif_shifted = ndimage.shift(aif_value, delay / dt, mode='nearest')
        print('gctt: F {} E {} ve {} Tc {} a-1 {} td {}'.format(F, E, ve, Tc, alphainv, delay))  # , end='')

        alpha = 1 / alphainv
        tau = Tc / alpha
        kep = (E * F) / ve
        if (1 / tau - kep) < 0:
            raise ValueError("gammaincc(.., 1/tau - kep) is negative")

        Rv = sc.gammaincc(alpha, t / tau)
        Rp = (E * np.exp(-kep * t)) / ((1 - kep * tau) ** alpha) * \
             (1 - sc.gammaincc(alpha, (1 / tau - kep) * t))

        R = Rv + Rp
        h = F * np.convolve(R, aif_shifted)[:t.size]

        # Hv = np.convolve(Rv * F, aif_value)  # vascular part of IRF
        # Hp = np.convolve(Rp * F, aif_value)  # parenchymal part of IRF
        # print('gctt: {} {}'.format(h, b))
        return h

    # Set defaults
    # F, E, ve, Tc, alphainv, delay
    prm = default_parameters()
    # prm['x_scale'] = np.array([0.1, 1, 1, 1, 1, 1])  # [10 1 10 1 10];
    prm['x_scale'] = None
    prm['lower_bounds'] = np.array([0, 0, 1e-9, 0, 0, -1.5])  # -10])
    prm['upper_bounds'] = np.array([np.inf, 1, np.inf, np.inf, 1, 1.5])  # 30])
    # info.perfan.params = {'F [ml/(ml min)]', 'E [-]', 've [ml/ml]', 'Tc [min]', 'alphainv [-]', 'BAT [min]'};

    # optimization parameters, initialization
    # F [ml/(ml min)], E [-], ve [ml/ml], Tc [min], alphainv [-], BAT [min]};
    b0 = {'F': 0.1, 'E': 0.2, 've': 0.2, 'Tc': 10./60, 'alphainv': 0.6, 'delay': 0.}
    prm['parameters'] = ['F', 'E', 've', 'Tc', 'alphainv', 'delay']
    prm['units'] = {'F': 'ml/ml/min', 'E': None, 've': 'ml/ml/min', 'Tc': 'min',
                    'alphainv': None, 'delay': 'min'}

    # Apply user-provided parameters
    prm = prm | prmin
    b0 = b0 | b0in

    return gctt, b0, prm
