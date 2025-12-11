import numpy as np
import scipy.special as sc
from . import default_parameters


def make_gctt(aif_value, b0in, prmin) -> tuple[callable, dict, dict]:
    # def myfun_gctt(x, F, E, ve, Tc, alphainv, delay)

    def gctt(t, *b):
        """h = GCTT(t, b), where b = [F E ve Tc alfainv]
        Model based on
        Schabel, M. C. (2012). A unified impulse response model for DCE-MRI. Magnetic Resonance in Medicine, 68(5), 1632�1646. http://doi.org/10.1002/mrm.24162

        # F=b[0]: plasma flow
        # E=b[1]: Extraction fraction
        # ve=b[2]: EES volume
        # Tc=b[3]: mean of transit times (not sensitive - is multiplied by 10 to increase influence)
        # alfainv=b[4]: std.deviation of transit times
        """

        F, E, ve, Tc, alfainv = b
        t = t.copy() / 60.

        alfa = (1 / alfainv)
        tau = Tc / alfa
        kep = (E * F) / ve
        assert (1 / tau - kep) >= 0, "gammaincc(.., 1/tau - kep) is negative"

        Hv = sc.gammaincc(alfa, t / tau)
        Hp = ((E * np.exp(-kep * t)) / (pow((1 - kep * tau), alfa))) * \
              (1 - sc.gammaincc(alfa, (1 / tau - kep) * t))

        h = Hv + Hp
        h = np.convolve(h * F, aif_value, mode='same')

        # Hv = np.convolve(Hv * F, aif_value, mode='same')  # vascular part of IRF
        # Hp = np.convolve(Hp * F, aif_value, mode='same')  # parenchymal part of IRF

        return h

    # Set defaults
    prm = default_parameters()
    prm['x_fixed'] = np.array([False, False, False, False, False], dtype=bool)  # if "1", parameter is fixed on its starting value
    prm['x_start'] = np.array([
        [1.0, 0.5, 0.5, 0.10, 0.2],
        [1.0, 0.5, 0.5, 0.25, 0.2],
        [1.0, 0.5, 0.5, 0.50, 0.2],
        [1.0, 0.1, 0.5, 1.10, 0.2],
        [1.0, 1.0, 0.5, 1.25, 0.2]
    ])  # time - depend.parameters in minutes
    prm['rescale_parameters'] = np.array([0.1, 1, 1, 1, 1])  # [10 1 10 1 10];
    prm['lower_bounds'] = np.array([0, 0, 1e-9, 0, 0])
    prm['upper_bounds'] = np.array([np.inf, 1, np.inf, np.inf, 1])
    # info.perfan.params = {'F [ml/(ml min)]', 'E [-]', 've [ml/ml]', 'Tc [min]', 'alphainv [-]', 'BAT [min]'};

    # optimization parameters, initialization
    b0 = {'F': 0.1, 'E': 0.2, 've': 0.2, 'Tc': 10., 'alfainv': 0.6}

    # Apply user-provided parameters
    prm = prm | prmin
    b0 = b0 | b0in

    return gctt, b0, prm


def make_gctt_delay(aif_value, b0in, prmin) -> tuple[callable, dict, dict]:
    # def myfun_gctt(x, F, E, ve, Tc, alphainv, delay)

    def gctt(t, *b):
        """h = GCTT(t, b), where b = [F E ve Tc alfainv delay]
        Model based on
        Schabel, M. C. (2012). A unified impulse response model for DCE-MRI. Magnetic Resonance in Medicine, 68(5), 1632�1646. http://doi.org/10.1002/mrm.24162

        # F=b[0]: plasma flow
        # E=b[1]: Extraction fraction
        # ve=b[2]: EES volume
        # Tc=b[3]: mean of transit times (not sensitive - is multiplied by 10 to increase influence)
        # alfainv=b[4]: std.deviation of transit times
        # delay=b[5]: delay in AIF arrival
        """

        F, E, ve, Tc, alfainv, delay = b
        t = t.copy() / 60. + delay

        alfa = (1 / alfainv)
        tau = Tc / alfa
        kep = (E * F) / ve
        assert (1 / tau - kep) >= 0, "gammaincc(.., 1/tau - kep) is negative"

        Hv = sc.gammaincc(alfa, t / tau)
        Hp = ((E * np.exp(-kep * t)) / (pow((1 - kep * tau), alfa))) * \
             (1 - sc.gammaincc(alfa, (1 / tau - kep) * t))

        h = Hv + Hp
        h = np.convolve(h * F, aif_value, mode='same')

        # Hv = np.convolve(Hv * F, aif_value, mode='same')  # vascular part of IRF
        # Hp = np.convolve(Hp * F, aif_value, mode='same')  # parenchymal part of IRF
        # print('gctt: {} {}'.format(h, b))
        return h

    # Set defaults
    prm = default_parameters()
    prm['x_fixed'] = np.array([False, False, False, False, False], dtype=bool)  # if "1", parameter is fixed on its starting value
    prm['x_start'] = np.array([
        [1.0, 0.5, 0.5, 0.10, 0.2],
        [1.0, 0.5, 0.5, 0.25, 0.2],
        [1.0, 0.5, 0.5, 0.50, 0.2],
        [1.0, 0.1, 0.5, 1.10, 0.2],
        [1.0, 1.0, 0.5, 1.25, 0.2]
    ])  # time - depend.parameters in minutes
    prm['rescale_parameters'] = np.array([0.1, 1, 1, 1, 1, 1])  # [10 1 10 1 10];
    prm['lower_bounds'] = np.array([0, 0, 1e-9, 0, 0, -np.inf])  # -10])
    prm['upper_bounds'] = np.array([np.inf, 1, np.inf, np.inf, 1, np.inf])  # 30])
    # info.perfan.params = {'F [ml/(ml min)]', 'E [-]', 've [ml/ml]', 'Tc [min]', 'alphainv [-]', 'BAT [min]'};

    # optimization parameters, initialization
    b0 = {'F': 0.1, 'E': 0.2, 've': 0.2, 'Tc': 10., 'alfainv': 0.6, 'delay': 0.}

    # Apply user-provided parameters
    prm = prm | prmin
    b0 = b0 | b0in

    return gctt, b0, prm
