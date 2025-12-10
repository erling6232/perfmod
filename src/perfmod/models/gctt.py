import numpy as np
from scipy.special import gammainc
from . import default_parameters


def make_gctt(aif_value, b0in, prmin) -> tuple[callable, dict, dict]:
    # def myfun_gctt(x, F, E, ve, Tc, alphainv, delay)

    def gctt(x, *b):
        """h = GCTT(x, b), where b = [F E ve Tc alfainv]
        Model based on
        Schabel, M. C. (2012). A unified impulse response model for DCE-MRI. Magnetic Resonance in Medicine, 68(5), 1632�1646. http://doi.org/10.1002/mrm.24162

        # F=b[0]: plasma flow
        # E=b[1]: Extraction fraction
        # ve=b[2]: EES volume
        # Tc=b[3]: mean of transit times (not sensitive - is multiplied by 10 to increase influence)
        # alfainv=b[4]: std.deviation of transit times
        """

        F, E, ve, Tc, alfainv = b

        Ts = x[2] - x[1]  # Skip first time point

        alfa = (1 / alfainv)
        tau = Tc / alfa
        kep = (E * F) / ve

        Hv = 1 - gammainc(alfa, x / tau)
        Hp = ((E * np.exp(-kep * x)) / (pow((1 - kep * tau), alfa))) * \
              (1 - (1 - gammainc(alfa, (1 / tau - kep) * x)))

        h = Hv + Hp
        h = h * F * Ts

        Hv = Hv * F * Ts  # vascular part of IRF
        Hp = Hp * F * Ts  # parenchymal part of IRF
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
