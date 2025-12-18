import numpy as np
from . import default_parameters


def make_annet(aif_value, b0in, prmin) -> tuple[callable, dict, dict]:
    # Construction to pass aif_value to function

    def annet(x, *b):

        # b = [tau d fa k21 k12]
        # tau = b(1);
        # d   = b(2);
        # fa  = b(3);
        # k21 = b(4);
        # k12 = b(5);
        tau, d, fa, k21, k12 = b

        # ntime = numel(x);
        # ntime = np.prod(x.size).item()

        # delta t
        # dt = (x([2:end end]) - x([1 1:end-1]))/2;
        dt = np.empty_like(x, dtype=np.float32)
        dt[1:] = x[1:] - x[:-1]
        # dt(1) = dt(2);
        dt[0] = dt[1]
        # dt(end) = dt(end-1);

        m = np.mean(aif_value)
        # f1 = interp1(x,aif_value,x + tau,'linear',m);
        f1 = np.interp(x + tau, x, aif_value, left=m, right=m)
        # f1 = aif_value;
        f2 = dt * np.exp(-x / d)
        caprime = np.convolve(f1, f2 / d)[:x.size]

        # hematocrit adjustment
        # From Annet, page 845
        # caprime = caprime/(1 - 0.55);

        f1 = dt * np.exp(-k12 * x)
        f2 = caprime
        ck = k21 * np.convolve(f1, f2)[:x.size]
        c = fa * caprime + ck

        return c

    # Set defaults
    # tau, d, fa, k21, k12
    prm = default_parameters()
    # prm['x_scale'] = np.array([0.1, 1, 1, 1, 1, 1])  # [10 1 10 1 10];
    prm['x_scale'] = None
    prm['lower_bounds'] = np.array([-np.inf, 0, 0, 0, 0])  # -10])
    prm['upper_bounds'] = np.array([np.inf, np.inf, 1, np.inf, np.inf])  # 30])

    # optimization parameters, initialization
    b0 = {'tau': 0.1, 'd': 0.1, 'fa': 0.1, 'k21': 0.1, 'k12': 0.9}
    prm['parameters'] = ['tau', 'd', 'fa', 'k21', 'k12']
    prm['units'] = {'tau': 'min', 'd': None, 'fa': None, 'k21': 'mL/min',
                    'k12': 'mL/min'};

    # Apply user-provided parameters
    prm = prm | prmin
    b0 = b0 | b0in

    return annet, b0, prm

