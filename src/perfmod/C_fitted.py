import numpy as np
from scipy.special import gammainc
from .aif.parker import parker
from .models.gctt import gctt, init_gctt


def default_parameters() -> dict:
    return {
        'red': False,
        'vis': False,
        'meanc': True,
        'collapse': False,
        'initializepatlak': False,
        'optmethod': 'lsqcurvefit',
        'gammafunc': False,
        'lower_bounds': (0, 1, 0, 0),
        'upper_bounds': (0.9, 50, 0.5, 300),
        'k21': 0.001,

        # Saving figure?
        'savefig': False,

        # Integration method
        # prm['intmethod'] = 'matrix';
        'intmethod': 'conv',
        # prm['intmethod'] = 'numint';
        # 'intmethod': 'loop',
        'loss': 'linear',
        'f_scale': 1.0,
    }



def make_C_fitted(aif_value, b0in, prmin):
    """Get a model according to the specified parameters.

    Construction to pass aif_value to function C_fitted.
    """
    def C_fitted(x, *b):
        """Perform convolution with AIF.
        """

        # tau, d, fa, k21, k12 = b
        Ts = x[1] - x[0]
        N = len(x)

        if nargin > 5
            b2 = b;
            b = zeros(1, length(x_fixed));
            b(~x_fixed) = b2;
            b(x_fixed) = x_start(x_fixed);
        end

        b = b / prm['rescale_parameters']

        delay = b[-1]
        h = prm['model'](x, b[:-1])
        # aif_shifted = getParker(aif, 1 / norm_coeff, delay, x)
        aif_shifted = (parker(prm['parker_parameters'], len(x), aif_value.timeline - delay)
                       / prm['norm_coeff'])

        #  aif_shifted = posun(aif, delay)
        # Cfit = konvf(h, aif_shifted)
        Cfit = np.convolve(h, aif_shifted)
        return Cfit

    # Set defaults
    prm = default_parameters()

    # optimization parameters, initialization
    b0 = {'vp': 0.15, 'tp': 4.5, 'ft': 0.0044, 'tt': 30}

    match prmin['method']:
        case 'gctt':
            prm['model'], b0, prm = init_gctt(b0, prm)

    # Apply user-provided parameters
    prm = prm | prmin
    b0 = b0 | b0in

    # return fun, b0, prm
    return C_fitted, b0, prm


def make_C_fitted_delay_T1(aif_value, b0in, prmin):
    """Get a model according to the specified parameters.

    Construction to pass aif_value to function C_fitted_delay_minus_y_T1.
    """
    def C_fitted_delay_T1(x, *b):
        """Perform convolution with AIF.
        """

        # tau, d, fa, k21, k12 = b
        Ts = x[1] - x[0]
        N = len(x)

        if nargin > 5
            b2 = b;
            b = zeros(1, length(x_fixed));
            b(~x_fixed) = b2;
            b(x_fixed) = x_start(x_fixed);
        end

        b = b / prm['rescale_parameters']

        delay = b[-1]
        h = prm['model'](x, b[:-1])
        # aif_shifted = getParker(aif, 1 / norm_coeff, delay, x)
        aif_shifted = (parker(prm['parker_parameters'], len(x), aif_value.timeline - delay)
                       / prm['norm_coeff'])

        #  aif_shifted = posun(aif, delay)
        # Cfit = konvf(h, aif_shifted)
        Cfit = np.convolve(h, aif_shifted)
        return Cfit

    # Set defaults
    prm = default_parameters()

    # optimization parameters, initialization
    b0 = {'vp': 0.15, 'tp': 4.5, 'ft': 0.0044, 'tt': 30}

    match prmin['method']:
        case 'gctt':
            prm['model'], b0, prm = init_gctt(b0, prm)

    # Apply user-provided parameters
    prm = prm | prmin
    b0 = b0 | b0in

    # return fun, b0, prm
    return C_fitted_delay_T1, b0, prm


def make_C_fitted_delay_minus_y_T1(aif_value, b0in, prmin):
    """Get a model according to the specified parameters.

    Construction to pass aif_value to function C_fitted_delay_minus_y_T1.
    """
    def C_fitted_delay_minus_y_T1(x, *b):
        """Perform convolution with AIF.
        """

        # tau, d, fa, k21, k12 = b
        Ts = x[1] - x[0]
        N = len(x)

        if nargin > 6
            b2 = b;
            b = zeros(1, length(x_fixed));
            b(~x_fixed) = b2;
            b(x_fixed) = x_start(x_fixed);
        end

        b = b / prm['rescale_parameters']

        delay = b[-1]
        b = b[:-1]
        h = prm['model'](x, b)
        # aif_shifted = getParker(aif, 1 / norm_coeff, delay, x)
        aif_shifted = (parker(prm['parker_parameters'], len(x), aif_value.timeline - delay)
                       / prm['norm_coeff'])

        #  aif_shifted = posun(aif, delay)
        # Cfit = konvf(h, aif_shifted)
        Cfit = np.convolve(h, aif_shifted)
        Cfit = Cfit - y
        Cfit = Cfit(skipped)
        return Cfit

    # Set defaults
    prm = default_parameters()

    # optimization parameters, initialization
    b0 = {'vp': 0.15, 'tp': 4.5, 'ft': 0.0044, 'tt': 30}

    match prmin['method']:
        case 'gctt':
            prm['model'], b0, prm = init_gctt(b0, prm)

    # Apply user-provided parameters
    prm = prm | prmin
    b0 = b0 | b0in

    # return fun, b0, prm
    return C_fitted_delay_minus_y_T1, b0, prm
