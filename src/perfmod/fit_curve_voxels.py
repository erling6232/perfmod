import os.path
import numpy as np
from scipy.optimize import curve_fit
from .myfun import make_sourbron_conv, make_sourbron_numint, make_sourbron_matrix, make_sourbron_loop


def fit_curve_voxel_sourbron(im, aif_value, timeline, meanc, volume, hct, b0in=None, prmin=None):
    """MRI-measurement of perfusion and glomerular filtration in the human
    kidney with a separable compartment model. Sourbron SP, Michaely HJ,
    Reiser MF, Schoenberg SO. Invest Radiol. 2008 Jan;43(1):40-8.
    """

    prm = {'vis': False,
           'savefig': False,
           'intmethod': 'conv',
           # 'intmethod': 'matrix',
           # 'intmethod': 'numint',
           'loss': 'linear',
           'f_scale': 1.0,
           }
    # handlefig = [];

    # optimization parameters, initialization
    b0 = {'vp': 0.15, 'tp': 4.5, 'ft': 0.0044, 'tt': 30}

    # input parameters
    if b0in is not None:
        for key in b0in.keys():
            b0[key] = b0in[key]

    # input parameters
    if prmin is not None:
        for key in prmin.keys():
            prm[key] = prmin[key]

    # account for hematocrit
    # Sourbron 2013, eqs 37-40
    aif = aif_value / (1 - hct)

    ntime = im.shape[0]
    if meanc:
        im = np.mean(im, axis=(1, 2, 3)).reshape((ntime, 1))
        volume = np.sum(volume)
    # dim = im.ndim
    if im.ndim > 1:
        nvox = im.shape[1]
    else:
        nvox = 1
    xdata = timeline
    handlefig = None

    print('Dimension optimization data: {} x {}'.format(nvox, ntime))

    # print to screen
    print('This is {}\n{}'.format(os.path.basename(__file__), prm))

    # lower and upper bound of the optimization parameters
    try:
        lb = prm['lower_bounds']
    except KeyError:
        lb = (0, 0, 0, 0)
    try:
        ub = prm['upper_bounds']
    except KeyError:
        ub = (1, 30, 0.020, 100)

    # Optimization method
    # options = optim_options('lsqcurvefit', 'MaxFunEvals', 1000,'Algorithm','trust-region-reflective')
    # options = {'Algorithm','trust-region-reflective'};
    # options.MaxFunEvals = 4000;
    if prm['intmethod'] == 'conv':
        # f = @(b,x)myfun_sourbron_conv(b,x,aif)
        fun = make_sourbron_conv
    elif prm['intmethod'] == 'numint':
        # f = @(b,x)myfun_sourbron_numint(b,x,aif)
        fun = make_sourbron_numint
    elif prm['intmethod'] == 'matrix':
        # f = @(b,x)myfun_sourbron_matrix(b,x,aif)
        fun = make_sourbron_matrix
    elif prm['intmethod'] == 'loop':
        # f = @(b,x)myfun_sourbron_loop(b,x,aif)
        fun = make_sourbron_loop
    else:
        raise ValueError('Unknown optimization method: {}'.format(prm['intmethod']))
    # initial values of model data and GFR
    # f = np.empty((ntime, nvox), dtype=np.float64)
    f = np.full((ntime, nvox), np.nan, dtype=np.float64)
    # f[:] = np.nan
    data = {
        'aif_value': aif,
        'xdata': xdata
    }

    b0in = [b0['vp'], b0['tp'], b0['ft'], b0['tt']]

    # prm['vis'] = True
    b = np.full((4, nvox), np.nan, dtype=np.float64)
    for i in range(nvox):
        print('Voxel {} out of {}'.format(i + 1, nvox))

        if im.ndim > 1:
            data['ydata'] = im[:, i]
        else:
            data['ydata'] = im[:]

        # boutls,resnorm,residls = lsqcurvefit(f,b0in,data['xdata'],data['ydata'],lb,ub,options);
        boutls, pcov = curve_fit(fun(aif), data['xdata'], data['ydata'], p0=b0in, bounds=(lb, ub),
                                 loss=prm['loss'], f_scale=prm['f_scale'])
        # perr = np.sqrt(np.diag(pcov))  # One standard deviation

        # nonlinear fit, gives completely another result, and highly varying!!!
        # boutnl = nlinfit(data.xdata,data.ydata,f,b0(i,:));

        # coefficients and function value
        b[:, i] = boutls

        print('Obtained parameters: {}'.format(boutls))

        # compute the response function
        compute = fun(aif)
        # if prm['intmethod'] == 'conv':
        #     compute = myfun_sourbron_conv(boutls, data['xdata'], aif)
        # elif prm['intmethod'] == 'numint':
        #     compute = myfun_sourbron_numint(boutls, data['xdata'], aif)
        # elif prm['intmethod'] == 'matrix':
        #     compute = myfun_sourbron_matrix(boutls, data['xdata'], aif)
        # elif prm['intmethod'] == 'loop':
        #     compute = myfun_sourbron_loop(boutls, data['xdata'], aif)
        # else:
        #     raise ValueError('Unknown optimization method: {}'.format(prm['intmethod']))

        # convert units
        # GFR comes out as: ml/s/ml
        # Want GFR as: ml/min/voxel

        # here in ml/min
        # boutls = ['vp', 'tp', 'ft', 'tt']
        gfr[i] = convert_gfr(boutls[2], volume)

        # function value
        f[:, i] = compute(data['xdata'], boutls[0], boutls[1], boutls[2], boutls[3])

        # new initial values
        b0in = np.nanmean(b[:, 0:i + 1], axis=1)

    # From mm³ to ml
    vol = volume * 1e-3

    out = {'vp': b[0], 'tp': b[1], 'ft': b[2], 'tt': b[3]}

    out['fitted'] = f
    # ROI volume
    out['roivol'] = np.sum(vol)
    out['roivolunit'] = 'ml'

    # Account for hematocrit
    # Sourbron 2013, eqs 37-40
    out['fp'] = np.divide(out['vp'], out['tp']).item()
    out['bv'] = (out['vp'] / (1 - hct)).item()
    out['rbf'] = out['fp'] / (1 - hct)

    # Filtration fraction
    out['FF'] = (out['ft'] / out['fp']).item()

    # Single kidney blood flow
    out['BF'] = out['rbf'] * out['roivol'] * 60
    out['BFunit'] = 'ml/min'

    # Units ml/min/100ml

    # Fb
    out['rbf'] = out['rbf'] * 60 * 100
    out['rbfunit'] = 'ml/min/100ml'

    # Fp
    out['fp'] = out['fp'] * 60 * 100
    out['fpunit'] = 'ml/min/100ml'

    if im.ndim > 1:
        out['res'] = np.nanmean(np.power((im - f), 2)).item()
    else:
        out['res'] = np.nanmean(np.power((im - f[:, 0]), 2)).item()
    out['handlefig'] = handlefig

    print('Total ROI volume (ml): {:0.1f}'.format(out['roivol']))
    print('GFR (ml/min): {:0.1f}'.format(out['gfr']))
    print('Plasma flow (ml/min/100ml): {:0.1f}'.format(out['fp']))
    print('RBF (ml/min/100ml): {:0.1f}'.format(out['rbf']))
    print('Blood flow (ml/min): {:0.1f}'.format(out['BF']))
    print('Filtration fraction: {:0.3f}'.format(out['FF']))
    return out
