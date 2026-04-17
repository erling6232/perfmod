import numpy as np
from scipy import ndimage
from scipy.optimize import curve_fit
from collections import defaultdict
from imagedata import Series
from .fit_curve_voxels import fit_curve_voxels
from .models.gctt import make_gctt
from .models.tofts import make_tofts, make_extended_tofts
from .models.annet import make_annet
from .models.sourbron import make_sourbron_conv
# from .C_fitted import make_C_fitted, make_C_fitted_delay_minus_y_T1, make_C_fitted_delay_T1
from .aif.parker import parker
from .aif.find_delay import find_delay

import matplotlib.pyplot as plt


def show(signal, x=None, ax=None, show=False, label=None, color=None):
    if ax is None:
        fig, ax = plt.subplots(1)
        show = True
    if len(signal):
        if issubclass(type(signal), Series):
            signal = np.array(signal)
        if x is None:
            ax.plot(signal, label=label, color=color)
        else:
            ax.plot(x, signal, label=label, color=color)
        ax.grid(True)
        if label is not None:
            ax.legend(loc='upper right')
    if show:
        plt.show()


def fit_curves(im: np.ndarray | Series,
               method: str,
               aif_mask: Series = None,
               roi_mask: Series = None,
               h: np.ndarray = None,
               timeline_data: np.ndarray = None,
               im_aif: np.ndarray | Series = None,
               timeline_aif: np.ndarray = None,
               prmin: dict = None,
               b0in: dict = None,
               timeline: np.ndarray = None,
               hematocrit: float = 0.40,
               voxel_volume: float = None) -> dict:
    """Calculate perfusion curves from MR images.
    """

    assert im is not None, 'No image provided'
    assert method is not None, 'No method provided'
    assert hematocrit >= 0 and hematocrit <= 1, 'Hematocrit must be between 0 and 1'
    if issubclass(type(im), Series):
        assert im.axes[0].name == 'time', 'Image must be time series'

    # input variables - default values
    if prmin is None:
        prmin = {}
    prmin = {
                'optmethod': 'curve_fit',
                'aif_method': 'individual',
                'aif_normalization_method': 'max',
                'average_aif': None,
                'meanc': False,
                'vis': False,
            } | prmin
    # initial parameters
    if b0in is None:
        b0in = {}

    if im.ndim > 1:
        if aif_mask is not None:
            assert aif_mask.ndim <= im.ndim, "Wrong AIF ROI dimension {}, expected {}".format(
                aif_mask.ndim, im.ndim)
            aif_mask = aif_mask > 0
        if roi_mask is not None:
            assert roi_mask.ndim <= im.ndim, "Wrong tissue ROI dimension {}, expected {}".format(
                roi_mask.ndim, im.ndim)
            roi_mask = roi_mask > 0

        # aif image
        if im_aif is None:
            im_aif = im

    if timeline_data is None:
        timeline_data = im.timeline
    if timeline_aif is None:
        timeline_aif = im_aif.timeline
    if timeline is not None:
        timeline_data = timeline
        timeline_aif = timeline
    else:
        timeline = timeline_data
    assert np.all(timeline == timeline_aif), "Timelines do not match"

    # print('Parameters in fit_curves:\n{}'.format(prmin))
    if h is None:
        try:
            h = im.spacing
        except AttributeError:
            h = (1, 1, 1)
    assert len(h) == 3, "Wrong image spacing {}".format(h)

    # reduce size?
    # if prmin['red']
    #    [im,h] = redimagesize(im,h,'linear');
    #    [im_aif,h] = redimagesize(im_aif,h,'linear');
    #    dim = size(im);
    #    dim3 = dim(1:3);
    #    [roi_mask,h] = redimagesize(roi_mask,h,'nearest');
    #    [aif_mask,h] = redimagesize(aif_mask,h,'nearest');
    # end

    resample_data = False
    resample_aif = False
    # Use user-defined, fixed spacing of timeline
    if timeline is not None:
        if not np.all(timeline == timeline_data):
            resample_data = True
        if not np.all(timeline == timeline_aif):
            resample_aif = True

    if resample_data:
        print('Resampling image')
        raise ValueError('Resampling image not implemented')
        # interpmethod = 'linear'
        # [x{1},x{2},x{3},x{4}] = ndgrid(1:dim(1),1:dim(2),1:dim(3),timeline_data);
        # [xi{1},xi{2},xi{3},xi{4}] = ndgrid(1:dim(1),1:dim(2),1:dim(3),timeline);
        # im = interpn(x{1},x{2},x{3},x{4},im,xi{1},xi{2},xi{3},xi{4},interpmethod,0);

    if resample_aif:
        print('Resampling AIF image')
        raise ValueError('Resampling AIF image not implemented')
        # interpmethod = 'linear'
        # [x{1},x{2},x{3},x{4}] = ndgrid(1:dim(1),1:dim(2),1:dim(3),timeline_aif);
        # [xi{1},xi{2},xi{3},xi{4}] = ndgrid(1:dim(1),1:dim(2),1:dim(3),timeline);
        # im_aif = interpn(x{1},x{2},x{3},x{4},im_aif,xi{1},xi{2},xi{3},xi{4},interpmethod,0);

    # show4D(im,im_aif,aif_mask,roi_mask)
    # im_aif = im2vec4D(im_aif,aif_mask)
    if im_aif.ndim > 1:
        aif_value = np.mean(im_aif, axis=(1, 2, 3), where=aif_mask > 0)
    else:
        aif_value = im_aif

    if len(timeline) % 2 == 0:
        # Ensure odd length
        _dt = timeline[-1] - timeline[-2]
        timeline = np.append(timeline, [timeline[-1] + _dt])
        timeline_aif = np.append(timeline_aif, [timeline_aif[-1] + _dt])
        _dy = aif_value[-1] - aif_value[-2]
        aif_value = np.append(aif_value, [aif_value[-1] + _dy])

        if prmin['average_aif'] is not None and issubclass(type(prmin['average_aif']), np.ndarray):
            _dy = prmin['average_aif'][-1] - prmin['average_aif'][-2]
            prmin['average_aif'] = np.append(prmin['average_aif'], [prmin['average_aif'][-1] + _dy])

    aif_model = aif_value
    match prmin['aif_method']:
        case 'gammafunc':
            try:
                raise ValueError('gammafunc not implemented')
                # beta0 = [3, 1.5, 70, 20, 10]
                # timeline_fit = timeline - min(timeline)
                # beta = nlinfit(timeline_fit,aif_value,@kidney.gammafunc,beta0)
                # y = kidney.gammafunc(beta, timeline_fit)
                # plot(timeline,y);hold on;plot(timeline,aif_value,'-r');hold off
                aif_value = y
            except Exception as e:
                raise ValueError('WARNING: Could not generate a gamma variate: {}'.format(e))
        case 'parker':
            aif_model = parker(prmin['parker_parameters'], len(timeline_aif), timeline)  #  / 60)
            print('Normalization method for Parker: {}'.format(prmin['aif_normalization_method']))
        case 'average':
            aif_model = prmin['average_aif']
            print('Normalization method for average AIF: {}'.format(prmin['aif_normalization_method']))
        case 'individual':
            aif_model = aif_value
        case '_':
            raise ValueError('Unknown AIF method: {}'.format(prmin['aif_method']))
    # if prm['aif_method'] in ['parker', 'average']:
    aif_matched, norm_coeff = normalize_aif(aif_model, aif_value, prmin, timeline_aif)
    # print('Normalization coefficient: {}'.format(norm_coeff))
    if prmin['aif_normalization_method'] == 'auc_unity':
        aif_matched = aif_matched

    if im.ndim > 1:
        img = np.mean(im, axis=(1, 2, 3), where=roi_mask > 0)
    else:
        img = im
    _dy = img[-1] - img[-2]
    while len(img) < len(timeline):
        # Ensure same (odd) length as aif
        # img = np.append([0], img)
        img = np.append(img, [img[-1] + _dy])

    if voxel_volume is None:
        voxel_volume = np.prod(h)
    if roi_mask is not None:
        volume = voxel_volume * np.count_nonzero(roi_mask)
    else:
        volume = voxel_volume

    print('Using method {}'.format(method))
    methods = defaultdict(lambda *args: lambda *a: 'Invalid method', {
        'annet': make_annet,
        'sourbron': make_sourbron_conv,
        # 'patlak': make_patlak,
        'gctt': make_gctt,  # make_gctt_delay,  # make_C_fitted_delay_T1,  # make_C_fitted_delay_minus_y_T1,
        'tm': make_tofts,
        'etm': make_extended_tofts,
    })
    prmin['method'] = method
    print(f'fit_curves: b0in={b0in}')
    fun, b0, prm_model = methods[method](aif_matched, b0in, prmin)
    # print(f'fit_curves: b0={b0}')

    # prm_model = {'vis': prm['vis']}
    out = {'handle': [],
           'norm_coeff': norm_coeff,
           'volume': volume,
           'parameters': prm_model['parameters'],
           'units': prm_model['units'],
           }

    if prmin['vis']:
        fig, ax = plt.subplots(2, 2)
        # curve = np.mean(img, axis=(1, 2, 3), where=roi_mask == 1)
        show(img, ax=ax[0, 0], show=False, label='tissue data')
        show(aif_value*norm_coeff, ax=ax[0, 1], show=False, label='aif data')
        # curve = np.mean(img, axis=(1, 2, 3), where=aif_mask == 1)
        show(aif_model*norm_coeff, ax=ax[0, 1], show=False, label='aif model')
        # show(curve, ax=ax[0, 1], show=False, label='aif roi')
        show(aif_matched, ax=ax[1, 0], show=False, label='aif matched')
        # b0tt = [b0[_] for _ in prm_model['parameters']]
        # b0tt = [b0['F'], b0['E'], b0['ve'], b0['Tc'], b0['alphainv'], b0['delay']]
        # curve = fun(timeline, *b0tt)
        # print('curve: ', curve.shape)
        # show(curve, ax=ax[1, 1], show=False, label=method)

    # initialize by the Patlak model

    print('Fitting curves...')
    out = out | fit_curve_voxels(fun,
        img, aif_matched, timeline, prm_model['meanc'], volume, hematocrit, b0=b0, prm=prm_model
    )

    if prmin['vis']:
        show(img, ax=ax[1, 1], show=False, label='Tissue')
        show(out['fitted'], ax=ax[1, 1], show=True, label='Fitted')
    out['fun'] = fun
    out['timeline'] = timeline
    out['h'] = h
    out['aif'] = aif_matched
    return out

# prm.reduce image size
# function [imnew,hnew] = redimagesize(im,h,method)
#
#    dim = size(im);
#    if numel(dim) == 3
#        dim = [dim 1];
#    end;
#    dimnew = dim;
#
#    % factor to reduce with
#    p = 2;
#    dimnew(1:3) = round(dimnew(1:3)/p);
#    imnew = zeros(dimnew);
#    ntime = dim(4);
#    for i = 1 : ntime
#        imhere = im(:,:,:,i);
#        imnew(:,:,:,i) = imresize3d(imhere,dimnew,method);
#    end;
#    hnew = h*p;


def normalize_aif(model: Series | np.ndarray,
                  value: Series | np.ndarray,
                  prmin: dict,
                  timeline: np.ndarray) -> tuple[Series | np.ndarray, float]:
    normalization_method = prmin['aif_normalization_method']
    match normalization_method:
        case 'none':
            norm_coeff = 1.0
        case 'auc':
            norm_coeff = np.sum(value) / np.sum(model)
        case 'parker':
            _parker = parker(prmin['parker_parameters'], len(timeline), timeline)
            _norm = np.sum(value) / np.sum(_parker)
            model = value * _norm
            auc_diff = np.sum(value) / np.sum(model)
            # value = value * auc_diff
            norm_coeff = 1.0 / np.max(model)
            # model = value * norm_coeff
            delay = find_delay(value, model)
            matched = ndimage.shift(model, delay, mode='nearest')
            return matched, norm_coeff
        case 'max':
            norm_coeff = np.max(value) / np.max(model)
        case 'unity':
            norm_coeff = 1.0 / np.max(model)
        case '_':
            raise ValueError('Unknown normalization method: {}'.format(normalization_method))
    model = model * norm_coeff
    return model, norm_coeff
    # delay = find_delay(value, model)
    # matched = ndimage.shift(model, delay, mode='nearest')
    # return matched, norm_coeff


def smooth_reference(x, y, p0=None):
    def func(x, a, b, c):
        return a + b * x + c * x * x

    if p0 is None:
        p0 = (1, 1, 1)
    # print('smooth_reference: x={}, y={}'.format(x.shape, y.shape))
    par, pcov = curve_fit(func, x, y, p0=p0)
    # print('Reference curve: {}'.format(par))
    a, b, c = par
    return a + b * x + c * x * x
