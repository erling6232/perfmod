import numpy as np
from scipy.optimize import curve_fit
from .fit_curve_voxels import fit_curve_voxel_sourbron
from .aif import parker, find_delay


def fit_curves(im, method, aif_mask, roi_mask, h=None, timeline_data=None,
               im_aif=None, timeline_aif=None, prmin=None, b0in=None, timeline=None, hematocrit=0.40,
               voxel_volume=None):
    """Find GFR of kidney data
    """

    # prm['dtimeline'] = 3;
    prm = {'red': False,
           'meanc': True,
           'collapse': False,
           'initializepatlak': False,
           'optmethod': 'lsqcurvefit',
           'gammafunc': False,
           'lower_bounds': (0, 1, 0, 0),
           'upper_bounds': (0.9, 50, 0.5, 300),

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

    # input variables
    # prm = mergeinputpar(prm,prmin);
    if prmin is not None:
        for key in prmin.keys():
            prm[key] = prmin[key]

    if im.ndim > 1:
        assert aif_mask.ndim == 3, "Wrong AIF ROI dimension {}".format(aif_mask.ndim)
        assert roi_mask.ndim == 3, "Wrong tissue ROI dimension {}".format(roi_mask.ndim)
        aif_mask = aif_mask > 0
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

    # initial parameters
    if b0in is None:
        b0in = {}

    print('Parameters in fit_curves:\n{}'.format(prm))
    if h is None:
        h = im.spacing

    print('Using method {}'.format(method))

    # reduce size?
    # if prm['red']
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
        if (timeline != timeline_data).all():
            resample_data = True
        if (timeline != timeline_aif).all():
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
        aif_value = np.sum(im_aif, axis=(1, 2, 3), where=aif_mask > 0) / np.count_nonzero(aif_mask > 0)
    else:
        aif_value = im_aif
    # im_aif = np.where(aif_mask > 0, im_aif, np.nan)
    # aif_value = np.nanmean(im_aif, axis=(1, 2, 3))
    # aif_value = aif_value'

    # make column vector for gamma fit
    # timeline = timeline.reshape((timeline.shape[0], 1))

    if prm['gammafunc']:
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
    elif prm['aif_method'] == 'parker':
        aif_model = parker(prm['parker_parameters'], len(timeline_aif), timeline / 60)
        print('Normalization method for Parker: {}'.format(prm['aif_normalization_method']))
    elif prm['aif_method'] == 'average':
        aif_model = prm['average_aif']
        print('Normalization method for average AIF: {}'.format(prm['aif_normalization_method']))
    else:
        aif_model = aif_value
    # if prm['aif_method'] in ['parker', 'average']:
    aif_matched, norm_coeff = normalize_aif(aif_model, aif_value, prm['aif_normalization_method'])
    print('Normalization coefficient: {}'.format(norm_coeff))

    if im.ndim > 1:
        # new dimension of image
        # dim = im.ndim
        dim3 = im.shape[1:]

        # reshape the data to n x t matrix to only optimize in the kidney region, to
        # save time
        imini = im
        # im = im2vec4D(im,roi_mask);
        img = np.sum(im, axis=(1, 2, 3), where=roi_mask > 0) / np.count_nonzero(roi_mask > 0)
    else:
        dim3 = (1,)
        imini = im
        img = im

    out = {'handle': [],
           'norm_coeff': norm_coeff
           }
    # initialize by the Patlak model

    if voxel_volume is None:
        voxel_volume = np.prod(h)
    # volume = ones(nnz(roi_mask),1)*voxel_volume;
    # volume = np.full((len(roi_mask.nonzero()[0]), 1), voxel_volume * len(roi_mask.nonzero()[0]))
    if roi_mask is not None:
        volume = voxel_volume * len(roi_mask.nonzero()[0])
    else:
        volume = voxel_volume
    if method == 'annet':
        # prm_model['option'] = prm['option']
        prm_model['upper_bounds'] = prm['upper_bounds']
        prm_model['lower_bounds'] = prm['lower_bounds']
        prm_model['k21'] = prm['k21']
        prm_model['optmethod'] = prm['optmethod']

        # initialize fa and k21 by patlak model
        # if prm['initializepatlak']
        #     msg = ['Initializing by the Patlak model'];
        #     disp(msg);
        #     [outi.k21,outi.fa,outi.c,outi.im,outi.aif_value,outi.gfr] = kidney.fitcurvepatlak(im,aif_value,timeline,prm['meanc'],roi_mask,prm_model);
        #     v = min(outi.fa(1),0.3);
        #     v = max(v,0.1);
        #     b0in.fa = v;
        #     b0in.k21 = outi.k21(1);
        #     msg = ['Obtained parameters'];
        #     disp(msg);
        #     b0in
        # end;
        # run the annet model
        out = kidney.fit_curve_voxel_annet(
            img, aif_matched, timeline, prm['meanc'], volume, hematocrit, b0in, prm_model
        )
    elif method == 'sourbron':
        prm_model['upper_bounds'] = prm['upper_bounds']
        prm_model['lower_bounds'] = prm['lower_bounds']
        prm_model['savefig'] = prm['savefig']
        prm_model['intmethod'] = prm['intmethod']
        prm_model['loss'] = prm['loss']
        prm_model['f_scale'] = prm['f_scale']
        out = fit_curve_voxel_sourbron(
            # img, aif_value, timeline, prm['meanc'], volume, hematocrit, b0in, prm_model
            img, aif_matched, timeline, False, volume, hematocrit, b0in, prm_model
        )
    elif method == 'patlak':
        prm_model = {'meanc': prm['meanc']}
        # prm_model['timepatlak'] = prm['timepatlak']
        out.k21, out.fa, out.c, out.im, out.aif_value, out.gfr = kidney.fit_curve_patlak(
            img, aif_matched, timeline_resample, prm['meanc'], roi_mask, prm_model
        )
    out['timeline'] = timeline
    # out['ind'] = find(roi_mask)
    out['dim3'] = dim3
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


def normalize_aif(model, value, normalization_method='auc'):
    if normalization_method == 'auc':
        norm_coeff = np.sum(value) / np.sum(model)
        # model = model * np.sum(value) / np.sum(model)
    elif normalization_method == 'max':
        norm_coeff = np.max(value) / np.max(model)
        # model = model * np.max(value) / np.max(model)
    elif normalization_method == 'unity':  # unity
        norm_coeff = 1.0 / np.max(model)
        # model = model / np.max(model)
    else:
        raise ValueError('Unknown normalization method: {}'.format(normalization_method))
    model = model * norm_coeff
    delay = find_delay(value, model)
    matched = np.zeros_like(model)
    l = matched.shape[0]
    if delay < 0:
        matched[:l - abs(delay)] = model[(abs(delay)):]
    elif delay > 0:
        matched[delay:] = model[:-delay]
    else:
        matched = model
    return matched, norm_coeff


def smooth_reference(x, y, p0=None):
    def func(x, a, b, c):
        return a + b * x + c * x * x

    if p0 is None:
        p0 = (1, 1, 1)
    print('smooth_reference: x={}, y={}'.format(x.shape, y.shape))
    par, pcov = curve_fit(func, x, y, p0=p0)
    print('Reference curve: {}'.format(par))
    a, b, c = par
    return a + b * x + c * x * x
