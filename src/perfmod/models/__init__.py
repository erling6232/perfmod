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
