def default_parameters() -> dict:
    return {
        'red': False,
        'vis': False,
        'meanc': True,
        'collapse': False,
        'initializepatlak': False,
        'optmethod': 'curve_fit',  # 'least_squares'
        'gammafunc': False,
        'k21': 0.001,

        # Saving figure?
        'savefig': False,

        # Integration method
        # prm['intmethod'] = 'matrix';
        'intmethod': 'conv',
        # prm['intmethod'] = 'numint';
        # 'intmethod': 'loop',
        'loss': 'linear',
        'x_scale': None,
        'f_scale': 1.0,
    }
