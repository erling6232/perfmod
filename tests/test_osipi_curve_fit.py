import os
import io
import unittest
import numpy as np
import pandas as pd
import urllib3
from perfmod.fit_curves import fit_curves

dce_etm = 'https://raw.githubusercontent.com/OSIPI/DCE-DSC-MRI_CodeCollection/develop/test/DCEmodels/data/dce_DRO_data_extended_tofts.csv'

if 'https_proxy' in os.environ:
    http = urllib3.ProxyManager(os.environ['https_proxy'])
else:
    http = urllib3.PoolManager()
response = http.request("GET", dce_etm)
if response.status != 200:
    raise FileNotFoundError('Failed to download file. Status code:', response.status)


class test_osipi_curve_fit(unittest.TestCase):
    def setUp(self):
        self.df = pd.read_csv(io.StringIO(response.data.decode("utf-8")), sep = ",")

    def test_etm(self):
        for currentvoxel in range(3):
            labelname = 'test_vox_T' + str(currentvoxel + 1) + '_highSNR'
            testdata = self.df[(self.df['label'] == labelname)]
            t = testdata['t'].to_numpy()
            t = np.fromstring(t[0], dtype=float, sep=' ')  # t in sec.
            c = testdata['C'].to_numpy()
            c = np.fromstring(c[0], dtype=float, sep=' ')
            ca = testdata['ca'].to_numpy()
            ca = np.fromstring(ca[0], dtype=float, sep=' ')

            ktrans_ref = testdata['Ktrans'].to_numpy()[0]
            ve_ref = testdata['ve'].to_numpy()[0]
            vp_ref = testdata['vp'].to_numpy()[0]


            method = 'etm'  # 'gctt'  # 'annet' 'tm'
            hematocrit = 0.45
            prmin = {
                'aif_method': 'individual',  # 'average',  # 'individual',  # 'parker',
                'aif_normalization_method': 'none',  # 'parker', 'unity', 'max', 'auc',
                'hematocrit': hematocrit,
                'optmethod': 'curve_fit',
                'Cp': False,
            }
            out = fit_curves(c, method,
                             im_aif=ca,
                             timeline_data=t,
                             timeline_aif=t,
                             prmin=prmin,
                             hematocrit=hematocrit)
            print(f'Parameters: {prmin["aif_method"]} {prmin["aif_normalization_method"]}')
            print(f'Fitted:{out["b"]}')

            ktrans = out['ktrans'] * 60  # Units of ml/ml/min
            ve = out['ve']
            vp = out['vp']

            self.assertAlmostEqual(ktrans_ref, ktrans, 4)
            self.assertAlmostEqual(ve_ref, ve, 3)
            self.assertAlmostEqual(vp_ref, vp, 2)

            if True:
                import matplotlib.pyplot as plt
                from perfmod.fit_curves import show
                fig, ax = plt.subplots(1, 2)
                fig.suptitle(f'AIF: {prmin["aif_method"]}. Normalization: {prmin["aif_normalization_method"]}')
                norm_coeff = out['norm_coeff']
                show(out['aif'], x=out['timeline'], ax=ax[0], show=False, label='AIF')

                tissue = c * norm_coeff
                if len(tissue) % 2 == 0:
                    tissue = np.append(tissue, [tissue[-1]])
                tissue = tissue.reshape((tissue.shape[0], 1))
                show(tissue, x=out['timeline'], ax=ax[1], show=False, label='tissue')
                show(out['fitted'], x=out['timeline'], ax=ax[1], show=False, label=method)

                plt.show()

    def test_gctt(self):
        for currentvoxel in range(3):
            labelname = 'test_vox_T' + str(currentvoxel + 1) + '_highSNR'
            testdata = self.df[(self.df['label'] == labelname)]
            t = testdata['t'].to_numpy()
            t = np.fromstring(t[0], dtype=float, sep=' ')  # t in sec.
            c = testdata['C'].to_numpy()
            c = np.fromstring(c[0], dtype=float, sep=' ')
            ca = testdata['ca'].to_numpy()
            ca = np.fromstring(ca[0], dtype=float, sep=' ')

            ktrans_ref = testdata['Ktrans'].to_numpy()[0]
            ve_ref = testdata['ve'].to_numpy()[0]
            vp_ref = testdata['vp'].to_numpy()[0]


            method = 'gctt'  # 'gctt'  # 'annet' 'tm'
            hematocrit = 0.45
            prmin = {
                'aif_method': 'individual',  # 'average',  # 'individual',  # 'parker',
                'aif_normalization_method': 'none',  # 'parker', 'unity', 'max', 'auc',
                'hematocrit': hematocrit,
                'optmethod': 'curve_fit',
            }
            out = fit_curves(c, method,
                             im_aif=ca,
                             timeline_data=t,
                             timeline_aif=t,
                             prmin=prmin,
                             hematocrit=hematocrit)
            print(f'Parameters: {prmin["aif_method"]} {prmin["aif_normalization_method"]}')
            print(f'Fitted:{out["b"]}')

            ktrans = out['E'] * out['F'] * 60  # Units of ml/ml/min
            ve = out['ve']
            # vp = out['vp']

            self.assertAlmostEqual(ktrans_ref, ktrans, 2)
            self.assertAlmostEqual(ve_ref, ve, 2)
            # # self.assertAlmostEqual(expect_vp, vp, 2)

            if True:
                import matplotlib.pyplot as plt
                from perfmod.fit_curves import show
                fig, ax = plt.subplots(1, 2)
                fig.suptitle(f'AIF: {prmin["aif_method"]}. Normalization: {prmin["aif_normalization_method"]}')
                norm_coeff = out['norm_coeff']
                show(out['aif'], x=out['timeline'], ax=ax[0], show=False, label='Used AIF')

                tissue = c * norm_coeff
                if len(tissue) % 2 == 0:
                    tissue = np.append(tissue, [tissue[-1]])
                tissue = tissue.reshape((tissue.shape[0], 1))
                show(tissue, x=out['timeline'], ax=ax[1], show=False, label='tissue')
                # show(tissue_OM, x=dce.timeline/60, ax=ax[1], show=False, label='tissue_OM')
                show(out['fitted'], x=out['timeline'], ax=ax[1], show=False, label=method)
                # bout = out['b'][:, 0]
                # h = out['fun'](out['timeline'], *bout)
                # show(h, x=out['timeline'], ax=ax[1], show=False, label=method)

                plt.show()

    @unittest.skip("skipping test_annet")
    def test_annet(self):
        for currentvoxel in range(3):
            labelname = 'test_vox_T' + str(currentvoxel + 1) + '_highSNR'
            testdata = self.df[(self.df['label'] == labelname)]
            t = testdata['t'].to_numpy()
            t = np.fromstring(t[0], dtype=float, sep=' ')  # t in sec.
            c = testdata['C'].to_numpy()
            c = np.fromstring(c[0], dtype=float, sep=' ')
            ca = testdata['ca'].to_numpy()
            ca = np.fromstring(ca[0], dtype=float, sep=' ')

            ktrans_ref = testdata['Ktrans'].to_numpy()[0]
            ve_ref = testdata['ve'].to_numpy()[0]
            vp_ref = testdata['vp'].to_numpy()[0]


            method = 'annet'
            hematocrit = 0.45
            prmin = {
                'aif_method': 'individual',  # 'average',  # 'individual',  # 'parker',
                'aif_normalization_method': 'none',  # 'parker', 'unity', 'max', 'auc',
                'hematocrit': hematocrit,
                'optmethod': 'curve_fit',
            }
            out = fit_curves(c, method,
                             im_aif=ca,
                             timeline_data=t,
                             timeline_aif=t,
                             prmin=prmin,
                             hematocrit=hematocrit)
            print(f'Parameters: {prmin["aif_method"]} {prmin["aif_normalization_method"]}')
            print(f'Fitted:{out["b"]}')

            ktrans = out['k12']  # Units of ml/ml/min
            ve = out['ve']
            vp = out['vp']

            self.assertAlmostEqual(ktrans_ref, ktrans, 4)
            self.assertAlmostEqual(ve_ref, ve, 3)
            self.assertAlmostEqual(vp_ref, vp, 2)

            if True:
                import matplotlib.pyplot as plt
                from perfmod.fit_curves import show
                fig, ax = plt.subplots(1, 2)
                fig.suptitle(f'AIF: {prmin["aif_method"]}. Normalization: {prmin["aif_normalization_method"]}')
                norm_coeff = out['norm_coeff']
                show(out['aif'], x=out['timeline'], ax=ax[0], show=False, label='AIF')

                tissue = c * norm_coeff
                if len(tissue) % 2 == 0:
                    tissue = np.append(tissue, [tissue[-1]])
                tissue = tissue.reshape((tissue.shape[0], 1))
                show(tissue, x=out['timeline'], ax=ax[1], show=False, label='tissue')
                show(out['fitted'], x=out['timeline'], ax=ax[1], show=False, label=method)

                plt.show()

    def test_sourbron(self):
        for currentvoxel in range(3):
            labelname = 'test_vox_T' + str(currentvoxel + 1) + '_highSNR'
            testdata = self.df[(self.df['label'] == labelname)]
            t = testdata['t'].to_numpy()
            t = np.fromstring(t[0], dtype=float, sep=' ')  # t in sec.
            c = testdata['C'].to_numpy()
            c = np.fromstring(c[0], dtype=float, sep=' ')
            ca = testdata['ca'].to_numpy()
            ca = np.fromstring(ca[0], dtype=float, sep=' ')

            ktrans_ref = testdata['Ktrans'].to_numpy()[0]
            ve_ref = testdata['ve'].to_numpy()[0]
            vp_ref = testdata['vp'].to_numpy()[0]


            method = 'sourbron'
            hematocrit = 0.45
            prmin = {
                'aif_method': 'individual',  # 'average',  # 'individual',  # 'parker',
                'aif_normalization_method': 'none',  # 'parker', 'unity', 'max', 'auc',
                'hematocrit': hematocrit,
                'optmethod': 'curve_fit',
            }
            out = fit_curves(c, method,
                             im_aif=ca,
                             timeline_data=t,
                             timeline_aif=t,
                             prmin=prmin,
                             hematocrit=hematocrit)
            print(f'Parameters: {prmin["aif_method"]} {prmin["aif_normalization_method"]}')
            print(f'Fitted:{out["b"]}')

            # Fp = out['vp'] / out['Tp']
            # E = out['Ft'] / Fp
            # ktrans = E * Fp * 60  # Units of ml/ml/min
            # ve = fp * (out['Tt'] - out['Tp'])
            # vp = out['vp']

            # self.assertAlmostEqual(ktrans_ref, ktrans, 4)
            # self.assertAlmostEqual(ve_ref, ve, 3)
            # self.assertAlmostEqual(vp_ref, vp, 2)
            match currentvoxel:
                case 0:
                    self.assertAlmostEqual(0.0019, out['vp'], 4)
                    self.assertAlmostEqual(0.1665, out['Tp'], 4)
                    self.assertAlmostEqual(9.6678e-5, out['Ft'], 4)
                    self.assertAlmostEqual(165.4501, out['Tt'], 4)
                case 1:
                    self.assertAlmostEqual(0.0024, out['vp'], 4)
                    self.assertAlmostEqual(0.1896, out['Tp'], 4)
                    self.assertAlmostEqual(0.00013049, out['Ft'], 4)
                    self.assertAlmostEqual(118.1522, out['Tt'], 4)
                case 2:
                    self.assertAlmostEqual(0.00069, out['vp'], 4)
                    self.assertAlmostEqual(0.2827, out['Tp'], 4)
                    self.assertAlmostEqual(0.0001278, out['Ft'], 4)
                    self.assertAlmostEqual(244.4024, out['Tt'], 4)

            if True:
                import matplotlib.pyplot as plt
                from perfmod.fit_curves import show
                fig, ax = plt.subplots(1, 2)
                fig.suptitle(f'AIF: {prmin["aif_method"]}. Normalization: {prmin["aif_normalization_method"]}')
                norm_coeff = out['norm_coeff']
                show(out['aif'], x=out['timeline'], ax=ax[0], show=False, label='AIF')

                tissue = c * norm_coeff
                if len(tissue) % 2 == 0:
                    tissue = np.append(tissue, [tissue[-1]])
                tissue = tissue.reshape((tissue.shape[0], 1))
                show(tissue, x=out['timeline'], ax=ax[1], show=False, label='tissue')
                show(out['fitted'], x=out['timeline'], ax=ax[1], show=False, label=method)

                plt.show()

