import unittest
import numpy as np
import pandas as pd
import os
import io
import urllib3
from perfmod.semiquant import *

dce_etm = 'https://raw.githubusercontent.com/OSIPI/DCE-DSC-MRI_CodeCollection/develop/test/DCEmodels/data/dce_DRO_data_extended_tofts.csv'

if 'https_proxy' in os.environ:
    http = urllib3.ProxyManager(os.environ['https_proxy'])
else:
    http = urllib3.PoolManager()
response = http.request("GET", dce_etm)
if response.status != 200:
    raise FileNotFoundError('Failed to download file. Status code:', response.status)



class TestSemiquant(unittest.TestCase):
    def setUp(self):
        self.df = pd.read_csv(io.StringIO(response.data.decode("utf-8")), sep = ",")

    def test_1D(self):
        desired_washin = [0.0221636, 0.0247124, 0.0080519]
        desired_ttp = [13., 14., 141.]
        desired_peak = [0.2627947, 0.2953006, 0.1713341]
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


            washin_map, ttp_map, peak_amp = compute_washin(c, times=t, metric=c,  # metric='percent',
                                                           baseline_frames=60)
            np.testing.assert_almost_equal(washin_map, desired_washin[currentvoxel], err_msg=f'Washin map for voxel {currentvoxel} is incorrect')
            np.testing.assert_almost_equal(ttp_map, desired_ttp[currentvoxel], err_msg=f'TTP map for voxel {currentvoxel} is incorrect')
            np.testing.assert_almost_equal(peak_amp, desired_peak[currentvoxel], err_msg=f'Peak amp for voxel {currentvoxel} is incorrect')

    def test_1D_percent(self):
        desired_washin = [420587382080923.1, 438882964127041.1, 25273495817833.117]
        desired_ttp = [13., 14., 141.]
        desired_peak = [5470997579656065.0, 6147723110382640.0, 3566924522918533.5]
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

            washin_map, ttp_map, peak_amp = compute_washin(c, times=t, metric='percent',
                                                           baseline_frames=60,
                                                           method='arrival-to-peak')
            np.testing.assert_almost_equal(washin_map, desired_washin[currentvoxel], err_msg=f'Washin map for voxel {currentvoxel} is incorrect', decimal=3)
            np.testing.assert_almost_equal(ttp_map, desired_ttp[currentvoxel], err_msg=f'TTP map for voxel {currentvoxel} is incorrect', decimal=3)
            np.testing.assert_almost_equal(peak_amp, desired_peak[currentvoxel], err_msg=f'Peak amp for voxel {currentvoxel} is incorrect', decimal=3)


    def test_1D_logistic(self):
        desired_washin = [420587382080923.1, 438882964127041.1, 25273495817833.117]
        desired_ttp = [13., 14., 141.]
        desired_peak = [5470997579656065.0, 6147723110382640.0, 3566924522918533.5]
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

            washin_map, ttp_map, peak_amp = compute_washin(c, times=t, metric='percent',
                                                           baseline_frames=60,
                                                           method='logistic')
            # np.testing.assert_almost_equal(washin_map, desired_washin[currentvoxel],
            #                                err_msg=f'Washin map for voxel {currentvoxel} is incorrect', decimal=3)
            # np.testing.assert_almost_equal(ttp_map, desired_ttp[currentvoxel],
            #                                err_msg=f'TTP map for voxel {currentvoxel} is incorrect', decimal=3)
            # np.testing.assert_almost_equal(peak_amp, desired_peak[currentvoxel],
            #                                err_msg=f'Peak amp for voxel {currentvoxel} is incorrect', decimal=3)
            #

if __name__ == '__main__':
    unittest.main()
