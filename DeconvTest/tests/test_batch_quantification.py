import unittest

from ddt import ddt
import os
import pandas as pd
import shutil

from DeconvTest.batch import simulation as sim
from DeconvTest.batch import quantification as quant


@ddt
class TestSimulation(unittest.TestCase):

    def test_simulate(self):
        sim.generate_cell_parameters('data/params.csv', number_of_cells=2)
        sim.generate_cells_batch(params_file='data/params.csv',
                                 outputfolder='data/cells/',
                                 input_voxel_size=1,
                                 print_progress=False)
        sim.generate_psfs_batch('data/psfs',
                                psf_sigmas=[1.5],
                                psf_aspect_ratios=[4],
                                input_voxel_size=1,
                                print_progress=False)
        sim.convolve_batch('data/cells/',
                           'data/psfs',
                           'data/convolved/',
                           print_progress=False)

        quant.accuracy_batch(inputfolder='data/convolved',
                             reffolder='data/cells',
                             outputfolder='data/accuracy',
                             print_progress=False)
        files = os.listdir('data/accuracy')
        self.assertEqual(len(files), 1)
        files = os.listdir('data/accuracy/psf_sigma_1.5_aspect_ratio_4')
        self.assertEqual(len(files), 2)
        self.assertEqual(os.path.exists('data/accuracy.csv'), True)

        stat = pd.read_csv('data/accuracy.csv', sep='\t', index_col=0)
        for col in ['Voxel size y', 'Voxel size']:
            self.assertIn(col, stat.columns)

        shutil.rmtree('data/')


if __name__ == '__main__':
    unittest.main()



