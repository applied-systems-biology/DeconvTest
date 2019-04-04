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
        sim.generate_cell_parameters('data/params.csv', number_of_cells=5)
        sim.generate_cells_batch(params_file='data/params.csv', outputfolder='data/cells/',
                                 resolution=0.5, print_progress=False)
        sim.generate_psfs_batch('data/psfs', sigmas=[1.5], elongations=[4], resolution=0.5, print_progress=False)
        sim.convolve_batch('data/cells/', 'data/psfs', 'data/convolved/', print_progress=False)

        quant.segment_batch('data/convolved', 'data/segmented', thr=5, preprocess=True, print_progress=False,
                            log_computing_time=True)
        files = os.listdir('data/segmented')
        self.assertEqual(len(files), 1)
        files = os.listdir('data/segmented/psf_sigma_1.5_elongation_4')
        self.assertEqual(len(files), 10)

        quant.compare_to_ground_truth_batch('data/segmented', 'data/cells', 'data/accuracy', print_progress=False)
        files = os.listdir('data/accuracy')
        self.assertEqual(len(files), 1)
        files = os.listdir('data/accuracy/psf_sigma_1.5_elongation_4')
        self.assertEqual(len(files), 5)
        self.assertEqual(os.path.exists('data/accuracy.csv'), True)

        quant.measure_dimensions_batch('data/segmented', 'data/dimensions', print_progress=False)
        files = os.listdir('data/dimensions')
        self.assertEqual(len(files), 1)
        files = os.listdir('data/dimensions/psf_sigma_1.5_elongation_4')
        self.assertEqual(len(files), 5)

        quant.measure_dimensions_batch('data/cells', 'data/dimensions', print_progress=False)
        files = os.listdir('data/dimensions')
        self.assertEqual(len(files), 6)

        self.assertEqual(os.path.exists('data/dimensions.csv'), True)

        quant.compute_dimension_errors('data/dimensions.csv', 0.5)
        self.assertEqual(os.path.exists('data/dimensions_errors.csv'), True)

        quant.combine_log('data/log')
        self.assertEqual(os.path.exists('data/log.csv'), True)

        quant.extract_metadata('data/accuracy.csv', 0.5)
        stat = pd.read_csv('data/accuracy.csv', sep='\t', index_col=0)
        for col in ['Kind', 'Voxel size y', 'resolution']:
            self.assertIn(col, stat.columns)

        quant.extract_metadata('data/log.csv', 0.5)
        stat = pd.read_csv('data/log.csv', sep='\t', index_col=0)
        for col in ['Kind', 'Voxel size x', 'resolution']:
            self.assertIn(col, stat.columns)

        stat = pd.read_csv('data/dimensions_errors.csv', sep='\t', index_col=0)
        for col in ['Kind', 'Voxel size z', 'resolution', 'size_y']:
            self.assertIn(col, stat.columns)

        shutil.rmtree('data/')


if __name__ == '__main__':
    unittest.main()



