import unittest

from ddt import ddt
import os
import pandas as pd
import shutil


from DeconvTest.batch import simulation as sim


@ddt
class TestSimulation(unittest.TestCase):

    def test_cell_params(self):
        sim.generate_cell_parameters('data/params', number_of_cells=3)
        data = pd.read_csv('data/params.csv', sep='\t')
        self.assertEqual(len(data), 3)
        self.assertNotIn('stack', data.columns)
        shutil.rmtree('data/')

    def test_generate_cells(self):
        sim.generate_cell_parameters('data/params.csv', number_of_cells=3)
        sim.generate_cells_batch(params_file='data/params.csv', outputfolder='data/cells/',
                                 input_voxel_size=0.8, print_progress=False)
        files = os.listdir('data/cells/')
        self.assertEqual(len(files), 6)
        shutil.rmtree('data/')

    def test_stack_params(self):
        sim.generate_cell_parameters('data/stack_params.csv', number_of_stacks=5, number_of_cells=3)
        data = pd.read_csv('data/stack_params.csv', sep='\t')
        self.assertEqual(len(data), 15)
        self.assertIn('stack', data.columns)
        shutil.rmtree('data/')

    def test_generate_stacks(self):
        sim.generate_cell_parameters('data/stack_params.csv', number_of_stacks=2)
        sim.generate_cells_batch(params_file='data/stack_params.csv', outputfolder='data/stacks',
                                 input_voxel_size=[1, 0.3, 0.3], stack_size_microns=[10, 30, 30],
                                 print_progress=False)
        files = os.listdir('data/stacks')
        self.assertEqual(len(files), 4)
        shutil.rmtree('data/')

    def test_generate_psfs(self):
        sim.generate_psfs_batch('data/psfs', psf_sigmas=[2, 1.5], psf_aspect_ratios=[1.5, 3, 5],
                                input_voxel_size=0.8, print_progress=False)
        files = os.listdir('data/psfs')
        self.assertEqual(len(files), 12)
        shutil.rmtree('data/')

    def test_simulate(self):
        sim.generate_cell_parameters('data/params.csv', number_of_cells=2)
        sim.generate_cells_batch(params_file='data/params.csv', outputfolder='data/cells/',
                                 input_voxel_size=1, print_progress=False)
        sim.generate_psfs_batch('data/psfs', psf_sigmas=[1.5], psf_aspect_ratios=[1.5],
                                input_voxel_size=1, print_progress=False)
        sim.convolve_batch('data/cells/', 'data/psfs', 'data/convolved/', print_progress=False)
        files = os.listdir('data/convolved')
        self.assertEqual(len(files), 3)
        files = os.listdir('data/convolved/psf_sigma_1.5_aspect_ratio_1.5')
        self.assertEqual(len(files), 4)

        sim.resize_batch('data/convolved/', 'data/resized/', voxel_sizes_for_resizing=[4], print_progress=False)
        files = os.listdir('data/resized')
        self.assertEqual(len(files), 3)
        files = os.listdir('data/resized/psf_sigma_1.5_aspect_ratio_1.5_voxel_size_[4._4._4.]')
        self.assertEqual(len(files), 4)

        sim.add_noise_batch('data/resized/', 'data/noise/', noise_kind=['gaussian', 'poisson'],
                            snr=[None, 10], test_snr_combinations=True, print_progress=False)
        files = os.listdir('data/noise/')
        self.assertEqual(len(files), 6)
        files = os.listdir('data/noise/psf_sigma_1.5_aspect_ratio_1.5_'
                           'voxel_size_[4._4._4.]_noise_gaussian_snr=10_poisson_snr=None')
        self.assertEqual(len(files), 4)
        shutil.rmtree('data/')

    def test_simulate_stack(self):
        sim.generate_cell_parameters('data/stack_params.csv', number_of_stacks=2)
        sim.generate_cells_batch(params_file='data/stack_params.csv', outputfolder='data/stacks',
                                 input_voxel_size=1, stack_size_microns=[10, 30, 30], print_progress=False)
        sim.generate_psfs_batch('data/psfs', psf_sigmas=[1.5], psf_aspect_ratios=[1.5],
                                input_voxel_size=1, print_progress=False)
        sim.convolve_batch('data/stacks/', 'data/psfs', 'data/convolved/', print_progress=False)
        files = os.listdir('data/convolved')
        self.assertEqual(len(files), 3)
        files = os.listdir('data/convolved/psf_sigma_1.5_aspect_ratio_1.5')
        self.assertEqual(len(files), 4)

        sim.resize_batch('data/convolved/', 'data/resized/', voxel_sizes_for_resizing=[3], print_progress=False)
        files = os.listdir('data/resized')
        self.assertEqual(len(files), 3)
        files = os.listdir('data/resized/psf_sigma_1.5_aspect_ratio_1.5_voxel_size_[3._3._3.]')
        self.assertEqual(len(files), 4)

        sim.add_noise_batch('data/resized/', 'data/noise/', noise_kind='gaussian', snr=[5, 2], print_progress=False)
        files = os.listdir('data/noise/')
        self.assertEqual(len(files), 4)
        files = os.listdir('data/noise/psf_sigma_1.5_aspect_ratio_1.5_voxel_size_'
                           '[3._3._3.]_noise_gaussian_snr=5')
        self.assertEqual(len(files), 4)

        shutil.rmtree('data/')


if __name__ == '__main__':
    unittest.main()
