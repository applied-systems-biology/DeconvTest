import unittest

from ddt import ddt
import os
import shutil


from DeconvTest.batch import simulation as sim


@ddt
class TestSimulation(unittest.TestCase):

    def test_cell_params(self):
        sim.generate_cell_parameters('data/params')
        self.assertEqual(os.path.exists('data/params.csv'), True)
        shutil.rmtree('data/')

    def test_generate_cells(self):
        sim.generate_cell_parameters('data/params.csv', number_of_cells=5)
        sim.generate_cells_batch(params_file='data/params.csv', outputfolder='data/cells/',
                                 resolution=0.3, print_progress=False)
        files = os.listdir('data/cells/')
        self.assertEqual(len(files), 10)
        shutil.rmtree('data/')

    def test_stack_params(self):
        sim.generate_stack_parameters('data/stack_params', number_of_stacks=5)
        files = os.listdir('data/stack_params')
        self.assertEqual(len(files), 5)
        shutil.rmtree('data/')

    def test_generate_stacks(self):
        sim.generate_stack_parameters('data/stack_params', number_of_stacks=5)
        sim.generate_stacks_batch(params_folder='data/stack_params', outputfolder='data/stacks',
                                  resolution=[1, 0.3, 0.3], stack_size_microns=[30, 100, 100], print_progress=False)
        files = os.listdir('data/stacks')
        self.assertEqual(len(files), 10)
        shutil.rmtree('data/')

    def test_generate_psfs(self):
        sim.generate_psfs_batch('data/psfs', sigmas=[2, 1.5], aspect_ratios=[1.5, 3, 5],
                                resolution=0.3, print_progress=False)
        files = os.listdir('data/psfs')
        self.assertEqual(len(files), 12)
        shutil.rmtree('data/')

    def test_simulate(self):
        sim.generate_cell_parameters('data/params.csv', number_of_cells=3)
        sim.generate_cells_batch(params_file='data/params.csv', outputfolder='data/cells/',
                                 resolution=0.8, print_progress=False)
        sim.generate_psfs_batch('data/psfs', sigmas=[2, 1.5], aspect_ratios=[1.5, 3],
                                resolution=0.8, print_progress=False)
        sim.convolve_batch('data/cells/', 'data/psfs', 'data/convolved/', print_progress=False)
        files = os.listdir('data/convolved')
        self.assertEqual(len(files), 12)
        files = os.listdir('data/convolved/psf_sigma_1.5_aspect_ratio_1.5')
        self.assertEqual(len(files), 6)

        sim.resize_batch('data/convolved/', 'data/resized/', resolutions=[3, [3, 1, 1]], print_progress=False)
        files = os.listdir('data/resized')
        self.assertEqual(len(files), 24)
        files = os.listdir('data/resized/psf_sigma_1.5_aspect_ratio_1.5_voxel_size_[3_1_1]')
        self.assertEqual(len(files), 6)

        sim.add_noise_batch('data/resized/', 'data/noise/', kind=['gaussian', 'poisson'],
                            snr=[None, 5], test_snr_combinations=True, print_progress=False)
        files = os.listdir('data/noise/')
        self.assertEqual(len(files), 48)
        files = os.listdir('data/noise/psf_sigma_1.5_aspect_ratio_1.5_'
                           'voxel_size_[3_1_1]_noise_gaussian_snr=5_poisson_snr=None')
        self.assertEqual(len(files), 6)
        shutil.rmtree('data/')

    def test_simulate_stack(self):
        sim.generate_stack_parameters('data/stack_params', number_of_stacks=2)
        sim.generate_stacks_batch(params_folder='data/stack_params', outputfolder='data/stacks',
                                  resolution=1, stack_size_microns=[10, 30, 30], print_progress=False)
        sim.generate_psfs_batch('data/psfs', sigmas=[1.5], aspect_ratios=[1.5],
                                resolution=1, print_progress=False)
        sim.convolve_batch('data/stacks/', 'data/psfs', 'data/convolved/', print_progress=False)
        files = os.listdir('data/convolved')
        self.assertEqual(len(files), 3)
        files = os.listdir('data/convolved/psf_sigma_1.5_aspect_ratio_1.5')
        self.assertEqual(len(files), 4)

        sim.resize_batch('data/convolved/', 'data/resized/', resolutions=[3], print_progress=False)
        files = os.listdir('data/resized')
        self.assertEqual(len(files), 3)
        files = os.listdir('data/resized/psf_sigma_1.5_aspect_ratio_1.5_voxel_size_[3_3_3]')
        self.assertEqual(len(files), 4)

        sim.add_noise_batch('data/resized/', 'data/noise/', kind='gaussian', snr=[5, 2], print_progress=False)
        files = os.listdir('data/noise/')
        self.assertEqual(len(files), 4)
        files = os.listdir('data/noise/psf_sigma_1.5_aspect_ratio_1.5_voxel_size_'
                           '[3_3_3]_noise_gaussian_snr=5')
        self.assertEqual(len(files), 4)

        shutil.rmtree('data/')


if __name__ == '__main__':
    unittest.main()
