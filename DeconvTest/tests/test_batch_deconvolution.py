import unittest

from ddt import ddt
import os
import shutil


from DeconvTest.batch import simulation as sim
from DeconvTest.batch.deconvolution import deconvolve_batch


@ddt
class TestDeconvolution(unittest.TestCase):

    def test_simulate(self):
        sim.generate_cell_parameters('data/params.csv', number_of_cells=1)
        sim.generate_cells_batch(params_file='data/params.csv', outputfolder='data/cells/',
                                 input_voxel_size=3, print_progress=False)
        sim.generate_psfs_batch('data/psfs', psf_sigmas=[0.1], psf_aspect_ratios=[1.5],
                                input_voxel_size=3, print_progress=False)
        sim.convolve_batch('data/cells/', 'data/psfs', 'data/convolved/', print_progress=False)
        deconvolve_batch(inputfolder='data/convolved/',
                         outputfolder='data/deconvolved/',
                         deconvolution_algorithm=['deconvolution_lab_rif', 'iterative_deconvolve_3d'],
                         deconvolution_lab_rif_regularization_lambda=0.001,
                         iterative_deconvolve_3d_normalize=[True, False],
                         iterative_deconvolve_3d_wiener=[0.1],
                         iterative_deconvolve_3d_terminate=0.1,
                         print_progress=False, max_threads=4)
        files = os.listdir('data/deconvolved')
        self.assertEqual(len(files), 3)
        self.assertEqual(len(os.listdir('data/deconvolved/deconvolution_lab_rif_regularization_lambda=0.001')), 1)
        shutil.rmtree('data/')


if __name__ == '__main__':
    unittest.main()



