import unittest

from ddt import ddt
import os
import shutil


from DeconvTest.batch import simulation as sim
from DeconvTest.batch import deconvolution as dc


@ddt
class TestDeconvolution(unittest.TestCase):

    def test_simulate(self):
        sim.generate_cell_parameters('data/params.csv', number_of_cells=2)
        sim.generate_cells_batch(params_file='data/params.csv', outputfolder='data/cells/',
                                 resolution=0.3, print_progress=False)
        sim.generate_psfs_batch('data/psfs', sigmas=[2], elongations=[1.5, 3],
                                resolution=0.3, print_progress=False)
        sim.convolve_batch('data/cells/', 'data/psfs', 'data/convolved/', print_progress=False)
        sim.resize_batch('data/convolved/', 'data/resized/', resolutions=[0.5], print_progress=False)
        sim.add_noise_batch('data/resized/', 'data/noise/', gaussian_snrs=[None],
                            poisson_snrs=[None], print_progress=False)
        dc.deconvolve_batch(inputfolder='data/noise/',
                            outputfolder='data/deconvolved/', algorithm='deconvolution_lab_rif',
                            rif_lambda=[0.001, 0.002], print_progress=False, max_threads=4)
        self.assertEqual(os.path.exists('data/deconvolved/DeconvolutionLab-RIF_lambda=0.001'), True)
        self.assertEqual(len(os.listdir('data/deconvolved/DeconvolutionLab-RIF_lambda=0.001')), 2)
        shutil.rmtree('data/')

if __name__ == '__main__':
    unittest.main()



