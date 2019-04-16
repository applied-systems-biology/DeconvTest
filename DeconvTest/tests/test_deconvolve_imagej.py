import unittest

from ddt import ddt

from DeconvTest import Cell
from DeconvTest import PSF
from DeconvTest.modules.deconvolution_fiji import *


@ddt
class TestDeconvolve(unittest.TestCase):

    def test_rif(self):
        cell = Cell(resolution=0.3, size=[10, 10, 10], phi=0, theta=0)
        psf = PSF(sigma=0.5, aspect_ratio=2)
        psf.save('data/psf.tif')
        cell.convolve(psf)
        cell.save('data/cell.tif')
        imagej_path = get_fiji_path()
        deconvolution_lab_rif(imagej_path=imagej_path, inputfile=os.getcwd() + '/data/cell.tif',
                              psffile=os.getcwd() + '/data/psf.tif',
                              regularization_lambda=0.001,
                              outputfile=os.getcwd() + '/data/deconvolved.tif', **dict())
        self.assertEqual(os.path.exists('data/deconvolved.tif'), True)
        shutil.rmtree('data/')

    def test_rltv(self):
        cell = Cell(resolution=0.3, size=[10, 10, 10], phi=0, theta=0)
        psf = PSF(sigma=0.5, aspect_ratio=2)
        psf.save('data/psf.tif')
        cell.convolve(psf)
        cell.save('data/cell.tif')
        imagej_path = get_fiji_path()
        deconvolution_lab_rltv(imagej_path=imagej_path, inputfile=os.getcwd() + '/data/cell.tif',
                               psffile=os.getcwd() + '/data/psf.tif',
                               regularization_lambda=0.001, iterations=5,
                               outputfile=os.getcwd() + '/data/deconvolved.tif')
        self.assertEqual(os.path.exists('data/deconvolved.tif'), True)
        shutil.rmtree('data/')

    def test_iterative(self):
        cell = Cell(resolution=0.3, size=[10, 10, 10], phi=0, theta=0)
        psf = PSF(sigma=0.5, aspect_ratio=2)
        psf.save('data/psf.tif')
        cell.convolve(psf)
        cell.save('data/cell.tif')
        iterative_deconvolve_3d(inputfile=os.getcwd() + '/data/cell.tif',
                                psffile=os.getcwd() + '/data/psf.tif',
                                outputfile=os.getcwd() + '/data/deconvolved.tif',
                                normalize=False, perform=True, detect=True, wiener=0.1,
                                low=1, terminate=0.0001, iterations=2)
        self.assertEqual(os.path.exists('data/deconvolved.tif'), True)
        shutil.rmtree('data/')


if __name__ == '__main__':

    unittest.main()


