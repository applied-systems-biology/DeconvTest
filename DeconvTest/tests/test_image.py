import unittest

import numpy as np
from ddt import ddt, data
import shutil

from DeconvTest.classes.image import Image
from DeconvTest import PSF


@ddt
class TestImageClass(unittest.TestCase):
    def test_empty_arguments(self):
        img = Image()
        for var in ['image', 'metadata', 'filename']:
            self.assertIn(var, img.__dict__)
            self.assertEqual(img.__dict__[var], None)

    def test_empty_arguments_from_file(self):
        img = Image()
        self.assertRaises(TypeError, img.from_file)

    def test_from_file_does_not_exist(self):
        img = Image()
        self.assertRaises(ValueError, img.from_file, filename='bla/image.tif')

    @data(
        'data/test_data/cell.tif',
        'data/test_data/output',
        'data/test_data/output.png'
    )
    def test_save_and_read(self, filename):
        img = Image()
        img.image = np.ones([5, 5, 5])
        img.save(filename)
        fn = img.filename
        img.from_file(fn)
        self.assertIsNotNone(img.image)
        shutil.rmtree('data/')

    @data(
        'cell00.tif',
        'input_45.png',
        'input_46.0',
        'cell23.tif',
        'cell23.tif',
        'ImageJ_Lab_RIF/lambda=10e10/psf_sigma_1_elongation_3.5/cell-16.tif',
        'Gaussian_SNR=30/voxel_size_3._0.3_0.3/psf_sigma_0.1_elongation_2/data/cell_049.tif',
        'SNR=5.5/iterations=245/lambda=1e-5//voxel_size_2-0.1/psf_sigma=3.5_elongation=2/data/cell=25.0.tif'
    )
    def test_filename(self, filename):
        img = Image(filename)
        for var in ['image', 'metadata', 'filename']:
            self.assertIn(var, img.__dict__)

        for var in ['image', 'metadata']:
            self.assertEqual(img.__dict__[var], None)

        self.assertEqual(img.filename, filename)

    @data(
        ([100, 100, 100], 0.1, [10, 10, 10]),
        ([100, 100, 100], [0.2, 0.1, 0.1], [20, 10, 10]),
        ([100, 100, 100], 0.3366, [34, 34, 34]),
        ([100, 100, 100], 0.33333, [33, 33, 33])
    )
    def test_resize(self, case):
        old_shape, zoom, new_shape = case
        img = Image()
        img.image = np.ones(old_shape)
        img.resize(zoom=zoom)
        self.assertEqual(img.image.shape, tuple(new_shape))

    @data(
        ([10, 100, 100], 0.1),
        ([10, 100, 100], [0.5, 0.001, 0.001]),
        ([100, 10, 100], 0.1)
    )
    def test_resize_larger_than_3(self, case):
        old_shape, zoom = case
        img = Image()
        img.image = np.ones(old_shape)
        img.resize(zoom=zoom)
        for i in range(len(img.image.shape)):
            self.assertGreater(img.image.shape[i], 3)

    def test_convolve(self):
        img = Image()
        arr = np.zeros([50, 50, 50])
        arr[10:-10, 10:-10, 10:-10] = 255
        img.image = arr
        psf = PSF()
        psf.generate(5, 4)
        img.convolve(psf)
        self.assertEqual(len(img.image.shape), len(arr.shape))

    def test_convolve_None(self):
        img = Image()
        psf = PSF()
        psf.generate(5, 4)
        self.assertRaises(ValueError, img.convolve, psf)

    def test_convolve_None2(self):
        img = Image()
        arr = np.zeros([50, 50, 50])
        arr[10:-10, 10:-10, 10:-10] = 255
        psf = PSF()
        self.assertRaises(ValueError, img.convolve, psf)

    def test_noise(self):
        img = Image()
        arr = np.zeros([50, 50, 50])
        arr[10:-10, 10:-10, 10:-10] = 255
        img.image = arr
        img.add_noise(kind='gaussian', snr=2)
        self.assertEqual(img.image.shape, arr.shape)

    def test_noise2(self):
        img = Image()
        arr = np.zeros([50, 50, 50])
        arr[10:-10, 10:-10, 10:-10] = 255
        img.image = arr
        img.add_noise(kind='poisson', snr=2)
        self.assertEqual(img.image.shape, arr.shape)

    def test_noise_None(self):
        img = Image()
        self.assertRaises(ValueError, img.add_noise, 'poisson', 2)

    def test_noise_None2(self):
        img = Image()
        self.assertRaises(ValueError, img.add_noise, 'gaussian', 2)

    @data(
        'gaussian',
        'poisson',
        ['poisson', 'gaussian'],
        ['gaussian', 'poisson'],
        ['poisson', 'poisson', 'gaussian'],
    )
    def test_valid_noise_types(self, kind):
        img = Image()
        arr = np.zeros([50, 50, 50])
        arr[10:-10, 10:-10, 10:-10] = 255
        img.image = arr
        img.add_noise(kind=kind, snr=10)
        self.assertIsNotNone(img.image)
        self.assertGreater(np.sum(np.abs(img.image - arr)), 0)

    @data(
        'argwr',
        'invalid_noise',
    )
    def test_valid_types(self, kind):
        img = Image()
        arr = np.zeros([50, 50, 50])
        arr[10:-10, 10:-10, 10:-10] = 255
        img.image = arr
        self.assertRaises(AttributeError, img.add_noise, kind=kind, snr=10)


if __name__ == '__main__':

    unittest.main()
