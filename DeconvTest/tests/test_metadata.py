import unittest

from ddt import ddt, data
from DeconvTest import Metadata


@ddt
class TestMetadataFromFilename(unittest.TestCase):

    def test_cell_id_float(self):
        self.assertRaises(ValueError, Metadata, string="input 4.5")

    @data(
        ('cell00', 0),
        ('input_45', 45),
        ('input_46.0', 46),
        ('cell23.tif', 23),
        ('cell11/input_42.0', 42),
        ('cell-16', 16)
    )
    def test_cell_id(self, case):
        input_name, result = case
        metadata = Metadata(string=input_name)
        self.assertEqual(metadata['CellID'], result)

    @data(
        ('cell00', 'Input'),
        ('ImageJ_Lab_RIF/lambda=10e10/psf_sigma_1_elongation_3.5/cell-16.tif', 'Convolved'),
        ('ImageJ_Lab_RIF/lambda=10e10/elongation_3.5/cell-16.tif', 'Input'),
        ('input_46.0', 'Input'),
        ('cell.tif', None)
    )
    def test_data_kind(self, case):
        input_name, result = case
        metadata = Metadata(string=input_name)
        self.assertEqual(metadata['Kind'], result)

    @data(
        ('input.tif', None),
        ('ImageJ_Lab_RIF/lambda=10e10/psf_sigma_1_elongation_3.5/cell-16.tif', 1),
        ('Gaussian_SNR=30/voxel_size_3._0.3_0.3/psf_sigma_0.1_elongation_2/data/cell_049.tif', 0.1),
        ('SNR=5.5/iterations=245/lambda=1e-5//voxel_size_2-0.1/psf_sigma=3.5_elongation=2/data/cell=25.0.tif', 3.5)
    )
    def test_psf_sigma(self, case):
        input_name, result = case
        metadata = Metadata(string=input_name)
        self.assertEqual(metadata['PSF sigma X'], result)

    @data(
        ('input.tif', None),
        ('ImageJ_Lab_RIF/lambda=10e10/psf_sigma_1_elongation_3.5/cell-16.tif', 3.5),
        ('Gaussian_SNR=30/voxel_size_3._0.3_0.3/psf_sigma_0.1_elongation_2/data/cell_049.tif', 2),
        ('SNR=5.5/iterations=245/lambda=1e-5//voxel_size_2-0.1/psf_sigma=3.5_elongation=2/data/cell=25.0.tif', 2)
    )
    def test_psf_elongation(self, case):
        input_name, result = case
        metadata = Metadata(string=input_name)
        self.assertEqual(metadata['PSF elongation'], result)

    @data(
        ('input.tif', None),
        ('ImageJ_Lab_RIF/lambda=10e10/psf_sigma_1_elongation_3.5/cell-16.tif', 3.5),
        ('Gaussian_SNR=30/voxel_size_3._0.3_0.3/psf_sigma_0.1_elongation_2/data/cell_049.tif', 0.2),
        ('SNR=5.5/iterations=245/lambda=1e-5//voxel_size_2-0.1/psf_sigma=3.5_elongation=2/data/cell=25.0.tif', 7)
    )
    def test_psf_sigma_z(self, case):
        input_name, result = case
        metadata = Metadata(string=input_name)
        self.assertEqual(metadata['PSF sigma Z'], result)

    @data(
        ('input.tif', None),
        ('ImageJ_Lab_RIF/lambda=10e10/psf_sigma_1_elongation_3.5/cell-16.tif', None),
        ('Gaussian_SNR=30/voxel_size_3._0.3_0.3/psf_sigma_0.1_elongation_2/data/cell_049.tif', 3),
        ('SNR=5.5/iterations=245/lambda=1e-5//voxel_size_2-0.1/psf_sigma=3.5_elongation=2/data/cell=25.0.tif', 2)
    )
    def test_voxel_size1(self, case):
        input_name, result = case
        metadata = Metadata(string=input_name)
        self.assertEqual(metadata['Voxel size z'], result)

    @data(
        ('input.tif', None),
        ('ImageJ_Lab_RIF/lambda=10e10/psf_sigma_1_elongation_3.5/cell-16.tif', None),
        ('Gaussian_SNR=30/voxel_size_3._0.3_0.3/psf_sigma_0.1_elongation_2/data/cell_049.tif', 0.3),
        ('SNR=5.5/iterations=245/lambda=1e-5//voxel_size_2-0.1/psf_sigma=3.5_elongation=2/data/cell=25.0.tif', 0.1)
    )
    def test_voxel_size2(self, case):
        input_name, result = case
        metadata = Metadata(string=input_name)
        self.assertEqual(metadata['Voxel size y'], result)

    @data(
        ('input.tif', None),
        ('ImageJ_Lab_RIF/lambda=10e10/psf_sigma_1_elongation_3.5/cell-16.tif', None),
        ('Gaussian_SNR=30/voxel_size_3._0.3_0.3/psf_sigma_0.1_elongation_2/data/cell_049.tif', 0.3),
        ('SNR=5.5/iterations=245/lambda=1e-5//voxel_size_2-0.1/psf_sigma=3.5_elongation=2/data/cell=25.0.tif', 0.1)
    )
    def test_voxel_size3(self, case):
        input_name, result = case
        metadata = Metadata(string=input_name)
        self.assertEqual(metadata['Voxel size x'], result)

    @data(
        ('input.tif', None),
        ('ImageJ_Lab_RIF/lambda=10e10/psf_sigma_1_elongation_3.5/cell-16.tif', None),
        ('Gaussian_SNR=30/voxel_size_3._0.3_0.3/psf_sigma_0.1_elongation_2/data/cell_049.tif', [3, 0.3, 0.3]),
        ('SNR=5.5/iterations=245/lambda=1e-5//voxel_size_2-0.1/psf_sigma=3.5_elongation=2/data/cell=25.0.tif', [2, 0.1, 0.1])
    )
    def test_resolution(self, case):
        input_name, result = case
        metadata = Metadata(string=input_name)
        test_res = metadata['resolution']
        if result is None:
            self.assertEqual(test_res, None)
        else:
            self.assertEqual(len(test_res), len(result))
            for i in range(len(result)):
                self.assertEqual(test_res[i], result[i])

    @data(
        ('input.tif', 'None'),
        ('ImageJ_Lab_RIF/lambda=10e10/psf_sigma_1_elongation_3.5/cell-16.tif', 'None'),
        ('gaussian_snr=30/voxel_size_3._0.3_0.3/psf_sigma_0.1_elongation_2/data/cell_049.tif', '30'),
        ('gaussian_snr=5.5/iterations=245/lambda=1e-5//voxel_size_2-0.1/psf_sigma=3.5_elongation=2/data/cell=25.0.tif', '5.5')
    )
    def test_snr(self, case):
        input_name, result = case
        metadata = Metadata(string=input_name)
        self.assertEqual(metadata['Gaussian_SNR'], result)

    @data(
        ('input.tif', None),
        ('ImageJ_Lab_RIF/lambda=0.0001/psf_sigma_1_elongation_3.5/cell-16.tif', 0.0001),
        ('Gaussian_SNR=30/voxel_size_3._0.3_0.3/psf_sigma_0.1_elongation_2/data/cell_049.tif', None),
        ('SNR=5.5/iterations=245/lambda=0.00001//voxel_size_2-0.1/psf_sigma=3.5_elongation=2/data/cell=25.0.tif', 10**(-5))
    )
    def test_lambda(self, case):
        input_name, result = case
        metadata = Metadata(string=input_name)
        self.assertEqual(metadata['Lambda'], result)

    @data(
        ('input.tif', None),
        ('Gaussian_SNR=30/voxel_size_3._0.3_0.3/psf_sigma_0.1_elongation_2/data/cell_049.tif', None),
        ('SNR=5.5/iterations=245/lambda=1e-5//voxel_size_2-0.1/psf_sigma=3.5_elongation=2/data/cell=25.0.tif', 245)
    )
    def test_iterations(self, case):
        input_name, result = case
        metadata = Metadata(string=input_name)
        self.assertEqual(metadata['Iterations'], result)

if __name__ == '__main__':
    unittest.main()
