import unittest

from ddt import ddt, data

from DeconvTest import PSF


@ddt
class TestPSFClass(unittest.TestCase):

    @data(
        (1, 3),
        (3, 1.5),
        (10, 2.5)
    )
    def test_measure_psf(self, case):
        sigma, aspect_ratio = case
        psf = PSF()
        psf.generate(sigma=sigma, aspect_ratio=aspect_ratio)
        sigmas = psf.measure_psf_sigma()
        self.assertEqual(len(sigmas), 3)
        self.assertAlmostEqual(sigma/sigmas[1], 1, 1)
        self.assertAlmostEqual(sigma/sigmas[2], 1, 1)
        self.assertAlmostEqual(sigma*aspect_ratio/sigmas[0], 1, 1)


if __name__ == '__main__':
    unittest.main()
