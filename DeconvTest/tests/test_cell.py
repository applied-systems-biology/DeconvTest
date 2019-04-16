import unittest

from ddt import ddt, data
import numpy as np


from DeconvTest import Cell
from DeconvTest import CellParams


@ddt
class TestCellClass(unittest.TestCase):

    @data(
        [[1, 2, 3], [1, 1, 1], [1, 1, 1, 5]],
        [[1, 2, 3], [1, 1], [1, 1, 1]],
        [[1, 2], [1, 1, 1], [1, 1, 1]]

    )
    def test_from_index_length(self, ind):
        cell = Cell()
        self.assertRaises(IndexError, cell.from_index, ind)

    @data(
        ([[1, 2, 3], [1, 1, 1]], 2),
        ([[1, 2, 3], [1, 1, 1], [1, 1, 1]], 3),
        ([[1, 2], [1, 1], [1, 1], [1, 1]], 4),
        ([[1, 2, 3]], 1)

    )
    def test_from_index_shape(self, case):
        cell = Cell()
        ind, shape = case
        cell.from_index(ind)
        self.assertEquals(len(cell.image.shape), shape)

    @data(
        ([[1, 2, 3], [1, 1, 1]], 2),
        ([[1, 2, 3], [1, 1, 1], [1, 1, 1]], 3),
        ([[1, 2], [1, 1], [1, 1], [1, 1]], 4),
        ([[1, 2, 3]], 1)

    )
    def test_from_index_shape2(self, case):
        ind, shape = case
        cell = Cell(ind=ind)
        self.assertEquals(len(cell.image.shape), shape)

    @data(
        (dict({'size': [5, 6, 5],
                'phi': 0,
                'theta': 0}), 0.3, 2908.88),
        (dict({'size': [5, 6, 5]}), 0.3, 2908.88)
    )
    def test_generate(self, case):
        params, res, volume = case
        cell = Cell()
        cell.generate(res, **params)
        self.assertAlmostEqual(cell.volume()/volume, 1, 2)

    @data(
        (dict({'size': [5, 6, 5],
                'phi': 0,
                'theta': 0}), 0.3, 2908.88),
        (dict({'size': [5, 6, 5]}), 0.3, 2908.88),
        (dict({'size_x': 5, 'size_y': 6, 'size_z':5}), 0.3, 2908.88)
    )
    def test_generate2(self, case):
        params, res, volume = case
        cell = Cell(resolution=res, **params)
        self.assertAlmostEqual(cell.volume()/volume, 1, 2)

    def test_spiky_cell(self):
        Cell(kind='spiky_cell', resolution=0.3, size=[10, 10, 10], phi=np.pi/4, theta=np.pi/2,
             spikiness=0.5, spike_size=1, spike_smoothness=0.1)

    def test_segment(self):
        img = Cell()
        arr = np.zeros([50, 50, 50])
        arr[10:-10, 10:-10, 10:-10] = 255
        img.image = arr
        img.segment()
        self.assertEqual(np.sum(np.abs(img.image - arr)), 0)

    @data(
        (np.ones([10, 10, 10]), [10, 10, 10]),
        (np.zeros([10, 10, 10]), [0, 0, 0]),
        (np.array([[[0, 0, 0, 0],
                    [0, 1, 1, 1],
                    [0, 0, 0, 0]],
                  [[0, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 0, 0]],
                  [[0, 0, 0, 0],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0]]]
                  ), [2, 1, 3]),
        (np.array([[[0, 0, 1, 1],
                    [0, 1, 1, 0],
                    [1, 1, 0, 0]],
                  [[0, 0, 1, 1],
                   [0, 0, 1, 0],
                   [1, 0, 0, 0]]]
                  ), [2, 3, 4])
    )
    def test_dimensions(self, case):
        img, dim = case
        cell = Cell()
        cell.image = img
        self.assertEqual(tuple(cell.dimensions()), tuple(dim))

    def test_dimensions_input_is_binary(self):
        cell = Cell()
        cell.image = np.arange(8).reshape((2, 2, 2))
        self.assertRaises(ValueError, cell.dimensions)

    def test_dimensions_image_is_None(self):
        cell = Cell()
        self.assertRaises(ValueError, cell.dimensions)

    def test_overlap_error(self):
        cell = Cell()
        cell.generate(size=[5, 6, 5], resolution=0.5)
        errors = cell.compare_to_ground_truth(cell)
        for c in ['Overdetection error', 'Underdetection error', 'Overlap error']:
            self.assertEqual(errors[c].iloc[0], 0)
        for c in ['Jaccard index', 'Sensitivity', 'Precision']:
            self.assertEqual(errors[c].iloc[0], 1)

    def test_cell_from_params(self):
        celldata = CellParams(number_of_cells=5, spikiness_range=(0, 1), spike_size_range=(0.1, 1),
                              coordinates=False)
        cell = Cell(resolution=0.5, **dict(celldata.iloc[0]))
        self.assertIsNotNone(cell.image)

    @data(
            'ellipsoid',
            'spiky_cell'
        )
    def test_valid_types(self, kind):
        cell = Cell()
        cell.generate(kind=kind, resolution=0.5)
        self.assertIsNotNone(cell.image)

    @data(
        'ellipsoids',
        'invalid_shape'
    )
    def test_invalid_types(self, kind):
        self.assertRaises(AttributeError, CellParams, kind=kind)


if __name__ == '__main__':
    unittest.main()
