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
        self.assertEqual(len(cell.image.shape), shape)

    @data(
        ([[1, 2, 3], [1, 1, 1]], 2),
        ([[1, 2, 3], [1, 1, 1], [1, 1, 1]], 3),
        ([[1, 2], [1, 1], [1, 1], [1, 1]], 4),
        ([[1, 2, 3]], 1)

    )
    def test_from_index_shape2(self, case):
        ind, shape = case
        cell = Cell(ind=ind)
        self.assertEqual(len(cell.image.shape), shape)

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
        for c in params:
            self.assertIn(c, cell.metadata)

    @data(
        (dict({'size': [5, 6, 5],
                'phi': 0,
                'theta': 0}), 0.3, 2908.88),
        # (dict({'size': [5, 6, 5]}), 0.3, 2908.88),
        # (dict({'size_x': 5, 'size_y': 6, 'size_z': 5}), 0.3, 2908.88)
    )
    def test_generate2(self, case):
        params, res, volume = case
        cell = Cell(input_voxel_size=res, **params)
        self.assertAlmostEqual(cell.volume()/volume, 1, 2)

    def test_spiky_cell(self):
        Cell(kind='spiky_cell', input_voxel_size=0.8, size=[5, 5, 5], phi=np.pi/4, theta=np.pi/2,
             spikiness=0.5, spike_size=0.1, spike_smoothness=0.1)

    def test_cell_from_params(self):
        celldata = CellParams(number_of_cells=5, spikiness_range=(0, 1), spike_size_range=(0.1, 1),
                              coordinates=False)
        cell = Cell(input_voxel_size=0.5, **dict(celldata.iloc[0]))
        self.assertIsNotNone(cell.image)

    @data(
            'ellipsoid',
            'spiky_cell'
        )
    def test_valid_types(self, kind):
        cell = Cell()
        cell.generate(kind=kind, input_voxel_size=3)
        self.assertIsNotNone(cell.image)

    @data(
        'ellipsoids',
        'invalid_shape'
    )
    def test_invalid_types(self, kind):
        self.assertRaises(AttributeError, CellParams, input_cell_kind=kind)

    def test_accuracy(self):
        cell = Cell()
        cell.generate(size=[5, 6, 5], input_voxel_size=0.5)
        errors = cell.compute_accuracy_measures(cell)
        for c in ['RMSE', 'NRMSE']:
            self.assertEqual(errors[c].iloc[0], 0)


if __name__ == '__main__':
    unittest.main()
