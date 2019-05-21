import unittest

from ddt import ddt, data
import shutil
import os

from DeconvTest import CellParams


@ddt
class TestCellData(unittest.TestCase):

    @data(
        (8, 2),
        [10, 1]
    )
    def test_cell_data(self, size):
        celldata = CellParams(number_of_cells=1, size_mean_and_std=size)
        celldata.save('data/celldata.csv')
        celldata.plot_size_distribution()
        shutil.rmtree('data/')

    def test_read_write(self):
        celldata = CellParams(number_of_cells=5, spikiness_range=(0, 0.5), spike_size_range=(0.1, 1))
        celldata.save('data/celldata.csv')
        celldata2 = CellParams()
        celldata2.read_from_csv('data/celldata.csv')
        self.assertEqual(len(celldata), len(celldata2))
        for c in celldata.columns:
            self.assertEqual(c in celldata2.columns, True)
        shutil.rmtree('data/')

    @data(
            'ellipsoid',
            'spiky_cell'
        )
    def test_valid_types(self, kind):
        celldata = CellParams(kind=kind)
        celldata.save('data/celldata.csv')
        self.assertEqual(os.path.exists('data/celldata.csv'), True)
        shutil.rmtree('data/')

    @data(
        'ellipsoids',
        'invalid_shape'
    )
    def test_invalid_types(self, kind):
        self.assertRaises(AttributeError, CellParams, input_cell_kind=kind)


if __name__ == '__main__':
    unittest.main()


