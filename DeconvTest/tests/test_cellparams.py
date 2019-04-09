import unittest

from ddt import ddt, data
import shutil

from DeconvTest import CellParams


@ddt
class TestCellData(unittest.TestCase):

    def test_cell_data(self):
        celldata = CellParams()
        celldata.save('data/celldata.csv')
        pl = celldata.plot_size_distribution()
        pl.savefig('data/celldata_size.png')
        pl = celldata.plot_angle_distribution()
        pl.savefig('data/celldata_angle.png')
        shutil.rmtree('data/')

    @data(
        (8, 2),
        [10, 1]
    )
    def test_cell_data(self, size):
        celldata = CellParams(size_mean_and_std=size)
        celldata.save('data/celldata.csv')
        pl = celldata.plot_size_distribution()
        pl.savefig('data/celldata_size.png')
        pl = celldata.plot_angle_distribution()
        pl.savefig('data/celldata_angle.png')
        shutil.rmtree('data/')

    def test_cell_data100(self):
        celldata = CellParams(number_of_cells=100)
        celldata.save('data/celldata1.csv')
        pl = celldata.plot_size_distribution()
        pl.savefig('data/celldata1_size.png')
        pl = celldata.plot_angle_distribution()
        pl.savefig('data/celldata1_angle.png')
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


if __name__ == '__main__':
    unittest.main()


