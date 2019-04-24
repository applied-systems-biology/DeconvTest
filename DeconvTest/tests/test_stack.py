import unittest

import numpy as np
import pandas as pd
from ddt import ddt, data
from scipy import ndimage

from DeconvTest import Stack
from DeconvTest import Cell
from DeconvTest import CellParams


@ddt
class TestStackClass(unittest.TestCase):
    def test_empty_arguments(self):
        img = Stack()
        for var in ['image','filename']:
            self.assertIn(var, img.__dict__)
            self.assertEqual(img.__dict__[var], None)

    def test_position_cell(self):
        stack = Stack()
        stack.image = np.zeros([10, 10, 10])
        ind = np.where(np.ones([3, 3, 3]) > 0)
        cell = Cell(ind=ind)
        cell.position = (1, 1, 1)
        stack.position_cell(cell)
        self.assertEqual(np.sum(stack.image > 1), 27)

    def test_position_cell2(self):
        stack = Stack()
        stack.image = np.zeros([10, 10, 10])
        ind = np.where(np.ones([3, 3, 3]) > 0)
        cell = Cell(ind=ind)
        cell.position = (1, 1, 1)
        stack.position_cell(cell)
        cell.position = (5, 5, 5)
        stack.position_cell(cell)

    def test_position_cell3(self):
        stack = Stack()
        stack.image = np.zeros([10, 6, 6])
        cell = Cell()
        cell.image = np.zeros([15, 8, 8])
        cell.image[8:10, 2:-1, 2:-1] = 255
        cell.position = ndimage.center_of_mass(np.ones_like(stack.image))
        stack.position_cell(cell)
        self.assertEqual(np.sum(stack.image > 0), np.sum(cell.image > 0))

    @data(
        (pd.DataFrame({'size_x': [5],
                   'size_y': [6],
                    'size_z': [5],
                    'x': [0.5],
                    'y': [0.5],
                    'z': [0.5]}), 1, [10, 10, 10], 78.54),
        (pd.DataFrame({'size_x': [5],
                      'size_y': [6],
                      'size_z': [5],
                      'x': [0.0],
                      'y': [1],
                      'z': [1]}), 1, [10, 10, 10], 78.54)
    )
    def test_from_params(self, case):
        params, res, stacksize, volume = case
        stack = Stack()
        stack.generate(params, res, stacksize)
        self.assertAlmostEqual(np.sum(stack.image > 0)/volume, 1, 1)

    def test_is_segmented(self):
        stack = Stack()
        stack.image = np.array([[[0, 0, 2, 5],
                    [0, 0, 1, 0],
                    [10, 0, 0, 0]],
                  [[0, 0, 1, 1],
                   [0, 0, 0, 0],
                   [2, 0, 0, 0]],
                  [[0, 0, 0, 0],
                   [0, 0, 0, 0],
                   [15, 2, 0, 0]]]
                  )
        self.assertEqual(stack.is_segmented, False)
        self.assertEqual(stack.is_labeled, False)
        stack.segment()
        self.assertEqual(stack.is_segmented, True)
        self.assertEqual(stack.is_labeled, True)

    @data(
        (np.array([[[0, 0, 2, 5],
                    [0, 0, 0, 0],
                    [10, 0, 0, 0]],
                  [[0, 0, 1, 1],
                   [0, 0, 0, 0],
                   [2, 0, 0, 5]],
                  [[0, 0, 0, 0],
                   [0, 0, 0, 0],
                   [15, 2, 0, 0]]]
                  ), 2)
    )
    def test_segmentation_preprocess(self, case):
        img, num = case
        stack = Stack()
        stack.image = img
        stack.segment(thr=0, preprocess=True)
        self.assertEqual(len(stack.cells), num)

    @data(
        (np.array([[[0, 0, 2, 5],
                    [0, 0, 1, 0],
                    [10, 0, 0, 0]],
                  [[0, 0, 1, 1],
                   [0, 0, 0, 0],
                   [2, 0, 0, 0]],
                  [[0, 0, 0, 0],
                   [0, 0, 0, 0],
                   [15, 2, 0, 0]]]
                  ), 2, [4, 5]),
        (np.array([[[0, 0, 2, 5],
                    [0, 0, 0, 0],
                    [10, 0, 0, 0]],
                  [[0, 0, 1, 1],
                   [0, 0, 0, 0],
                   [2, 0, 0, 5]],
                  [[0, 0, 0, 0],
                   [0, 0, 0, 0],
                   [15, 2, 0, 0]]]
                  ), 3, [4, 4, 1])
    )
    def test_segmentation(self, case):
        img, num, sizes = case
        stack = Stack()
        stack.image = img
        stack.segment(thr=0)
        self.assertEqual(len(stack.cells), num)
        sizes.sort()
        cellsizes = []
        for cell in stack.cells:
            cellsizes.append(cell.volume())
        cellsizes.sort()
        self.assertEqual(tuple(cellsizes), tuple(sizes))

    @data(
        (np.array([[[0, 0, 2, 5],
                    [0, 0, 1, 0],
                    [10, 0, 0, 0]],
                  [[0, 0, 1, 1],
                   [0, 0, 0, 0],
                   [2, 0, 0, 0]],
                  [[0, 0, 0, 0],
                   [0, 0, 0, 0],
                   [15, 2, 0, 0]]]
                  ), 2, [4, 2]),
        (np.array([[[0, 0, 2, 5],
                    [0, 0, 0, 0],
                    [10, 0, 0, 0]],
                  [[0, 0, 1, 1],
                   [0, 0, 0, 0],
                   [2, 0, 0, 5]],
                  [[0, 0, 0, 0],
                   [0, 0, 0, 0],
                   [15, 2, 0, 0]]]
                  ), 3, [4, 2, 1])
    )
    def test_segmentation_thr1(self, case):
        img, num, sizes = case
        stack = Stack()
        stack.image = img
        stack.segment(thr=1)
        self.assertEqual(len(stack.cells), num)
        sizes.sort()
        cellsizes = []
        for cell in stack.cells:
            cellsizes.append(cell.volume())
        cellsizes.sort()
        self.assertEqual(tuple(cellsizes), tuple(sizes))

    def test_split_to_cells_nonsegmented(self):
        img = np.array([[[0, 0, 1, 1],
                    [0, 1, 1, 0],
                    [1, 1, 0, 0]],
                  [[0, 0, 1, 1],
                   [0, 0, 1, 0],
                   [1, 0, 0, 0]]]
                  )
        stack = Stack()
        stack.image = img
        self.assertRaises(ValueError, stack.split_to_cells)

    def test_split_to_cells_image_is_None(self):
        stack = Stack()
        stack.is_segmented = True
        self.assertRaises(ValueError, stack.split_to_cells)

    @data(
        (np.ones([10, 10, 10]), [[10, 10, 10]]),
        (np.zeros([10, 10, 10]), [[0, 0, 0]]),
        (np.array([[[0, 0, 1, 1],
                    [0, 0, 1, 0],
                    [2, 0, 0, 0]],
                  [[0, 0, 1, 1],
                   [0, 0, 0, 0],
                   [2, 0, 0, 0]],
                  [[0, 0, 0, 0],
                   [0, 0, 0, 0],
                   [2, 2, 0, 0]]]
                  ), [[2, 2, 2], [3, 1, 2]]),
        (np.array([[[0, 0, 1, 1],
                    [0, 1, 1, 0],
                    [1, 1, 0, 0]],
                  [[0, 0, 1, 1],
                   [0, 0, 1, 0],
                   [1, 0, 0, 0]]]
                  ), [[2, 3, 4]])
    )
    def test_dimensions(self, case):
        img, dim = case
        dim = np.array(dim)
        stack = Stack()
        stack.image = img
        stack.is_segmented = True
        stack.split_to_cells()
        self.assertEqual(tuple(stack.dimensions().flatten()), tuple(dim.flatten()))

    def test_compare_to_ground_truth(self):
        stack = Stack()
        params = pd.DataFrame({'size_x': [5, 5], 'size_y': [6, 5], 'size_z': [5, 5],
                               'x': [0.5, 0.1], 'y': [0.5, 0.1], 'z': [0.5, 0.1]})
        stack.generate(params, 0.5, [10, 10, 10])
        stack.split_to_cells()
        errors = stack.compute_binary_accuracy_measures(stack)
        for i in range(len(errors)):
            self.assertEqual(errors.iloc[i]['Overlap error'], 0)
            self.assertEqual(errors.iloc[i]['Overdetection error'], 0)
            self.assertEqual(errors.iloc[i]['Underdetection error'], 0)
            self.assertEqual(errors.iloc[i]['Jaccard index'], 1)
            self.assertEqual(errors.iloc[i]['Sensitivity'], 1)
            self.assertEqual(errors.iloc[i]['Precision'], 1)

    def test_compare_to_ground_truth2(self):
        stack = Stack()
        params = pd.DataFrame({'size_x': [5], 'size_y': [6], 'size_z': [5],
                               'x': [0.5], 'y': [0.5], 'z': [0.5]})
        stack.generate(params, 0.5, [10, 10, 10])
        stack.split_to_cells()
        vol = np.prod(stack.image.shape)
        volumes = []
        for i in range(len(stack.cells)):
            volumes.append((vol - stack.cells[i].volume())/stack.cells[i].volume())
        volumes.sort()

        stack.generate(params, 0.5, [10, 10, 10])
        stack2 = Stack()
        stack2.image = np.ones_like(stack.image)
        stack2.is_segmented = True
        stack2.split_to_cells()
        errors = list(stack2.compute_binary_accuracy_measures(stack)['Overlap error'])
        errors.sort()
        self.assertEqual(tuple(volumes), tuple(errors))

    def test_compare_to_ground_truth3(self):
        stack = Stack()
        params = pd.DataFrame({'size_x': [5], 'size_y': [6], 'size_z': [5],
                               'x': [0.5], 'y': [0.5], 'z': [0.5]})
        stack.generate(params, 0.5, [10, 10, 10])
        stack2 = Stack()
        stack2.image = np.zeros_like(stack.image)
        stack2.is_segmented = True
        stack2.split_to_cells()
        errors = stack2.compute_binary_accuracy_measures(stack)
        for i in range(len(errors)):
            self.assertEqual(errors.iloc[i]['Overlap error'], 1)
            self.assertEqual(errors.iloc[i]['Overdetection error'], 0)
            self.assertEqual(errors.iloc[i]['Underdetection error'], 1)
            self.assertEqual(errors.iloc[i]['Jaccard index'], 0)
            self.assertEqual(errors.iloc[i]['Sensitivity'], 0)
            self.assertEqual(errors.iloc[i]['Precision'], 0)

    def test_from_stack_params(self):
        params = CellParams(number_of_stacks=1, number_of_cells=5)
        stack = Stack(cell_params=params[params['stack'] == 0], input_voxel_size=0.5, stack_size=[10, 10, 10])
        self.assertIsNotNone(stack.image)

    def test_from_stack_params2(self):
        params = pd.DataFrame({'size_x': [10, 10, 10],
                               'size_y': [9, 10, 8],
                               'size_z': [10, 11, 10],
                               'phi': [0, np.pi / 4, np.pi / 2],
                               'theta': [0, 0, 0],
                               'x': [0.1, 0.8, 0.2],
                               'y': [0.5, 0.5, 0.5],
                               'z': [0.2, 0.5, 0.8],
                               'spikiness': [0, 0, 100],
                               'spike_size': [0, 0, 1],
                               'spike_smoothness': [0.05, 0.05, 0.05]})
        stack = Stack(cell_params=params, input_voxel_size=0.5, stack_size=[50, 50, 50])
        self.assertIsNotNone(stack.image)

    def test_range_for_number_of_cells(self):
        params = CellParams(number_of_stacks=3, number_of_cells=[5, 10])
        stack = Stack(cell_params=params[params['stack'] == 2].reset_index(), input_voxel_size=0.5, stack_size=[10, 10, 10])
        self.assertIsNotNone(stack.image)


if __name__ == '__main__':
    unittest.main()
