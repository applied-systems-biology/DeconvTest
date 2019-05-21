import unittest

import os
import numpy as np
import pandas as pd
from ddt import ddt

from DeconvTest import Metadata


@ddt
class TestMetadataFromFilename(unittest.TestCase):

    def test_metadata(self):
        data = pd.Series({"Voxel size x": 2,
                          'Voxel size y': 3,
                          'Voxel size z': 5,
                          'aaa': 'afgw'})
        data.to_csv('test.csv', sep='\t', header=False)
        metadata = Metadata(filename='test.csv')
        self.assertEqual(metadata['Voxel size'], '[5. 3. 2.]')
        os.remove('test.csv')

    def test_metadata2(self):
        data = pd.Series({"Voxel size x": 2,
                          'Voxel size y': 3,
                          'aaa': 'afgw'})
        data.to_csv('test.csv', sep='\t', header=False)
        metadata = Metadata(filename='test.csv')
        self.assertNotIn('Voxel size', metadata.index)
        os.remove('test.csv')

    def test_metadata3(self):
        data = pd.Series({"Voxel size x": 2,
                          'Voxel size y': 3,
                          'Voxel size': [5, 3, 2],
                          'aaa': 'afgw'})
        data.to_csv('test.csv', sep='\t', header=False)
        metadata = Metadata(filename='test.csv')
        self.assertEqual(metadata['Voxel size'], '[5, 3, 2]')
        os.remove('test.csv')

    def test_metadata4(self):
        data = pd.Series()
        data.to_csv('test.csv', sep='\t', header=False)
        metadata = Metadata(filename='test.csv')
        self.assertNotIn('Voxel size', metadata.index)
        os.remove('test.csv')


if __name__ == '__main__':
    unittest.main()
