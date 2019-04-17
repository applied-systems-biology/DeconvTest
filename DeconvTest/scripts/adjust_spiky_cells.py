from DeconvTest.batch import simulation as sim
from helper_lib import filelib
from DeconvTest import Cell
import ipyvolume.pylab as p3
import numpy as np


def generate_cells_with_given_parameter_combination(path, number_of_cells, size_mean_and_std, equal_dimensions,
                                                    spikiness, spike_size, spike_smoothness):

    sim.generate_cell_parameters(outputfile=path + 'cell_parameters.csv',
                                 kind='spiky_cell',
                                 number_of_cells=number_of_cells,
                                 size_mean_and_std=size_mean_and_std,
                                 equal_dimensions=equal_dimensions,
                                 spikiness_range=(spikiness, spikiness),
                                 spike_size_range=(spike_size, spike_size),
                                 spike_smoothness_range=(spike_smoothness, spike_smoothness))

    sim.generate_cells_batch(params_file=path + 'cell_parameters.csv',
                             outputfolder=path + 'cells/',
                             resolution=0.2,
                             max_threads=6,
                             print_progress=False)

    for fn in filelib.list_subfolders(path + 'cells/'):
        cell = Cell(filename=path + 'cells/' + fn)
        p3.clear()
        p3.plot_isosurface(cell.image)
        p3.save(path + 'cells_html/' + fn[:-4] + '.html')


path = '../../../Data/test_spiky_cells/'
number_of_cells = 10
size_mean_and_std = (10, 2)
equal_dimensions = False


# test spike smoothness
spikiness = 0.2
spike_size = 0.5
for spike_smoothness in np.arange(0, 0.1, 0.01):
    generate_cells_with_given_parameter_combination(path + 'test_spike_smoothness/'
                                                    + str(round(spike_smoothness, 3)) + '/',
                                                    number_of_cells, size_mean_and_std, equal_dimensions,
                                                    spikiness, spike_size, spike_smoothness)

# test spike size
spikiness = 0.2
spike_smoothness = 0.03
for spike_size in np.arange(0, 1.5, 0.1):
    generate_cells_with_given_parameter_combination(path + 'test_spike_size/' + str(round(spike_size, 3)) + '/',
                                                    number_of_cells, size_mean_and_std, equal_dimensions,
                                                    spikiness, spike_size, spike_smoothness)

# test spikiness
spike_size = 0.5
spike_smoothness = 0.03
for spikiness in np.arange(0, 1.1, 0.1):
    generate_cells_with_given_parameter_combination(path + 'test_spikiness/' + str(round(spikiness, 3)) + '/',
                                                    number_of_cells, size_mean_and_std, equal_dimensions,
                                                    spikiness, spike_size, spike_smoothness)









