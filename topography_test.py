import numpy
import scipy
import sys
from matplotlib import pyplot
import pylab
from mpl_toolkits.mplot3d import Axes3D

# data_file = "3d_reconstruction_575_ones.txt.npy"
# data_file = "3d_reconstruction_185161_ones.txt.npy"
# data_file = "3d_reconstruction_77857_ones_2013-07-15_13-35-07.txt.npy"
# data_file = "3d_reconstruction_90974_ones_2013-07-15_13-49-21.txt.npy"
data_file = "3d_reconstruction_941864_ones_2013-07-19_14-45-49.txt.npy"
# data_file = "apple.npy"

def num_ones_in_matrix(matrix):
    ones = 0
    for num in matrix.flatten():
        if num == 1:
            ones += 1
    return ones


def test_print_matrix(mat):
    x, y = mat.shape
    for i in xrange(0, x):
        for j in xrange(0, y):
            sys.stdout.write(str(int(mat[i, j])))
        sys.stdout.write("\n")


def mat_to_coord_array(mat):
    result = []
    x, y, z = mat.shape
    for i in xrange(0, x):
        for j in xrange(0, y):
            for k in xrange(0, z):
                if mat[i, j, k] == 1:
                    result.append((i, j, k))
    return result


def display_compound_slices(mat):
    x, y, z = mat.shape
    test_matrix = numpy.zeros((x, y))
    three_d_point_array = mat_to_coord_array(mat)
    for i in xrange(0, x):
        for j in xrange(0, y):
            # for k in xrange(0, z):
                if mat[i, j, z / 4] == 1:
                    test_matrix[i, j] = 1
    # for (a, b, c) in three_d_point_array:
    #     test_matrix[b - 1, c - 1] = 1
    test_print_matrix(test_matrix)


def svr_test(mat):
    sparse_mat = scipy.sparse.csr_matrix



def display_3d(mat):
    fig = pylab.figure()
    ax = Axes3D(fig)
    coord_point_array = mat_to_coord_array(mat)
    sequence_containing_x_vals = [x for (x, y, z) in coord_point_array]
    sequence_containing_y_vals = [y for (x, y, z) in coord_point_array]
    sequence_containing_z_vals = [z for (x, y, z) in coord_point_array]
    ax.scatter(sequence_containing_x_vals, sequence_containing_y_vals, sequence_containing_z_vals)
    pyplot.show()


def main():
    test_load_matrix = numpy.load(data_file, None)
    print test_load_matrix.shape
    elements = 0
    elements = reduce(lambda x, y: x * y, test_load_matrix.shape)
    print elements
    ones = sum(test_load_matrix.flatten())
    print ones
    print "Sparsity = " + str(ones / elements * 100)
    # return
    # ones = num_ones_in_matrix(test_load_matrix)
    # print "Ones: " + str(ones) + " !!!!"
    # display_compound_slices(test_load_matrix)

    display_3d(test_load_matrix)
    # svr_test(test_load_matrix)



if __name__ == '__main__':
    main()
