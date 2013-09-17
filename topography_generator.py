import numpy
import scipy
import sys
import math
# import threading
from matplotlib import pyplot
import pylab
from mpl_toolkits.mplot3d import Axes3D
from time import gmtime, strftime
import random
from scipy.spatial import distance
from sklearn.cluster import DBSCAN


# point_file_location = "/Library/Application Support/Google SketchUp 8/SketchUp/Plugins/face_mesh_points_dump.txt"
# point_file_location = "/Library/Application Support/Google SketchUp 8/SketchUp/Plugins/face_mesh_points_dump1.txt"

kRADIAL_SEARCH_RADIUS = 5


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


# Returns an array of indices surrounding a point if
# that index is a 1
def get_nearby_points(i, j, mat):
    nearby_points = []
    for k in xrange(i - kRADIAL_SEARCH_RADIUS, i + kRADIAL_SEARCH_RADIUS):
        for l in xrange(j - kRADIAL_SEARCH_RADIUS, j + kRADIAL_SEARCH_RADIUS):
            if 0 < k < mat.shape[0] and 0 < l < mat.shape[1] and mat[k, l] == 1:
                nearby_points.append((l, k))
    return nearby_points


def euclidean_distance(i, j, k, l):
    return vector_magnitude(i - k, j - l)


def manhattan_distance(i, j, k, l):
    return (i - k) + (j - l)


def vector_magnitude(i, j):
    return math.sqrt(i**2 + j**2)


#Given two points, it fills in the points connecting them
def connect_points(i, j, k, l, matrix_slice):
    step_vector = (i - k, j - l)
    step_vector_length = vector_magnitude(step_vector[0], step_vector[1])
    # print str(step_vector_length) + ", " + str(vector_magnitude(i, j)) + ", " + str(vector_magnitude(k, l))

    # step_vector_length = math.sqrt(step_vector[0]**2 + step_vector[1]**2)
    if step_vector_length != 0:
        step_vector = (step_vector[0] / step_vector_length, step_vector[1] / step_vector_length)
    start_vector = (i, j)
    while vector_magnitude(start_vector[0], start_vector[1]) < step_vector_length:
        a, b = (start_vector[0] + step_vector[0], start_vector[1] + step_vector[1])
        if 0 < a < matrix_slice.shape[0] and 0 < b < matrix_slice.shape[1]:
            matrix_slice[a, b] = 1
        start_vector = (a, b)


# Fills in the missing spaces in the relativly sparse matrices imported from
# sketchup that lose 
def coaxial_expansion(mat):
    for axis in xrange(0, len(mat.shape)):
        num_slices = mat.shape[axis]
        # print mat.mask(axis).shape
        for slice_index in xrange(0, num_slices):
            matrix_slice = numpy.zeros((0, 0))
            if axis == 0:
                matrix_slice = mat[slice_index, :, :]
            elif axis == 1:
                matrix_slice = mat[:, slice_index, :]
            elif axis == 2:
                matrix_slice = mat[:, :, slice_index]
            for i in xrange(0, matrix_slice.shape[0]):
                for j in xrange(0, matrix_slice.shape[1]):
                    if matrix_slice[i, j] == 1:
                        nearby_points = get_nearby_points(i, j, matrix_slice)
                        # thread1 = threading.Thread(target=connect_points_wrapper, args=(i, j, nearby_points, matrix_slice))
                        # thread1.start()
                        #connect_points_wrapper(i, j, matrix_slice)
                        for (k, l) in nearby_points:
                            connect_points(i, j, k, l, matrix_slice)
            if slice_index % 10 == 0:
                print "axis: " + str(axis) + ", index: " + str(slice_index)
    return mat


# Generates a 3d matrix of zeros and sets all indices equal to 1 where
# there is a point in the input data.
def make_pre_fill_matrix(points):
    min_max_xyz = min_max_3d(points)
    # print min_max_xyz
    # print len(points)
    # print min_max_xyz
    # print points[0]
    pre_expansion_matrix = numpy.zeros(tuple((min_max_xyz[i][1] for i in range(0, 3))))
    # print pre_expansion_matrix.shape
    # test_matrix = numpy.zeros(tuple((min_max_xyz[i][1] for i in range(0, 2))))
    # count = 0
    test_matrix = numpy.zeros((308, 104))

    for (x, y, z) in points:
        pre_expansion_matrix[x - 1, y - 1, z - 1] = 1
        # if z < 90:
            # break
        # test_matrix[y - 1, z - 1] = 1
        # if z < 90:
        #     break
    # test_print_matrix(test_matrix)
    return pre_expansion_matrix


# Returns the a tuple of (min_in_i^th_dimension, max_in_i^th_dimension)
# for a 3d list of points
def min_max_3d(point_array):
    return [(min(point_array, key=lambda x: x[i])[i], max(point_array, key=lambda x: x[i])[i]) for i in range(0, 3)]


# Scales data so that all indices are positive
# Decreases data values by an order of magnitude so that matrix isn't i000*j000*k000 in size
def scale_data(point_array):
    min_max_xyz = min_max_3d(point_array)
    new_points = []
    for dat in point_array:
        scaled_data = tuple(int((dat[i] - min_max_xyz[i][0]) / 10) for i in range(0, 3))
        # if scaled_data[0] == 0.00018699999964155722:
        #     print dat
        # print scaled_data
        # if scaled_data[0] > 5000:
        #     print scaled_data
        # else:
        #     new_points.append(scaled_data)
        new_points.append(scaled_data)
    return new_points


# Loads strings of 3d points from file and coverts to int tuple array
def get_points_from_file():
    point_file = open(point_file_location, "r")
    array = []
    for line in point_file:
        three_points = line.split(", ")
        # This parsing will have to be changed for points that do not have inch marks, eg
        # (-3726.125298", 314.072525", 820.749506")
        points_int_tuple = (float(three_points[0][1:-1]), float(three_points[1][:-1]), float(three_points[2][:-3]))
        array.append(points_int_tuple)
    return array


def display_3d(coord_point_array):
    fig = pylab.figure()
    ax = Axes3D(fig)
    sequence_containing_x_vals = [x for i, (x, y, z) in enumerate(coord_point_array) if i % 2 == 0]
    sequence_containing_y_vals = [y for i, (x, y, z) in enumerate(coord_point_array) if i % 2 == 0]
    sequence_containing_z_vals = [z for i, (x, y, z) in enumerate(coord_point_array) if i % 2 == 0]
    ax.scatter(sequence_containing_x_vals, sequence_containing_y_vals, sequence_containing_z_vals)
    pyplot.show()


# Calculates the mean of all values in a point array
def calculate_mean(coord_array):
    coord_sums = [(0) for i in range(0, len(coord_array[0]))]
    for i in xrange(0, len(coord_array)):
        for j in xrange(0, len(coord_sums)):
            coord_sums[j] += coord_array[i][j]
    coord_sums = tuple((coord_sums[i] / float(len(coord_array))) for i in range(0, len(coord_sums)))
    return coord_sums


# Generates the sigma covariance matrix = 1/m * (X - mu)(X - mu)^T
def calculate_covariance(mu, coord_array):
    covariance_matrix = numpy.zeros((len(coord_array[0]), len(coord_array[0])))
    m, n = covariance_matrix.shape
    for vector in coord_array:
        vec = [vector[i] - mu[i] for i in range(0, len(mu))]
        mmult = [[vec[i] * vec[j] for i in range(0, len(vec))] for j in range(0, len(vec))]
        for i in range(0, m):
            for j in range(0, n):
                covariance_matrix[i, j] += mmult[i][j]
    covariance_matrix /= len(coord_array)
    return covariance_matrix


def tuple_to_vector(tup):
    temp = numpy.zeros((len(tup), 1))
    for i in range(0, len(tup)):
        temp[i, 0] = tup[i]
    return temp


# Returns the value of a data point evaluated on a multivariate normal distribution
# with precalculated parameters
def multivariate_gaussian_probability(mu, sigma, training_example):
    training_example = tuple_to_vector(training_example)
    left = 1 / (math.pow((2 * math.pi), 1 / sigma.shape[0]) * math.sqrt(numpy.linalg.det(sigma)))
    tr_ex_minus_mean = training_example - tuple_to_vector(mu)
    right = -.5 * (numpy.transpose(tr_ex_minus_mean).dot(numpy.linalg.pinv(sigma)).dot(tr_ex_minus_mean))[0, 0]
    return left * math.exp(right)


# Calculates the mean and covariance matrices.
def gaussian_parameters(training_set):
    mu = calculate_mean(training_set)
    sigma = calculate_covariance(mu, training_set)
    return mu, sigma


def anomaly_detection(coord_array):
    # sample_training_set = [coord_array[random.randrange(0, len(coord_array))] for i in range(0, 50)]
    sample_training_set = random.sample(coord_array, 50)
    mu, sigma = gaussian_parameters(sample_training_set)
    probabilities_and_points = []
    for point in coord_array:
        probability = multivariate_gaussian_probability(mu, sigma, list(point))
        # probabilities_and_points.append(probability)
        probabilities_and_points.append((probability, point))
        # print probability
    # epsilon = determine_epsilon_cutoff(probabilities_and_points)
    result = []
    probabilities_and_points = sorted(probabilities_and_points, key=lambda x: x[0], reverse=True)
    probability_list = numpy.array([probabilities_and_points[i][0] for i in range(0, len(probabilities_and_points))])
    standard_deviation = numpy.std(probability_list, dtype=numpy.float64)
    mean = numpy.mean(probability_list)
    print standard_deviation
    print mean 
    # print 1/0
    for index, x in enumerate(numpy.array([probabilities_and_points[i][0] for i in range(0, len(probabilities_and_points))])):
        if x < mean - standard_deviation:
            print str(x) + ", less"
        else:
            print x
            result.append(x)
    print len(result)
    return coord_array[:len(result)]
    print 1/0
    print len([x > standard_deviation ** 3 for x in numpy.array([probabilities_and_points[i][0] for i in range(0, len(probabilities_and_points))])])
    print 1/0
    non_anomalous_points = [probabilities_and_points[i][1] for i in range(0, len(probabilities_and_points) / 2)]
    mu, sigma = gaussian_parameters(non_anomalous_points)
    epsilon = determine_epsilon_cutoff(mu, [multivariate_gaussian_probability(mu, sigma, list(point)) for point in non_anomalous_points])


def distance_between_points(point_1, point_2):
    # print str(point_1) + " | " + str(point_2)
    return math.sqrt(sum([(point_1[i] - point_2[i]) ** 2 for i in range(0, len(point_1))]))


def get_nearest_centroid(point, centroids):
    centroid_distances = [distance_between_points(point, centroid) for centroid in centroids]
    min_distance = min(centroid_distances)
    min_index = 0
    for i in range(0, len(centroid_distances)):
        if centroid_distances[i] == min_distance:
            return i


def k_means_clustering(clusters, point_tuple_array):
    centroids = random.sample(point_tuple_array, clusters)
    iterations = 20
    point_centroid_map = {}
    for i in xrange(0, iterations):
        print i
        for point in point_tuple_array:
            nearest_centroid = get_nearest_centroid(point, centroids)
            point_centroid_map[point] = nearest_centroid
        # centroids = [0 for i in range(0, clusters)]
        centroid_counts = [0 for i in range(0, clusters)]
        for point in point_centroid_map:
            centroid_index = point_centroid_map[point]
            # print centroid_index
            # print centroids
            centroids[centroid_index] = tuple([point[i] + centroids[centroid_index][i] for i in range(0, len(point))])
            centroid_counts[centroid_index] += 1
        # for clus in centroids:
        # print "Iteration " + str(i) + " sizes: " + str(centroid_counts) #" Clusters: " + str(len(clus[0])) + ", " + str(len(clus[1])) + " " + str(len(clus[2]))
        print "sizes: " + str(centroid_counts) #" Clusters: " + str(len(clus[0])) + ", " + str(len(clus[1])) + " " + str(len(clus[2]))
        # print centroid_counts
        # print centroids
        for j, centroid in enumerate(centroids):
            new_point = []
            for val in centroid:
                new_point.append(val / float(centroid_counts[j]))
            centroids[j] = tuple(new_point)

        # centroids = [(centroids[i][j] / float(centroid_counts[i])) for j in range(0, len(centroids[i])) for i in range(0, clusters)]
        # print centroids
        # return []
    return point_centroid_map


def main():

    point_tuple_array = get_points_from_file()
    # # print len(point_tuple_array)
    # anomaly_detection(point_tuple_array)
    # return

    point_tuple_array = scale_data(point_tuple_array)

    # num_elements = 1
    # for point in min_max_3d(point_tuple_array):
    #     num_elements *= point[1]
    # # num_elements = reduce(lambda x, y: x * y, min_max_3d(point_tuple_array))
    # print len(point_tuple_array)
    # print num_elements
    # print len(point_tuple_array) / float(num_elements) * 100
    # # display_3d(point_tuple_array)
    # return
    point_tuple_array = list(set(point_tuple_array))
    point_tuple_array.sort(key=lambda x: x[2])
    point_tuple_array.reverse( )

    # point_cluster_map = k_means_clustering(2, point_tuple_array)

    # for point in point_cluster_map:
    #     print str(point) + " - " + str(point_cluster_map[point])
    # point_tuple_array = anomaly_detection(point_tuple_array)
    # print len(point_tuple_array)
    # display_3d(point_tuple_array)
    # return

    # print len(point_tuple_array)
    # return
    # for point in point_tuple_array:
    #     print point
    # return
    three_d_block_representation = make_pre_fill_matrix(point_tuple_array)
    coaxial_expansion_matrix = coaxial_expansion(three_d_block_representation)
    ones = num_ones_in_matrix(coaxial_expansion_matrix)
    time = strftime("%Y-%m-%d_%H-%M-%S")
    print "ones after: " + str(ones)
    filename = "3d_reconstruction_" + str(ones) + "_ones_" + time + ".txt"
    numpy.save(filename, coaxial_expansion_matrix)


if __name__ == '__main__':
    main()









