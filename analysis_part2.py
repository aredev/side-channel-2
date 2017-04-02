import scipy.io
import numpy
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt
from random import randint

__author__ = "Tom Sandmann (s4330048) & Abdullah Rasool (s4350693)"

sbox = dict([
    (0, 12),
    (1, 5),
    (2, 6),
    (3, 11),
    (4, 9),
    (5, 0),
    (6, 10),
    (7, 13),
    (8, 3),
    (9, 14),
    (10, 15),
    (11, 8),
    (12, 4),
    (13, 7),
    (14, 1),
    (15, 2),
])


def read_input_file():
    """ Read the input trace file
    :return: array of all the input traces"""
    return scipy.io.loadmat('input.mat')['input']


def calculate_y(inputs):
    value_predication_matrix = numpy.zeros((2000, 15))

    row = 0
    keys = range(0, 15) # All possible keys k
    for in_value in inputs:
        in_value = in_value[0]
        for k in keys:
            x0 = r0 = randint(0, 15) # 4 bit random value
            in1 = in_value ^ r0
            x1 = in1 ^ k
            y0 = lookup_table_A(x0, x1)
            y1 = lookup_table_B(x0, x1)
            value_predication_matrix[row][k] = y0 ^ y1
        row += 1
    return value_predication_matrix


def lookup_table_A(x0, x1):
    bin_x0 = format(x0, '04b')
    bin_x1 = format(x1, '04b')
    return "a"

def lookup_table_B(x0, x1):
    # sbox[hex(x0 ^ x1)] ^ lookup_table_A(x0Z)
    return "b"


def calculate_y_no_masking(inputs):
    value_predication_matrix = numpy.zeros((2000, 15))

    row = 0
    keys = range(0, 15)  # All possible keys k
    for in_value in inputs:
        in_value = in_value[0]
        for k in keys:
            sin = in_value ^ k
            value_predication_matrix[row][k] = sbox[sin]
        row += 1
    return value_predication_matrix


def read_traces_file():
    """
    Returns the traces matrix
    :return: 2.000 rows with 10 elements/columns each 
    """
    return scipy.io.loadmat('leakage_y0_y1.mat')['L']


def preprocess_traces(traces):
    traces_prime = []
    for trace in traces:
        traces_prime.append(preprocess_row(trace))

    return numpy.asarray(traces_prime)


def preprocess_row(row):
    """
    Multiply an element with all elements after that element 
    :param row: 
    :return: Row with an element multiplied with all other elements, 45 in total (in accordance with n chooses k)
    """
    row_prime = []
    row = row.tolist()
    for t in row:
        start_index = row.index(t)+1
        for o in row[start_index:]:
            row_prime.append(t * o)
    return numpy.asarray(row_prime)


def correlation_analysis(p_traces, vpm):
    candidates = []

    for candidate in range(15):
        time_samples = []
        coefficients = []
        for sample in range(45):
            corcoeff = abs(pearsonr(vpm[:, candidate], p_traces[:, sample])[0])
            time_samples.append(sample)
            coefficients.append(corcoeff)
        candidates.append((candidate, time_samples, coefficients, max(coefficients)))

    sorted_candidates = sorted(candidates, key=lambda tup: tup[3], reverse=True)
    print("Sorted candidates: ")
    print("Candidate:\t\tCorrelation Value:")
    for c in sorted_candidates:
        print(str(c[0]), str(c[3]), sep="\t\t\t")

    highest_correlated_candidate = sorted_candidates[0]

    print("Possible candidate is: " + str(highest_correlated_candidate[0]) + " correlation value " + str(
        highest_correlated_candidate[3]))

    for p in sorted_candidates:
        if p[0] != highest_correlated_candidate[0]:
            plt.plot(p[1], p[2], 'r', label=p[0])
        else:
            plt.plot(p[1], p[2], 'g', label=p[0])

    plt.ylabel('Correlation')
    plt.xlabel('Samples')
    plt.legend()
    plt.show()


inputs = read_input_file()
traces = read_traces_file()
pp_traces = preprocess_traces(traces)
vpm = calculate_y_no_masking(inputs)
correlation_analysis(pp_traces, vpm)
# ppm = calculate_y(inputs)
# print(len(pp_traces))   # Still 2000 traces
# print(pp_traces[0]) # With 45 elements each
