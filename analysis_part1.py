import scipy.io
import numpy
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt

__author__ = "Tom Sandmann (s4330048) & Abdullah Rasool (s4350693)"

# Inverted S-box layer, courtesy of https://gist.github.com/bonsaiviking/5571001
Sbox_inv = (
            0x52, 0x09, 0x6A, 0xD5, 0x30, 0x36, 0xA5, 0x38, 0xBF, 0x40, 0xA3, 0x9E, 0x81, 0xF3, 0xD7, 0xFB,
            0x7C, 0xE3, 0x39, 0x82, 0x9B, 0x2F, 0xFF, 0x87, 0x34, 0x8E, 0x43, 0x44, 0xC4, 0xDE, 0xE9, 0xCB,
            0x54, 0x7B, 0x94, 0x32, 0xA6, 0xC2, 0x23, 0x3D, 0xEE, 0x4C, 0x95, 0x0B, 0x42, 0xFA, 0xC3, 0x4E,
            0x08, 0x2E, 0xA1, 0x66, 0x28, 0xD9, 0x24, 0xB2, 0x76, 0x5B, 0xA2, 0x49, 0x6D, 0x8B, 0xD1, 0x25,
            0x72, 0xF8, 0xF6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xD4, 0xA4, 0x5C, 0xCC, 0x5D, 0x65, 0xB6, 0x92,
            0x6C, 0x70, 0x48, 0x50, 0xFD, 0xED, 0xB9, 0xDA, 0x5E, 0x15, 0x46, 0x57, 0xA7, 0x8D, 0x9D, 0x84,
            0x90, 0xD8, 0xAB, 0x00, 0x8C, 0xBC, 0xD3, 0x0A, 0xF7, 0xE4, 0x58, 0x05, 0xB8, 0xB3, 0x45, 0x06,
            0xD0, 0x2C, 0x1E, 0x8F, 0xCA, 0x3F, 0x0F, 0x02, 0xC1, 0xAF, 0xBD, 0x03, 0x01, 0x13, 0x8A, 0x6B,
            0x3A, 0x91, 0x11, 0x41, 0x4F, 0x67, 0xDC, 0xEA, 0x97, 0xF2, 0xCF, 0xCE, 0xF0, 0xB4, 0xE6, 0x73,
            0x96, 0xAC, 0x74, 0x22, 0xE7, 0xAD, 0x35, 0x85, 0xE2, 0xF9, 0x37, 0xE8, 0x1C, 0x75, 0xDF, 0x6E,
            0x47, 0xF1, 0x1A, 0x71, 0x1D, 0x29, 0xC5, 0x89, 0x6F, 0xB7, 0x62, 0x0E, 0xAA, 0x18, 0xBE, 0x1B,
            0xFC, 0x56, 0x3E, 0x4B, 0xC6, 0xD2, 0x79, 0x20, 0x9A, 0xDB, 0xC0, 0xFE, 0x78, 0xCD, 0x5A, 0xF4,
            0x1F, 0xDD, 0xA8, 0x33, 0x88, 0x07, 0xC7, 0x31, 0xB1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xEC, 0x5F,
            0x60, 0x51, 0x7F, 0xA9, 0x19, 0xB5, 0x4A, 0x0D, 0x2D, 0xE5, 0x7A, 0x9F, 0x93, 0xC9, 0x9C, 0xEF,
            0xA0, 0xE0, 0x3B, 0x4D, 0xAE, 0x2A, 0xF5, 0xB0, 0xC8, 0xEB, 0xBB, 0x3C, 0x83, 0x53, 0x99, 0x61,
            0x17, 0x2B, 0x04, 0x7E, 0xBA, 0x77, 0xD6, 0x26, 0xE1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0C, 0x7D
            )


def read_output_file():
    """Returns the array of output data of size 10.000"""
    file = scipy.io.loadmat('output_data.mat')
    return file['output_data']


def read_traces_file():
    """Returns the multi dimensional array of traces, 10.000 rows, 2.000 columns"""
    file = scipy.io.loadmat('hardware_traces.mat')
    return file['traces']


def generate_all_keys():
    """
    :return: All 256 keys from 8 bits 
    """
    return range(0, 256)


def hamming_distance(s1, s2):
    """
    Calculate the hamming distance of two values, courtesty of https://en.wikipedia.org/wiki/Hamming_distance
    :param s1: input, converted to 8 bit 
    :param s2: input, converted to 8 bit
    :return: the hamming distance of s1 and s2
    """
    s1 = format(s1, '08b') # Convert to 8 bit binary
    s2 = format(s2, '08b') # Convert to 8 bit binary
    if len(s1) != len(s2):
        raise ValueError("Undefined for sequences of unequal length")
    return sum(el1 != el2 for el1, el2 in zip(s1, s2))


def create_hamming_distance_matrix(outputs, keys):
    """
    Create the hamming distance matrix given the output values and all the keys
    :param outputs: 
    :param keys: all the keys
    :return: Hamming distance matrix
    """
    hamming_distance_matrix = numpy.zeros((10000, 256)) # 10.000 rows, 256 columns
    row = 0
    for output in outputs:
        output = output[0]
        for k in keys:
            output_xor_k = k ^ output # returns oddly enough an array, first element is the xor value...
            s_in = Sbox_inv[output_xor_k]
            hd_value = hamming_distance(output, s_in)
            hamming_distance_matrix[row][k] = hd_value
        row += 1

    return hamming_distance_matrix


def create_column_wise_correlation(traces, hamming_distance_matrix):
    """
    Compute the correlation and show the plot
    :param traces: all the traces
    :param hamming_distance_matrix: 
    :return: shows the correlation graph, prints the possible key
    """
    candidates = []

    for candidate in range(256):
        time_samples = []
        coefficients = []
        for time_sample in range(2000):
            # pearsonnr() returns tuple, first element is correlation value, second is p-value.
            corcoef = abs(pearsonr(hamming_distance_matrix[:, candidate], traces[:, time_sample])[0])
            time_samples.append(time_sample)
            coefficients.append(corcoef)
        candidates.append((time_samples, coefficients, candidate, max(coefficients)))

    sorted_candidates = sorted(candidates, key=lambda tup: tup[3], reverse=True)
    print("Sorted candidates: ")
    print("Candidate:\t\tCorrelation Value:")
    for c in sorted_candidates:
        print(str(c[2]), str(c[3]), sep="\t\t\t")

    highest_correlated_candidate = sorted_candidates[0]

    print("Possible candidate is: " + str(highest_correlated_candidate[2]) + " correlation value " + str(highest_correlated_candidate[3]))

    for p in sorted_candidates:
        if p[2] != highest_correlated_candidate[2]:
            plt.plot(p[0], p[1], 'r', label=p[2])
        else:
            plt.plot(p[0], p[1], 'g', label=p[2])

    plt.ylabel('Correlation')
    plt.xlabel('Time')
    plt.legend()
    plt.show()


outputs = read_output_file()
traces = read_traces_file()
keys = generate_all_keys()
hdm = create_hamming_distance_matrix(outputs, keys)
create_column_wise_correlation(traces, hdm)
