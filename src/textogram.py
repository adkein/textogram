#!/usr/bin/python

import ipdb
import numpy as np
import matplotlib.pyplot as plt

def textogram(vals, bins=None, fmt='{:5.1f}', with_counts=True):
    bins = np.linspace(0, 1, 11) if bins is None else bins
    vals = np.array(vals)
    vals = vals[(vals >= bins[0]) & (vals <= bins[-1])]
    bin_vals, left_edges, _ = plt.hist(vals, bins=bins)
    s = ''
    h = max(bin_vals)
    for i in range(len(left_edges)-1):
        right_edge = fmt.format(left_edges[i+1]) if i < len(left_edges) - 1 else '*'
        s += fmt.format(left_edges[i]) + ' - ' + right_edge + ': '
        s += '#' * int(20 * bin_vals[i] / h) + '\n'
    if with_counts:
        n = len(vals)
        s += '\nitem count = ' + str(n)
        s += '\nmax_height_value = ' + str(int(h))
        s += '\n'
    return s

def main(args):
    with open(args.infile, 'rb') as fp:
        vals = fp.readlines()
    print textogram(vals, fmt=args.fmt)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', default='/dev/stdin')
    parser.add_argument('--fmt', '-f', default='{:5.1f}')
    args = parser.parse_args()
    main(args)
