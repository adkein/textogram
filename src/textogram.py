#!/usr/bin/python

import ipdb
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(vals, bins=None, fmt='{:5.1f}', with_counts=True, display=True, yscale='lin'):
    bins = np.linspace(0, 1, 11) if bins is None else bins
    bins = np.linspace(min(vals), max(vals), 11)
    vals = np.array(vals).astype(float)
    vals = vals[(vals >= bins[0]) & (vals <= bins[-1])]
    bin_vals, left_edges, _ = plt.hist(vals, bins=bins)
    if yscale == 'log':
        bin_vals[bin_vals > 0] = np.log2(bin_vals[bin_vals > 0]) + 1
    s = '\n'
    h = max(bin_vals)
    for i in range(len(left_edges)-1):
        right_edge = fmt.format(left_edges[i+1]) if i < len(left_edges) - 1 else '*'
        s += fmt.format(left_edges[i]) + ' - ' + right_edge + ': '
        s += '#' * int(20 * bin_vals[i] / h) + '\n'
    if with_counts:
        n = len(vals)
        s += '\nitem count = ' + str(n)
        s += '\nmax_height_value = ' + str(int(h))
    if display:
        print s
    else:
        return s

def main(args):
    with open(args.infile, 'rb') as fp:
        vals = [float(line.strip()) for line in fp]
    plot(vals, fmt=args.fmt, yscale=args.yscale)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', nargs='?', default='/dev/stdin')
    parser.add_argument('--yscale', '-y', choices=['lin', 'log'], default='lin')
    parser.add_argument('--fmt', '-f', default='{:5.1f}')
    args = parser.parse_args()
    main(args)
