import ipdb
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(vals, bins, fmt='{:5.1f}', with_counts=True, yscale='lin'):
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
    return s

def main(args):
    with open(args.infile, 'rb') as fp:
        vals = []
        for line in fp:
            try:
                vals.append(float(line.strip()))
            except ValueError:
                pass
    bins = args.bins
    if bins is None:
        n = args.num_bins
        v_min = args.min if args.min is not None else min(vals)
        v_max = args.max if args.max is not None else max(vals)
        bins = np.linspace(v_min, v_max, n+1)
    kwargs = {
            'fmt': args.fmt,
            'yscale': args.yscale,
            'with_counts': args.with_counts,
            }
    print plot(vals, bins, **kwargs)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', nargs='?', default='/dev/stdin')
    parser.add_argument('--bins', '-b', nargs='+')
    parser.add_argument('--fmt', '-f', default='{:5.1f}')
    parser.add_argument('--min', '-a', type=float)
    parser.add_argument('--max', '-z', type=float)
    parser.add_argument('--num-bins', '-n', type=int, default=10)
    parser.add_argument('--yscale', '-y', choices=['lin', 'log'], default='lin')
    parser.add_argument('--with-counts', action='store_false')
    args = parser.parse_args()
    main(args)

