#!/usr/bin/env python3

from __future__ import print_function
from math import log
from typing import List, Optional, Union
import sys
import os


'''
A script to read in a list of numbers and output an ASCII histogram of them.
Various optional arguments control the format of the output.
'''


def textogram(vals: List[Union[str, float]], **kwargs) -> None:
    """Print a textogram (ASCII histogram) of the given values."""
    try:
        result = get_textogram(vals, **kwargs)
        print(result)
    except Exception as e:
        print(f"Error generating textogram: {e}", file=sys.stderr)
        sys.exit(1)


def linspace(a: float, b: float, n: int) -> List[float]:
    """Generate n evenly spaced points between a and b."""
    if n < 2:
        raise ValueError("n must be at least 2 for linspace")
    if a == b:
        return [a] * n
    
    dx = float(b - a) / (n - 1)
    return [a + i * dx for i in range(n)]


def get_textogram(vals: List[Union[str, float]], 
                  bins: Optional[List[float]] = None, 
                  N: int = 10, 
                  xmin: Optional[float] = None, 
                  xmax: Optional[float] = None,
                  with_counts: bool = True, 
                  yscale: str = 'lin', 
                  height: int = 60, 
                  char: str = '#',
                  categorical: bool = False) -> str:
    """
    Generate an ASCII histogram of the given values.
    
    Args:
        vals: List of values to histogram
        bins: Custom bin edges (optional)
        N: Number of bins (default 10)
        xmin: Minimum value for binning
        xmax: Maximum value for binning  
        with_counts: Show counts and percentages
        yscale: 'lin' or 'log' scaling
        height: Maximum bar height in characters
        char: Character to use for bars
        categorical: Treat data as categorical
        
    Returns:
        String representation of the histogram
    """
    # Input validation
    if not isinstance(vals, list):
        raise TypeError("vals must be a list")
    
    if len(vals) == 0:
        return 'No data provided.'
    
    if N < 1:
        raise ValueError("Number of bins (N) must be positive")
        
    if height < 1:
        raise ValueError("Height must be positive")
        
    if yscale not in ['lin', 'log']:
        raise ValueError("yscale must be 'lin' or 'log'")
        
    if not char:
        char = '#'  # Default fallback
    
    # Filter out None values and handle empty result
    vals = [v for v in vals if v is not None]
    if len(vals) == 0:
        return 'No valid data after filtering.'
    
    try:
        if categorical:
            return _generate_categorical_histogram(vals, height, char, with_counts)
        else:
            return _generate_numerical_histogram(
                vals, bins, N, xmin, xmax, with_counts, yscale, height, char
            )
    except Exception as e:
        raise RuntimeError(f"Failed to generate histogram: {e}")


def _generate_categorical_histogram(vals: List[str], height: int, char: str, with_counts: bool) -> str:
    """Generate histogram for categorical data."""
    categories = sorted(set(vals))
    if not categories:
        return 'No categories found.'
        
    bin_counts = [len([v for v in vals if v == c]) for c in categories]
    bin_vals = bin_counts.copy()  # For categorical, these are the same
    
    # Create labels with consistent formatting
    labels = [str(c) for c in categories]
    if labels:
        max_label_length = max(len(label) for label in labels)
        labels = [f'{label:>{max_label_length}}' for label in labels]
    
    return _format_histogram_output(bin_vals, bin_counts, labels, len(vals), height, char, with_counts)


def _generate_numerical_histogram(vals: List[Union[str, float]], 
                                 bins: Optional[List[float]], 
                                 N: int, 
                                 xmin: Optional[float], 
                                 xmax: Optional[float],
                                 with_counts: bool, 
                                 yscale: str, 
                                 height: int, 
                                 char: str) -> str:
    """Generate histogram for numerical data."""
    # Convert to float with error handling
    numeric_vals = []
    for v in vals:
        try:
            numeric_vals.append(float(v))
        except (ValueError, TypeError):
            # Skip non-numeric values silently
            continue
    
    if not numeric_vals:
        return 'No valid numeric data found.'
    
    # Determine min/max values safely
    try:
        v_min = xmin if xmin is not None else min(numeric_vals)
        v_max = xmax if xmax is not None else max(numeric_vals)
    except ValueError:
        return 'Unable to determine data range.'
    
    # Handle edge case where min == max
    if v_min == v_max:
        return f'All values are identical: {v_min}'
    
    # Generate bin edges
    if bins is None:
        try:
            left_edges = linspace(v_min, v_max, N + 1)
        except ValueError as e:
            raise ValueError(f"Failed to create bins: {e}")
    else:
        left_edges = list(bins)  # Make a copy
        if len(left_edges) < 2:
            raise ValueError("Custom bins must have at least 2 edges")
    
    # Separate data into bins and tails
    left_tail = [v for v in numeric_vals if v < left_edges[0]]
    right_tail = [v for v in numeric_vals if v > left_edges[-1]]
    binned_vals = [v for v in numeric_vals if left_edges[0] <= v <= left_edges[-1]]
    
    # Calculate histogram
    bin_vals = hist(binned_vals, left_edges)
    bin_counts = list(bin_vals)
    
    # Generate labels
    labels = get_bin_labels(left_edges, left_tail=len(left_tail) > 0, right_tail=len(right_tail) > 0)
    
    # Apply log scaling if requested
    if yscale == 'log':
        bin_vals = _apply_log_scaling(bin_vals)
    
    # Format output
    result = _format_histogram_output(bin_vals, bin_counts, labels, len(numeric_vals), height, char, with_counts)
    
    # Add tail information
    if left_tail:
        tail_label = labels.pop(0) if labels else f'<{left_edges[0]}'
        tail_info = f'{tail_label} : {format_percentage(len(left_tail)/len(numeric_vals), fmt="{:d}")} ({len(left_tail)})\n'
        result = tail_info + result
    
    if right_tail:
        tail_label = labels[-1] if labels else f'>{left_edges[-1]}'
        tail_info = f'{tail_label} : {format_percentage(len(right_tail)/len(numeric_vals), fmt="{:d}")} ({len(right_tail)})\n'
        result = result + tail_info
    
    return result


def _apply_log_scaling(bin_vals: List[int]) -> List[float]:
    """Apply log scaling to bin values."""
    scaled_vals = []
    for val in bin_vals:
        if val > 0:
            try:
                scaled_vals.append(log(val, 2) + 1)
            except (ValueError, ZeroDivisionError):
                scaled_vals.append(0)
        else:
            scaled_vals.append(0)
    return scaled_vals


def _format_histogram_output(bin_vals: List[Union[int, float]], 
                           bin_counts: List[int], 
                           labels: List[str], 
                           total_count: int, 
                           height: int, 
                           char: str, 
                           with_counts: bool) -> str:
    """Format the histogram output string."""
    if not bin_vals:
        return 'No bins to display.'
    
    # Avoid division by zero
    max_bin_val = max(bin_vals) if bin_vals else 1
    if max_bin_val == 0:
        max_bin_val = 1
    
    lines = []
    for i in range(len(bin_counts)):
        if i >= len(labels):
            label = f'Bin {i}'
        else:
            label = labels[i]
            
        prefix = f'{label} : '
        
        # Calculate bar height safely
        if max_bin_val > 0:
            bar_height = int(height * bin_vals[i] / max_bin_val)
        else:
            bar_height = 0
            
        # Ensure bar_height is within bounds
        bar_height = max(0, min(bar_height, height))
        
        # Build the bar
        bar = char * bar_height + ' ' * (height - bar_height)
        
        line = prefix + bar
        
        # Add counts if requested
        if with_counts and i < len(bin_counts) and bin_counts[i] > 0 and total_count > 0:
            percentage = format_percentage(bin_counts[i] / total_count)
            line += f'  {percentage} ({bin_counts[i]})'
        
        lines.append(line)
    
    return '\n' + '\n'.join(lines) + '\n'


def hist_categorical(vals: List[str], categories: List[str]) -> List[int]:
    """Calculate histogram for categorical data."""
    return [sum(1 for v in vals if v == category) for category in categories]


def hist(vals: List[float], left_edges: List[float]) -> List[int]:
    """Calculate histogram for numerical data."""
    if len(left_edges) < 2:
        return []
    
    result = []
    for i in range(len(left_edges) - 1):
        count = sum(1 for v in vals if left_edges[i] <= v < left_edges[i + 1])
        result.append(count)
    
    return result


def format_percentage(ratio: float, fmt: str = '{: >3d}') -> str:
    """Format a ratio as a percentage string."""
    try:
        percentage = int(100 * ratio)
        return fmt.format(percentage) + '%'
    except (ValueError, TypeError):
        return '  0%'


def get_bin_labels(edges: List[float], left_tail: bool = False, right_tail: bool = False) -> List[str]:
    """Generate labels for histogram bins."""
    if len(edges) < 2:
        return []
    
    try:
        # Determine width for formatting
        w = max(len(str(int(edge))) for edge in edges)
        
        # Calculate minimum step size for precision
        min_step = edges[-1] - edges[0]
        for i in range(len(edges) - 1):
            step = edges[i + 1] - edges[i]
            if step > 0:
                min_step = min(min_step, step)
        
        # Determine decimal places needed
        if min_step < 1 and min_step > 0:
            # Extract exponent from scientific notation
            sci_notation = f'{min_step:e}'
            if 'e' in sci_notation:
                exp_part = sci_notation.split('e')[1]
                p = -int(exp_part)
            else:
                p = 0
        else:
            p = 0
        
        # Generate format string
        fmt_suffix = f'{w + p}.{p}f}}'
        
        # Generate bin labels
        labels = []
        for i in range(len(edges) - 1):
            left_val = f'{edges[i]:>{w + p}.{p}f}'
            right_val = f'{edges[i + 1]:<{w + p}.{p}f}'
            label = f'{left_val} - {right_val}'
            labels.append(label)
        
        # Add tail labels if needed
        if labels:  # Only add tails if we have main labels
            label_width = len(labels[0]) if labels else 20
            
            if left_tail:
                tail_label = f'<{edges[0]:{w + p}.{p}f}'
                tail_label = tail_label.rjust(label_width)
                labels.insert(0, tail_label)
            
            if right_tail:
                tail_label = f'>{edges[-1]:{w + p}.{p}f}'
                tail_label = tail_label.rjust(label_width)
                labels.append(tail_label)
        
        return labels
        
    except (ValueError, TypeError, ZeroDivisionError) as e:
        # Fallback to simple labels
        return [f'Bin {i}' for i in range(len(edges) - 1)]


def safe_file_read(filepath: str, everything: bool = False) -> List[Union[str, float]]:
    """Safely read and parse data from a file."""
    vals = []
    
    # Handle stdin
    if filepath == '/dev/stdin' or filepath == '-':
        fp = sys.stdin
        mode = 'r'
    else:
        # Check if file exists
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Try to determine if file is binary
        try:
            with open(filepath, 'r', encoding='utf-8') as test_fp:
                test_fp.read(1024)  # Try to read first 1KB as text
            mode = 'r'
            encoding = 'utf-8'
        except UnicodeDecodeError:
            mode = 'rb'
            encoding = None
    
    try:
        if filepath == '/dev/stdin' or filepath == '-':
            lines = fp.readlines()
        else:
            with open(filepath, mode, encoding=encoding) as fp:
                if mode == 'rb':
                    lines = [line.decode('utf-8', errors='ignore') for line in fp.readlines()]
                else:
                    lines = fp.readlines()
        
        for line in lines:
            line = line.strip()
            if not line:  # Skip empty lines
                continue
                
            if everything:
                vals.append(line)
            else:
                try:
                    vals.append(float(line))
                except ValueError:
                    # Skip non-numeric lines when not using --everything
                    continue
                    
    except IOError as e:
        raise IOError(f"Error reading file {filepath}: {e}")
    
    return vals


def main(args) -> None:
    """Main function to handle command line execution."""
    try:
        # Read data from file
        vals = safe_file_read(args.infile, args.everything)
        
        if not vals:
            print("No data found in input file.", file=sys.stderr)
            sys.exit(1)
        
        # Prepare arguments for textogram
        kwargs = {
            'bins': args.bins,
            'categorical': args.categorical,
            'char': args.char,
            'height': args.height,
            'N': args.num_bins,
            'with_counts': args.with_counts,
            'xmin': args.min,
            'xmax': args.max,
            'yscale': args.yscale,
        }
        
        # Generate and print textogram
        textogram(vals, **kwargs)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate ASCII histograms from numeric data')
    parser.add_argument('infile', nargs='?', default='/dev/stdin',
                       help='Input file (default: stdin)')
    parser.add_argument('--bins', '-b', nargs='+', type=float,
                       help='Custom bin edges')
    parser.add_argument('--categorical', '-c', action='store_true',
                       help='Treat data as categorical')
    parser.add_argument('--char', '-C', default='#',
                       help='Character to use for bars (default: #)')
    parser.add_argument('--everything', '-e', action='store_true',
                       help='Include all lines, not just numeric ones')
    parser.add_argument('--height', type=int, default=50,
                       help='Maximum bar height (default: 50)')
    parser.add_argument('--min', '-a', type=float,
                       help='Minimum value for binning')
    parser.add_argument('--max', '-z', type=float,
                       help='Maximum value for binning')
    parser.add_argument('--num-bins', '-n', type=int, default=10,
                       help='Number of bins (default: 10)')
    parser.add_argument('--yscale', '-y', choices=['lin', 'log'], default='lin',
                       help='Y-axis scaling (default: lin)')
    parser.add_argument('--with-counts', action='store_false',
                       help='Hide counts and percentages')
    
    try:
        args = parser.parse_args()
        main(args)
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)
