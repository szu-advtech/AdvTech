#!/usr/bin/env python

'''
A simple script that reads a file containing the output of one or more
consecutive runs of schedbench, and aggregates the result.
For each schedule, it prints the average and median overhead.
It also ignores results where the number of outliers are higher
than outer-repetitions / 5.
'''

from __future__ import print_function
import argparse
from collections import OrderedDict

class AggregatedTaskTestResult:

    def __init__(self, overhead):
        self.overheads = [overhead]

    def add_result(self, overhead):
        self.overheads.append(overhead)

    def average(self):
        return sum(self.overheads) / len(self.overheads)

    def median(self):
        # Adjusted from http://codereview.stackexchange.com/questions/126890/find-the-median-value-of-a-list
        numbers = sorted(self.overheads)
        center = len(numbers) / 2
        if len(numbers) % 2 == 0:
            return sum(numbers[center - 1:center + 1]) / 2.0
        else:
            return numbers[center]

    def __str__(self):
        return "avg = {:8.4f} , median = {:8.4f} ({} runs)".format(self.average(), self.median(), len(self.overheads))

def extract_test_type(line):
    words = line.split()
    schedule = ''
    # Iterate second, third, and fourth word
    for word in words[1:5]:
        if word == 'time':
            break
        schedule += word + ' '
    return schedule.rstrip()

def extract_outliers(line):
    words = line.split()
    outlier_string = words[5]
    return int(outlier_string)

def extract_overhead(line):
    words = line.split()
    equal_index = words.index("=")
    overhead_string = words[equal_index + 1]
    return float(overhead_string)

def find_outer_repetitions(fname):
    outer_repetitions = 0
    # We assume it stays unchanged
    with open(fname) as f:
        for line in f:
            if "outer repetitions" in line:
                outer_repetitions = int(line.strip().split()[0])
                break
    return outer_repetitions

# From http://stackoverflow.com/questions/354038/how-do-i-check-if-a-string-is-a-number-float-in-python
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def process(fname, print_csv):
    outer_repetitions = find_outer_repetitions(fname)
    if not print_csv:
        print("Using outer_repetitions = {}".format(outer_repetitions))
    max_outliers = outer_repetitions / 5
    overhead_results = OrderedDict()
    with open(fname) as f:
        for line in f:
            if line.strip() == "":
                continue
            if "Computing" in line:
                test = extract_test_type(line)
                continue
            if all(map(lambda x: is_number(x), line.strip().split())):
                # Matched a test result line
                outliers = extract_outliers(line)
                continue
            if "overhead" in line:
                overhead = extract_overhead(line)
                if outliers > max_outliers:
                    print("FOUND MANY OUTLIERS: {}".format(outliers))
                else:
                    if test in overhead_results:
                        overhead_results[test].add_result(overhead)
                    else:
                        overhead_results[test] = AggregatedTaskTestResult(overhead)
                # Reset variables to something easily recognizable
                sched = "NULL"
                outliers = outer_repetitions
                overhead = -1337
    return overhead_results

def print_results(results, as_csv):
    for key, value in results.iteritems():
        if as_csv:
            print("{},{}".format(key, value.median()))
        else:
            key_adjusted = str(key).ljust(25)
            print("{} : {}".format(key_adjusted, str(value)))

parser = argparse.ArgumentParser()
parser.add_argument("--csv", help="print results as csv",
    action="store_true", default=False)
parser.add_argument("filename")
args = parser.parse_args()
fname = args.filename
print_csv = args.csv

results = process(fname, print_csv)
print_results(results, print_csv)
