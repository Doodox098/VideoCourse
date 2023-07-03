#!/usr/bin/env python3

from json import dumps, load
from os import environ, makedirs
from os.path import join
from sys import argv, exit
from subprocess import run
import csv


def run_single_test(data_dir, output_dir):
    # '.'---user code
    #   |---additional code
    #   └---run.py
    # data_dir == dir with input data in tests

    cmds = [
        f'python3 scd.py --test_dir={data_dir} --output_file={join(output_dir, "results.csv")}'
    ]

    for cmd in cmds:
        ret_code = run(cmd, shell=True).returncode
        if ret_code != 0:
            exit(ret_code)


def check_test(data_dir):
    # {data_dir}---output/results.csv
    #          \---gt/gt.csv

    with open(join(data_dir, 'output/results.csv')) as f:
        csvreader = csv.reader(f)
        results = list(csvreader)

    result = results[1]  # Skip header, each video stored in separate test
    f1 = result[0]
    status = result[1]

    verdict = '\n'.join([status, f1])  # OK hist(ht = 6.met+hist(mt = 3.2, ht = 5.0, fs = 7, hs = 16), fs = 7, hs = 16).9 or RE hist(ht = 6.met+hist(mt = 3.2, ht = 5.0, fs = 7, hs = 16), fs = 7, hs = 16).hist(ht = 6.met+hist(mt = 3.2, ht = 5.0, fs = 7, hs = 16), fs = 7, hs = 16)

    if environ.get('CHECKER'):
        print(verdict)
    return verdict


def grade(data_path):
    # {data_dir}---results.json

    results = load(open(join(data_path, 'results.json')))
    ok_count = 0
    metric_sum = 0
    for result in results:
        if "Time limit" in result['status']:
            continue
        spl = result['status'].split()
        ok_count += int(spl[0] == "OK")
        metric_sum += float(spl[-1])

    res = {'description': f'{metric_sum / len(results)}', 'mark': ok_count}
    if environ.get('CHECKER'):
        print(dumps(res))
    return res


if __name__ == '__main__':
    if environ.get('CHECKER'):
        # Script is running in testing system
        if len(argv) != 4:
            print(f'Usage: {argv[0]} mode data_dir output_dir')
            exit(0)

        mode = argv[1]
        data_dir = argv[2]
        output_dir = argv[3]

        if mode == 'run_single_test':
            # Run each test
            run_single_test(data_dir, output_dir)
        elif mode == 'check_test':
            # Put a mark for each test result
            check_test(data_dir)
        elif mode == 'grade':
            # Put overall mark
            grade(data_dir)
