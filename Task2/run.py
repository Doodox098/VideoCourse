#!/usr/bin/env python3

from json import dumps, load, loads
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
        f'python3 deint.py --data_dir={data_dir} --output_file={join(output_dir, "results.csv")}'
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

    metrics = results[1]  # Skip header, each video stored in separate test
    verdict = dumps({"PSNR": float(metrics[0]), "SSIM": float(metrics[1])})

    if environ.get('CHECKER'):
        print(verdict)
    return verdict


def grade(data_path):
    # {data_dir}---results.json

    results = load(open(join(data_path, 'results.json')))
    psnrs = []
    ssims = []
    for result in results:
        if "Time limit" in result['status'] or "Runtime error" in result['status']:
            continue
        metrics = loads(result['status'])
        psnrs.append(metrics["PSNR"])
        ssims.append(metrics["SSIM"])
    
    mean_psnr = sum(psnrs) / len(results)
    mean_ssim = sum(ssims) / len(results)
    res = {'description': f'PSNR: {round(mean_psnr, 6)}, SSIM: {round(mean_ssim, 6)}', 'mark': len(psnrs)}

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
