import argparse
import csv
import os


def run_test(data_dir):
    input_frames_dir = os.path.join(data_dir, "interlace")
    gt_path = os.path.join(data_dir, "gt")

    model_outputs = test_model(input_frames_dir)
    ssim, psnr = calculate_metrics(model_outputs, gt_path)

    return [psnr, ssim]


def save_results(results, filename):
    header = ["psnr", "ssim"]
    with open(filename, "w", newline="") as resfile:
        writer = csv.writer(resfile)
        writer.writerow(header)
        writer.writerow(results)
        resfile.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testing script")
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--output_file", type=str)

    args = parser.parse_args()
    results = run_test(args.data_dir)

    save_results(results, args.output_file)
