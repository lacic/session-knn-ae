import os
import sys


def generate_folder_structure(dataset):
    print("Generating Folders for: " + dataset)
    root = "data"
    os.makedirs(root + "/" + dataset + "/raw")
    os.makedirs(root + "/" + dataset + "/interim/infer")
    os.makedirs(root + "/" + dataset + "/interim/models")
    os.makedirs(root + "/" + dataset + "/interim/predict/base")
    os.makedirs(root + "/" + dataset + "/interim/predict/hyperparam")
    os.makedirs(root + "/" + dataset + "/interim/processed/eval/all")
    os.makedirs(root + "/" + dataset + "/interim/processed/eval/next")
    os.makedirs(root + "/" + dataset + "/interim/processed/eval/base/all")
    os.makedirs(root + "/" + dataset + "/interim/processed/eval/base/next")
    os.makedirs(root + "/" + dataset + "/interim/processed/eval/hyperparam/all")
    os.makedirs(root + "/" + dataset + "/interim/processed/eval/hyperparam/next")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage: python generate_folders.py dataset1 [dataset2 ...]")
    else:
        for dataset in sys.argv[1:]:
            generate_folder_structure(dataset)
