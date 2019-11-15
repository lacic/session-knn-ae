import os
import sys


def generate_folder_structure(dataset):
    print("Generating Folders for: " + dataset)
    root = "data"
    os.makedirs(root + "/" + dataset + "/raw")
    os.makedirs(root + "/" + dataset + "/interim/infer")
    os.makedirs(root + "/" + dataset + "/interim/models")
    os.makedirs(root + "/" + dataset + "/interim/predict/base/warm")
    os.makedirs(root + "/" + dataset + "/interim/predict/knn_candidate_sessions/warm")
    os.makedirs(root + "/" + dataset + "/interim/predict/content_candidate_sessions/warm")

    algo_types = ["base", "content_knn_candidates", "interaction_knn_candidates", "content_recent_sessions", "interaction_recent_sessions"]
    eval_types = ["all", "next"]

    for algo_type in algo_types:
        for eval_type in eval_types:
                os.makedirs(root + "/" + dataset + "/processed/eval/" + algo_type + "/" + eval_type)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage: python generate_folders.py dataset1 [dataset2 ...]")
    else:
        for dataset in sys.argv[1:]:
            generate_folder_structure(dataset)
