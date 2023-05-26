import os
import json
import numpy as np


def read_json_file(json_file):
    with open(json_file) as f:
        data = json.load(f)
    return data


def get_scores(data):
    scores = []
    for i in range(len(data["scores"])):
        scores.append(data["scores"][i]["score"])
    return scores


def get_labels(data):
    labels = []
    for i in range(len(data["scores"])):
        labels.append(data["scores"][i]["label"])
    return labels


# plot scores vs num to determine eer threshold
def plot_scores(scores, labels, title):
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(scores, label="scores")
    plt.plot(labels, label="labels")
    plt.title(title)
    plt.legend()
    plt.show()


def main():
    dirname = "logs/stats/val_base"

    for filename in os.listdir(dirname):
        if filename.endswith(".json"):
            json_file = os.path.join(dirname, filename)
            data = read_json_file(json_file)
            scores = get_scores(data)
            labels = get_labels(data)
            title = filename.split(".")[0]
            plot_scores(scores, labels, title)


if __name__ == "__main__":
    main()
