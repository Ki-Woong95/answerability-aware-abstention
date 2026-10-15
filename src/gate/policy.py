import numpy as np


def cumulative_mean(probs):
    probs = np.asarray(probs, dtype=float)

    if len(probs) == 0:
        return np.array([], dtype=float)

    return np.cumsum(probs) / np.arange(1, len(probs) + 1)


def trajectory_score(probs):
    scores = cumulative_mean(probs)

    if len(scores) == 0:
        return 0.0

    return float(scores.max())


def first_brake(probs, checkpoints, threshold):
    probs = np.asarray(probs, dtype=float)
    checkpoints = list(checkpoints)

    if len(probs) != len(checkpoints):
        raise ValueError(
            f"{len(probs)} probabilities but " f"{len(checkpoints)} checkpoints"
        )

    scores = cumulative_mean(probs)

    for score, checkpoint in zip(scores, checkpoints):
        if score >= threshold:
            return int(checkpoint)

    return None
