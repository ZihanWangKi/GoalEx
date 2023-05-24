import sklearn
import sklearn.metrics

import numpy as np
import scipy
import scipy.optimize
from collections import Counter
from typing import List, Tuple, Dict


def get_description_validation_quality(
    text_descriptions_matching,
) -> Tuple[float, float]:
    """
    Returns a list of tuples (num_predicted_as_description, num_only_predicted_as_description) for each description.
    num_predicted_as_description: number of text that is predicted as that description.
    num_only_predicted_as_description: number of text that is predicted as that description and only that description.

    Parameters
    ----------
    text_descriptions_matching: np.ndarray
        A binary matrix of shape (n_instances, n_descriptions) where each row is an instance and each column is a description.

    Returns
    -------
    List[Tuple]
        A list of tuples (num_predicted_as_description, num_only_predicted_as_description) for each description.
    """
    n_instances, n_descriptions = text_descriptions_matching.shape
    predictions_counts = []
    for j in range(n_descriptions):
        predicted_as_description = text_descriptions_matching[
            text_descriptions_matching[:, j] == 1
        ]
        num_predicted_as_description = len(predicted_as_description)
        only_predicted_as_description = predicted_as_description[
            predicted_as_description.sum(axis=1) == 1
        ]
        num_only_predicted_as_description = len(only_predicted_as_description)
        predictions_counts.append(
            (num_predicted_as_description, num_only_predicted_as_description)
        )
    return predictions_counts


def get_descriptions_performance(
    text_descriptions_matching,
) -> Tuple[float, float, List[float], List[float]]:
    """
    Gets the w/o label validation performance of a set of descriptions over a set of text.
    We use the following metrics:
    1. all_recall: percentage of text that is covered by at least one description
    2. all_precision: percentage of text that is by only one description, over all covered precisions
    3. single_recall (list): for each description, the percentage of text that is covered by that description,
                             over the total number of text.
    4. single_precision (list): for each description, the percentage of text that is covered by only that description,
                                over the total number of text that is covered by that description.

    Parameters
    ----------
    text_descriptions_matching: np.ndarray
        A binary matrix of shape (n_instances, n_descriptions) where each row is an instance and each column is a description.

    Returns
    -------
    Tuple[float, float, List[float], List[float]]
        A tuple (all_recall, all_precision, single_recall, single_precision).
    """
    n_instances, n_descriptions = text_descriptions_matching.shape
    prediction_counts = get_description_validation_quality(text_descriptions_matching)
    all_recall = (text_descriptions_matching.sum(axis=1) > 0).sum() / n_instances
    all_precision = (text_descriptions_matching.sum(axis=1) == 1).sum() / max(
        1, (text_descriptions_matching.sum(axis=1) > 0).sum()
    )
    single_recalls = [
        prediction_counts[j][0] / n_instances for j in range(n_descriptions)
    ]
    single_precisions = [
        prediction_counts[j][1] / max(1, prediction_counts[j][0])
        for j in range(n_descriptions)
    ]
    return all_recall, all_precision, single_recalls, single_precisions


def assign_labels(
    ground_truth_labels, predicted_labels
) -> Tuple[List[int], Dict[int, int]]:
    """
    Assigns predicted_labels to ground_truth_labels using the Hungarian algorithm.

    Parameters
    ----------
    ground_truth_labels: List[int]
        A list of ground truth labels.
    predicted_labels: List[int]
        A list of predicted labels.

    Returns
    -------
    Tuple[List[int], Dict[int, int]]
        A tuple (assigned_predicted_labels, mapping) where assigned_predicted_labels is the list of predicted labels and
        mapping is a dictionary mapping each predicted label to a ground truth label.
    """
    n = len(ground_truth_labels)
    assert n == len(predicted_labels)
    m = max(ground_truth_labels) + 1
    mp = max(predicted_labels) + 1
    cost_matrix = np.zeros((m, mp))
    for gt_label, pred_label in zip(ground_truth_labels, predicted_labels):
        cost_matrix[gt_label, pred_label] -= 1
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)
    row_ind, col_ind = row_ind.tolist(), col_ind.tolist()
    mapping = {col_ind[i]: row_ind[i] for i in range(len(col_ind))}
    for i in range(mp):
        if i not in mapping:
            mapping[i] = np.argmax(cost_matrix[:, i]).item()
    return [mapping[p] for p in predicted_labels], mapping


def get_cluster_performance(
    ground_truth_labels, predicted_labels
) -> Tuple[float, float, float]:
    """
    Gets the performance of a clustering algorithm given the ground truth labels and the predicted labels.
    We use the following metrics:
    1. normalized_mutial_info: normalized mutual information between the ground truth labels and the predicted labels.
    2. adjusted_rand_index: adjusted rand index between the ground truth labels and the predicted labels.
    3. macro_f1: macro f1 score between the ground truth labels and the predicted labels, after assigning the predicted
                 labels to the ground truth labels using the Hungarian algorithm.

    Parameters
    ----------
    ground_truth_labels: List[int]
        A list of ground truth labels.
    predicted_labels: List[int]
        A list of predicted labels.

    Returns
    -------
    Tuple[float, float, float]
        A tuple (normalized_mutial_info, adjusted_rand_index, macro_f1).
    """
    normalized_mutial_info = sklearn.metrics.cluster.normalized_mutual_info_score(
        ground_truth_labels, predicted_labels
    )
    adjusted_rand_index = sklearn.metrics.cluster.adjusted_rand_score(
        ground_truth_labels, predicted_labels
    )

    max_assignment_based_labels = assign_labels(ground_truth_labels, predicted_labels)[
        0
    ]
    macro_f1 = sklearn.metrics.f1_score(
        ground_truth_labels, max_assignment_based_labels, average="macro"
    )
    return normalized_mutial_info, adjusted_rand_index, macro_f1


def get_cluster_performance_generalized(ground_truth_labels, predicted_labels):
    """
    Gets the performance of a clustering algorithm given the ground truth labels and the predicted labels.
    We use the following metrics:
    1. normalized_mutual_info: normalized mutual information between the ground truth labels and the predicted labels.
    2. adjusted_rand_index: adjusted rand index between the ground truth labels and the predicted labels.
    3. macro_f1: macro f1 score between the ground truth labels and the predicted labels, after assigning the predicted
                 labels to the ground truth labels using the Hungarian algorithm.

    Parameters
    ----------
    ground_truth_labels: List[int]
        A list of ground truth labels.
    predicted_labels: List[int]
        A list of predicted labels.

    Returns
    -------
    Tuple[float, float, float]
        A tuple (normalized_mutial_info, adjusted_rand_index, macro_f1).
    """
    normalized_mutual_info = sklearn.metrics.cluster.normalized_mutual_info_score(
        ground_truth_labels, predicted_labels
    )

    max_assignment_based_labels = assign_labels(ground_truth_labels, predicted_labels)[
        0
    ]
    macro_f1 = sklearn.metrics.f1_score(
        ground_truth_labels, max_assignment_based_labels, average="macro"
    )
    acc = sklearn.metrics.accuracy_score(
        ground_truth_labels, max_assignment_based_labels
    )
    n = max(ground_truth_labels) + 1
    ground_truth_counts = Counter(ground_truth_labels)
    ground_truth_counts = [ground_truth_counts[i] for i in range(n)]
    predicted_counts = Counter(max_assignment_based_labels)
    predicted_counts = [predicted_counts[i] for i in range(n)]
    matched_counts = sklearn.metrics.confusion_matrix(
        ground_truth_labels, max_assignment_based_labels
    )
    matched_counts = [matched_counts[i][i] for i in range(len(matched_counts))]
    return (
        normalized_mutual_info,
        acc,
        macro_f1,
        (ground_truth_counts, predicted_counts, matched_counts),
    )
