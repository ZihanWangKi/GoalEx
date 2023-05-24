import numpy as np
from typing import List


def greedy_cover(can_cover: np.ndarray, num_clusters: int) -> List[List[int]]:
    """
    Given a matrix of scores, which the i-th row and j-th column is 1/0 whether the i-th text satisfies the j-th description, greedily select column-indexes (descriptions) to cover the most row-indexes (texts).
    Parameters
    ----------
    scores : np.ndarray
        A matrix of scores, which the i-th row and j-th column is how well the i-th text satisfies the j-th description.
    num_clusters : int
        The number of clusters to be selected.
    Returns
    -------
    List[List[int]]
        A list of column-indexes (descriptions) that can cover the most row-indexes (texts).
    """
    selected_idxes = []
    while np.sum(can_cover) > 0 and len(selected_idxes) < num_clusters:
        # find the description that can cover the most texts
        num_texts_covered_by_each_description = np.sum(can_cover, axis=0)
        best_description_index = np.argmax(num_texts_covered_by_each_description)
        selected_idxes.append(best_description_index)

        # remove the texts that are covered by the best description
        can_cover = can_cover * (
            1 - can_cover[:, best_description_index][:, np.newaxis]
        )

    return selected_idxes