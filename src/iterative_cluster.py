import os
import json
import time
import datetime
import os
import copy
import random
from collections import Counter
from typing import List, Tuple

import numpy as np
from propose_cluster_descriptions import (
    propose_descriptions_multi_round,
)
from assign_descriptions import (
    assign_descriptions,
    get_assigner,
    Assigner,
)
from cover_algo import greedy_cover, maximum_set_coverage

from cluster_problem import ClusterProblem as Problem, ClusterProblemLabel as Label
from utils import (
    get_max_num_samples_in_proposer,
    get_avg_length,
    get_length_in_gpt2_tokens,
    estimate_querying_cost,
)

from experiment_recorder import ExperimentRecorder


def estimate_cost_for_clustering(
    problem: Problem,
    # Proposer arguments
    proposer_model: str,
    proposer_num_descriptions_to_propose: int,
    proposer_num_rounds_to_propose: int,
    proposer_num_descriptions_per_round: int,
    proposer_template: str,
    proposer_num_samples: int,
    # assigner arguments
    assigner_name: str,
    assigner_for_proposed_descriptions_template: str,
    assigner_for_final_assignment_template: str,
    # clusterer arguments
    cluster_num_clusters: int,
    # iterative PAS arguments
    iterative_max_rounds: int,
) -> float:
    """
    Estimate the cost for clustering the given problem. Return the estimated total cost.
    """

    # length of the average sample and the expected length of the description
    averaged_sample_length = get_avg_length(problem.texts)
    description_expected_length = 10 if "short" in proposer_template else 40

    # figure out how many rounds to run the proposer for each iteration
    proposer_num_rounds_to_propose = (
        proposer_num_rounds_to_propose
        if proposer_num_rounds_to_propose is not None
        else (
            proposer_num_descriptions_to_propose // proposer_num_descriptions_per_round
            + 1
        )
    )

    # figure out how many tokens are there in the proposer prompt
    with open(proposer_template, "r") as f:
        proposer_template = f.read()
        proposer_template_length = get_length_in_gpt2_tokens(proposer_template)
    goal_length = get_length_in_gpt2_tokens(problem.goal)
    proposer_one_prompt_tokens = (
        goal_length
        + proposer_template_length
        + averaged_sample_length * proposer_num_samples
    )
    # one proposer completion token is the length of the description * the number of descriptions per round
    proposer_one_completion_tokens = (
        proposer_num_descriptions_per_round * description_expected_length
    )
    # estimate the cost of querying the proposer
    proposer_cost_per_query = estimate_querying_cost(
        proposer_one_prompt_tokens, proposer_one_completion_tokens, proposer_model
    )
    # estimate the cost of running the proposer for one iteration
    proposer_cost_per_iteration = (
        proposer_cost_per_query * proposer_num_rounds_to_propose
    )

    if "t5" in assigner_name:
        assigner_cost_per_iteration = 0
    else:
        with open(assigner_for_proposed_descriptions_template, "r") as f:
            assigner_for_proposed_descriptions_template = f.read()
            assigner_for_proposed_descriptions_template_length = (
                get_length_in_gpt2_tokens(assigner_for_proposed_descriptions_template)
            )

        # figure out how many tokens are there in an assigner's prompt
        assigner_one_prompt_tokens = (
            goal_length
            + assigner_for_proposed_descriptions_template_length
            + averaged_sample_length
            + description_expected_length
        )
        assigner_one_completion_tokens = 3
        assigner_cost_per_query = estimate_querying_cost(
            assigner_one_prompt_tokens, assigner_one_completion_tokens, assigner_name
        )

        # estimate the cost of running the assigner for one iteration
        assigner_cost_per_iteration = (
            assigner_cost_per_query
            * proposer_num_descriptions_to_propose
            * len(problem.texts)
        )

    each_iteration_cost = proposer_cost_per_iteration + assigner_cost_per_iteration

    # estimate the cost of the final assignment, similar to the assigner
    final_assignment_cost = 0
    if assigner_for_final_assignment_template is not None and "t5" not in assigner_name:
        with open(assigner_for_final_assignment_template, "r") as f:
            assigner_for_final_assignment_template = f.read()
            assigner_for_final_assignment_template_length = get_length_in_gpt2_tokens(
                assigner_for_final_assignment_template
            )
        assigner_one_prompt_tokens = (
            goal_length
            + assigner_for_final_assignment_template_length
            + averaged_sample_length
            + description_expected_length
        )
        assigner_one_completion_tokens = 3
        final_assignment_cost = estimate_querying_cost(
            assigner_one_prompt_tokens, assigner_one_completion_tokens, assigner_name
        )
        final_assignment_cost *= cluster_num_clusters * len(problem.texts)

    total = (
        proposer_cost_per_iteration + assigner_cost_per_iteration
    ) * iterative_max_rounds + final_assignment_cost
    print(
        "Here's a rough estimate of the cost; it might not be accurate and we are not responsible for any cost incurred."
    )
    print(
        f"Each round costs {each_iteration_cost:.2f}, proposer spending {proposer_cost_per_iteration:.2f}, assigner spending {assigner_cost_per_iteration:.2f}. The final assignment costs {final_assignment_cost:.2f}."
    )
    print(f"Total cost is {total + final_assignment_cost:.2f}")
    return total


def prune_descriptions(
    descriptions: List[str],
    text_descriptions_matching: np.ndarray,
    min_cluster_fraction: float,
    max_cluster_fraction: float,
) -> Tuple[List[str], np.ndarray]:
    """
    Prune descriptions based on their popularity.

    Parameters
    ----------
    descriptions : List[str]
        all the original descriptions
    text_descriptions_matching : np.ndarray
        The matrix of text-description matching.
    min_cluster_fraction : float
        The minimum fraction of clusters that a description must cover.
    max_cluster_fraction : float
        The maximum fraction of clusters that a description must cover.

    Returns
    -------
    Tuple[List[str], np.ndarray]
        The pruned descriptions and the pruned text-description matching matrix.
    """
    descriptions_index_to_remove = []
    for i in range(len(descriptions)):
        num_clusters_covered = np.sum(text_descriptions_matching[:, i])
        if num_clusters_covered < min_cluster_fraction * len(
            text_descriptions_matching
        ) or num_clusters_covered > max_cluster_fraction * len(
            text_descriptions_matching
        ):
            descriptions_index_to_remove.append(i)
    print(
        f"Dropping {len(descriptions_index_to_remove)} descriptions because they are too popular or too unpopular:"
    )
    descriptions = [
        descriptions[i]
        for i in range(len(descriptions))
        if i not in descriptions_index_to_remove
    ]
    text_descriptions_matching = np.delete(
        text_descriptions_matching, descriptions_index_to_remove, axis=1
    )
    return descriptions, text_descriptions_matching


def propose(
    problem: Problem,
    num_samples: int,
    example_descriptions: List[str],
    proposer_model: str = "gpt-3.5-turbo",
    proposer_num_descriptions_to_propose: int = 30,
    proposer_num_rounds_to_propose: int = None,
    proposer_num_descriptions_per_round: int = 8,
    proposer_template: str = "templates/gpt_cluster_proposer_short.txt",
    random_seed: int = 0,
) -> List[str]:
    """
    Proposal stage in the paper, which result in a list of candidate explanations for clusters. mainly calls the propose_descriptions_multi_round function.

    Parameters
    ----------
    problem : Problem
        The clustering problem to solve.
    num_samples : int
        The number of in-context samples to use to construct the prompt.
    example_descriptions : List[str]
        The example descriptions to use in the prompt. used to clarify what the goal is using some example descriptions.
    proposer_model : str, optional
        The model used to propose descriptions, by default "gpt-3.5-turbo"
    proposer_num_descriptions_to_propose : int, optional
        The number of descriptions to propose in total, by default 30
    proposer_num_rounds_to_propose : int, optional
        The number of rounds to propose descriptions, by default None
    proposer_num_descriptions_per_round : int, optional
        The number of descriptions to propose per round, by default 8
    proposer_template : str, optional
        The template used to construct the prompt, by default "templates/gpt_cluster_proposer_short.txt"; can switch to proposing more detailed descriptions by using "templates/gpt_cluster_proposer_detailed.txt"
    random_seed : int, optional
        The random seed, by default 0

    Returns
    -------
    List[str]
        The proposed descriptions. (we use descriptions and explanations interchangeably)
    """

    # obtain the proposer results for multiple rounds
    proposer_results = propose_descriptions_multi_round(
        problem=problem,
        num_samples=num_samples,
        model=proposer_model,
        template=proposer_template,
        example_descriptions=example_descriptions,
        num_descriptions_to_propose=proposer_num_descriptions_to_propose,
        num_rounds_to_propose=proposer_num_rounds_to_propose,
        num_descriptions_per_round=proposer_num_descriptions_per_round,
        random_seed=random_seed,
    )

    # gather the descriptions across multiple rounds
    descriptions = []
    for proposer_result in proposer_results:
        descriptions.extend(proposer_result.descriptions)
    descriptions = list(set(descriptions))[:proposer_num_descriptions_to_propose]

    return descriptions


def assign(
    descriptions: List[str],
    texts: List[str],
    assigner: Assigner,
    template: str,
    use_multi_assigner: bool = False,
    add_null_description: bool = False,
    progress_bar: bool = False,
):
    """
    The assignment stage in the paper, which results in a matrix of indicator variable for each pair of description and text. mainly calls the assign_descriptions function.

    Parameters
    ----------
    descriptions : List[str]
        The descriptions to assign. (we use descriptions and explanations interchangeably)
    texts : List[str]
        The texts to assign against.
    assigner : Assigner
        The assigner to use. This might be Flan-T5 or GPT-3/4 or Claude.
    template : str
        The template used to construct the prompt for the assigner.
    use_multi_assigner : bool, optional
        Whether to use multiple assigner, by default False. This means, given multiple dscriptions and a text, the assigner will return a score for each description.
    add_null_description : bool, optional
        Whether to add a null description for the multi-assigner, by default False. This means "None of the above" will be added to the list of descriptions.
    progress_bar : bool, optional
        Whether to show a progress bar for the assignment process.

    Returns
    -------
    np.ndarray
        The assignment scores, an |X| by J boolean matrix, where the entry xj = 1 if the description j supports the text sample x.
    """
    scores = assign_descriptions(
        descriptions=descriptions,
        texts=texts,
        assigner=assigner,
        template=template,
        use_multi_assigner=use_multi_assigner,
        add_null_description=add_null_description,
        progress_bar=progress_bar,
    )
    text_descriptions_matching = (scores >= 0.5).astype(int)
    return text_descriptions_matching


def select(
    text_descriptions_matching: np.ndarray,
    cluster_algo: str = "maximum_set_coverage",
    num_clusters: int = None,
    overlap_penalty: float = None,
    not_cover_penalty: float = None,
) -> List[int]:
    """
    The selection stage of our algorithm. Given an assignment matrix and a few hyperparameters for the objective function, this function returns a list of indices, where each index corresponds to a cluster.

    Parameters
    ----------
    text_descriptions_matching : np.ndarray
        The assignment matrix, an |X| by J boolean matrix, where the entry xj = 1 if the description j supports the text sample x.
    cluster_algo : str, optional
        The clustering algorithm to use, by default "maximum_set_coverage". Can be "greedy", or "maximum_set_coverage".
    num_clusters : int, optional
        The number of clusters to return, by default None. If None, then the number of clusters is determined by the algorithm.
    overlap_penalty : float, optional
        The overlap penalty for the objective function, by default None.
    not_cover_penalty : float, optional
        The not cover penalty for the objective function, by default None.

    Returns
    -------
    List[int]
        The list of indices, where each index corresponds to a cluster.
    """

    if cluster_algo == "greedy":
        cluster_predictions = greedy_cover(
            can_cover=text_descriptions_matching,
            num_clusters=num_clusters,
        )
    elif cluster_algo == "maximum_set_coverage":
        cluster_predictions = maximum_set_coverage(
            num_clusters=num_clusters,
            can_cover=text_descriptions_matching,
            overlap_penalty=overlap_penalty,
            not_cover_penalty=not_cover_penalty,
        )
    else:
        raise ValueError(f"Unknown cluster_algo: {cluster_algo}")
    return cluster_predictions


def run(
    problem: Problem,
    exp_dir: str,
    label: Label = None,
    random_seed: int = 0,
    verbose: bool = False,
    min_cluster_fraction: float = 0.0,
    max_cluster_fraction: float = 0.4,
    # Proposer arguments
    proposer_model: str = "gpt-3.5-turbo",
    proposer_num_descriptions_to_propose: int = 30,
    proposer_num_rounds_to_propose: int = None,
    proposer_num_descriptions_per_round: int = 8,
    proposer_template: str = "templates/gpt_cluster_proposer_short.txt",
    # assigner arguments
    assigner_name: str = "google/flan-t5-xl",
    assigner_for_proposed_descriptions_template: str = "templates/t5_assigner.txt",
    assigner_for_final_assignment_template: str = "templates/t5_multi_assigner_one_output.txt",
    # clusterer arguments
    cluster_algo: str = "maximum_set_coverage",
    cluster_num_clusters: int = None,
    cluster_overlap_penalty: float = 0.5,
    cluster_not_cover_penalty: float = 1.0,
    # iterative PAS arguments
    iterative_max_rounds: int = 5,
    iterative_stop_criteria: int = None,
    # estimate cost arguments
    approve_cost_before_running: bool = True,
):
    """
    The main function for running the iterative PAS.

    Parameters
    ----------
    problem: Problem
        The problem to solve.
    exp_dir: str
        The directory to save the results.
    label: Label (optional)
        The ground truth label. If provided, the performance of the iterative reclusterer will be evaluated.
    random_seed: int
        The random seed.
    verbose: bool
        Whether to print out the intermediate results.
    min_cluster_fraction: float
        The minimum fraction of the number of samples in a cluster to the total number of samples.
    max_cluster_fraction: float
        The maximum fraction of the number of samples in a cluster to the total number of samples.
    proposer_model: str
        The model to use for the proposer.
    proposer_num_descriptions_to_propose: int
        The number of descriptions to propose.
    proposer_num_rounds_to_propose: int
        The number of rounds to propose. By default, the rounds will be determined by proposer_num_descriptions_to_propose.
    proposer_num_descriptions_per_round: int
        The number of descriptions to propose in each proposing round.
    proposer_template: str
        The template to use for the proposer.
    assigner_name: str
        The name of the assigner to use.
    assigner_for_proposed_descriptions_template: str
        The template to use for the assigner for proposed descriptions.
    assigner_for_final_assignment_template: str
        The template to use for the assigner for final assignment. If set, will use a multi assigner without null descriptions at the final assignment step
    cluster_algo: str
        The algorithm to use for clustering.
    cluster_num_clusters: int
        The number of clusters to use for clustering.
    cluster_overlap_penalty: float
        The penalty for overlapping between descriptions.
    cluster_not_cover_penalty: float
        The penalty for not covering a text instance.
    iterative_max_rounds: int
        The maximum number of rounds to run the iterative reclusterer.
    iterative_stop_criteria: int
        If fewer than iterative_stop_criteria text instances have no description matched, stop.
    approve_cost_before_running: bool
        Ask the user to approve the cost before running the iterative reclusterer.

    Returns
    -------
    Dict[str, List[str]]
        A mapping from an explanation to a list of text instances (cluster) that are covered by the explanation.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    recorder = ExperimentRecorder(problem=problem, label=label)
    os.makedirs(exp_dir, exist_ok=True)
    recorder.set_output_dir(exp_dir)

    n = len(problem.texts)

    all_descriptions = []
    all_text_descriptions_matching = np.zeros((n, 0), dtype=np.int64)
    unselected_text_indicators = np.ones(n, dtype=bool)
    num_iterations = 0

    # proposer max_samples
    proposer_num_samples = min(
        get_max_num_samples_in_proposer(problem.texts, proposer_model), 50
    )  # don't give too much samples

    estimate_cost_for_clustering(  # arguments above
        problem=problem,
        proposer_model=proposer_model,
        proposer_num_descriptions_to_propose=proposer_num_descriptions_to_propose,
        proposer_num_rounds_to_propose=proposer_num_rounds_to_propose,
        proposer_num_descriptions_per_round=proposer_num_descriptions_per_round,
        proposer_template=proposer_template,
        proposer_num_samples=proposer_num_samples,
        assigner_name=assigner_name,
        assigner_for_proposed_descriptions_template=assigner_for_proposed_descriptions_template,
        assigner_for_final_assignment_template=assigner_for_final_assignment_template,
        cluster_num_clusters=cluster_num_clusters,
        iterative_max_rounds=iterative_max_rounds,
    )
    if approve_cost_before_running:
        feedback = input("Do you want to continue? (y/n)")
        if feedback != "y":
            exit(0)

    # assigner
    assigner = get_assigner(assigner_name, verbose=verbose)

    while num_iterations < iterative_max_rounds:
        if (
            iterative_stop_criteria
            and np.sum(unselected_text_indicators) < iterative_stop_criteria
        ):
            break

        ################ proposal stage ################
        _problem = copy.deepcopy(problem)

        # select the text instances that are not covered by any of the explanations, but still make sure that the number of text instances is not too small
        _problem.texts = [
            problem.texts[i] for i in range(n) if unselected_text_indicators[i]
        ]
        _problem.texts = (_problem.texts + problem.texts)[:proposer_num_samples]

        new_descriptions = propose(
            problem=_problem,
            num_samples=proposer_num_samples,
            proposer_model=proposer_model,
            # in the first iteration, we use example descriptions, otherwise we use 2 descriptions from the previous iteration
            example_descriptions=_problem.example_descriptions
            if len(all_descriptions) == 0
            else random.sample(all_descriptions, 2),
            proposer_num_descriptions_to_propose=proposer_num_descriptions_to_propose,
            proposer_num_rounds_to_propose=proposer_num_rounds_to_propose,
            proposer_num_descriptions_per_round=proposer_num_descriptions_per_round,
            proposer_template=proposer_template,
            random_seed=random_seed,
        )
        recorder.record_propose(new_descriptions, "proposer")

        new_descriptions = [
            description
            for description in set(new_descriptions)
            if description not in all_descriptions
        ]

        all_descriptions = all_descriptions + new_descriptions
        print(
            f"{len(all_descriptions)} descriptions proposed in total. Here are the descriptions:"
        )
        for description in all_descriptions:
            print(description)

        ################ assignment stage ################
        new_text_descriptions_matching = assign(
            descriptions=new_descriptions,
            texts=problem.texts,
            assigner=assigner,
            template=assigner_for_proposed_descriptions_template,
            use_multi_assigner=False,
            add_null_description=False,
            progress_bar=True,
        )
        all_text_descriptions_matching = np.concatenate(
            [all_text_descriptions_matching, new_text_descriptions_matching], axis=1
        )
        recorder.record_assign(
            all_descriptions, all_text_descriptions_matching, "assign"
        )

        ################ selection stage ################
        # first remove descriptions that are too unpopular or too popular
        pruned_descriptions, pruned_text_descriptions_matching = prune_descriptions(
            descriptions=all_descriptions,
            text_descriptions_matching=all_text_descriptions_matching,
            min_cluster_fraction=min_cluster_fraction,
            max_cluster_fraction=max_cluster_fraction,
        )

        # select step
        selected_description_idxes_ = select(
            cluster_algo=cluster_algo,
            text_descriptions_matching=pruned_text_descriptions_matching,
            num_clusters=cluster_num_clusters,
            overlap_penalty=cluster_overlap_penalty,
            not_cover_penalty=cluster_not_cover_penalty,
        )

        selected_descriptions = [
            pruned_descriptions[i] for i in selected_description_idxes_
        ]
        selected_text_descriptions_matching = pruned_text_descriptions_matching[
            :, selected_description_idxes_
        ]
        unselected_text_indicators = (
            np.max(selected_text_descriptions_matching, axis=1) == 0
        )

        # force assign texts to one cluster
        cluster_assignment = np.ones(n, dtype=np.int64) * -1
        cluster_assignment[~unselected_text_indicators] = np.argmax(
            selected_text_descriptions_matching[~unselected_text_indicators], axis=1
        )
        recorder.record_select(
            selected_descriptions,
            cluster_assignment,
            "select",
        )

        num_iterations += 1
        recorder.next_iteration()

    if assigner_for_final_assignment_template:
        selected_text_descriptions_matching = assign(
            descriptions=selected_descriptions,
            texts=problem.texts,
            assigner=assigner,
            template=assigner_for_final_assignment_template,
            use_multi_assigner=True,
            add_null_description=False,
            progress_bar=True,
        )
        cluster_assignment = np.argmax(selected_text_descriptions_matching, axis=1)
        recorder.record_select(selected_descriptions, cluster_assignment, "final")

    description2texts = {
        description: [
            problem.texts[i]
            for i in range(n)
            if selected_text_descriptions_matching[i, j] == 1
        ]
        for j, description in enumerate(selected_descriptions)
    }

    return description2texts


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="./processed_data/syngoalex/topic",
    )
    parser.add_argument("--exp_dir", type=str, default="./experiments")
    parser.add_argument("--subsample", type=int, default=0)
    parser.add_argument("--chunk_text_to_words", type=int, default=None)
    parser.add_argument("--turn_off_approval_before_running", action="store_true")
    parser.add_argument("--with_labels", action="store_true")

    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--min_cluster_fraction", type=float, default=0.0)
    parser.add_argument("--max_cluster_fraction", type=float, default=0.4)

    parser.add_argument("--proposer_model", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--proposer_num_descriptions_to_propose", type=int, default=30)
    parser.add_argument("--proposer_num_rounds_to_propose", type=int, default=None)
    parser.add_argument("--proposer_num_descriptions_per_round", type=int, default=8)
    parser.add_argument(
        "--proposer_template",
        type=str,
        default="templates/gpt_cluster_proposer_short.txt",
    )

    parser.add_argument("--assigner_name", type=str, default="google/flan-t5-xl")
    parser.add_argument(
        "--assigner_for_proposed_descriptions_template",
        type=str,
        default="templates/t5_assigner.txt",
    )
    parser.add_argument(
        "--assigner_for_final_assignment_template", type=str, default=None
    )

    parser.add_argument("--cluster_algo", type=str, default="maximum_set_coverage")
    parser.add_argument("--cluster_num_clusters", type=int, default=None)
    parser.add_argument("--cluster_overlap_penalty", type=float, default=0.5)
    parser.add_argument("--cluster_not_cover_penalty", type=float, default=1.0)

    parser.add_argument("--iterative_max_rounds", type=int, default=5)
    parser.add_argument("--iterative_stop_criteria", type=int, default=None)

    args = parser.parse_args()

    with open(os.path.join(args.data_path, "data.json"), "r") as f:
        problem = Problem.from_json(f.read())

    label = None
    if args.with_labels:
        with open(os.path.join(args.data_path, "labels.json"), "r") as f:
            label = Label.from_json(f.read())

    if args.cluster_num_clusters is not None and args.cluster_num_clusters <= 0:
        args.cluster_num_clusters = len(label.class_descriptions)

    if args.subsample > 0:
        problem.texts = problem.texts[: args.subsample]
        if label is not None:
            label.labels = label.labels[: args.subsample]

    # shorten the text in the problems, this will only work for languages with spaces.
    if args.chunk_text_to_words is not None:
        for i in range(len(problem.texts)):
            problem.texts[i] = " ".join(
                problem.texts[i].split()[: args.chunk_text_to_words]
            )

    if label is not None:
        label_cnts = Counter(label.labels)
        print(
            "\n".join(
                [
                    f"{label.class_descriptions[i]}: {label_cnts[i]}"
                    for i in range(len(label.class_descriptions))
                ]
            )
        )

    # add time to exp_dir
    args.exp_dir += f"/{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    os.makedirs(args.exp_dir, exist_ok=True)
    with open(os.path.join(args.exp_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    cluster_result = run(
        problem=problem,
        exp_dir=args.exp_dir,
        label=label,
        random_seed=args.random_seed,
        verbose=args.verbose,
        min_cluster_fraction=args.min_cluster_fraction,
        max_cluster_fraction=args.max_cluster_fraction,
        proposer_model=args.proposer_model,
        proposer_num_descriptions_to_propose=args.proposer_num_descriptions_to_propose,
        proposer_num_rounds_to_propose=args.proposer_num_rounds_to_propose,
        proposer_num_descriptions_per_round=args.proposer_num_descriptions_per_round,
        proposer_template=args.proposer_template,
        assigner_name=args.assigner_name,
        assigner_for_proposed_descriptions_template=args.assigner_for_proposed_descriptions_template,
        assigner_for_final_assignment_template=args.assigner_for_final_assignment_template,
        cluster_algo=args.cluster_algo,
        cluster_num_clusters=args.cluster_num_clusters,
        cluster_overlap_penalty=args.cluster_overlap_penalty,
        cluster_not_cover_penalty=args.cluster_not_cover_penalty,
        iterative_max_rounds=args.iterative_max_rounds,
        iterative_stop_criteria=args.iterative_stop_criteria,
        approve_cost_before_running=not args.turn_off_approval_before_running,
    )

    with open(os.path.join(args.exp_dir, "cluster_result.json"), "w") as f:
        f.write(json.dumps(cluster_result, indent=2))

    print("Clustering results saved at: ", args.exp_dir)