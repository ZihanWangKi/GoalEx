from __future__ import annotations
import json
import os
import numpy as np
from iterative_cluster import run
from cluster_problem import ClusterProblem as Problem
from dataclasses import dataclass
from dataclasses_json import dataclass_json
from typing import Dict


with open("templates/subgoal.txt") as f:
    SUBGOAL_TEMPLATE = f.read()


@dataclass_json
@dataclass
class Taxonomy:
    """
    A taxonomy is a tree structure that represents the clustering results. This represents a node in the tree.
    """

    node_name: str  # The name of the node. Naming convention is a sequence of description hashes of its ancestors.
    description: str  # The description/explanation of the node.
    texts: list  # The texts in this node.
    description2taxonomy: Dict[
        str, Taxonomy
    ] = None  # The children of this node. The key is the explanation and the value is the child node.


def cluster_subtree(
    taxonomy: Taxonomy,
    root_goal: str,
    proposer_model: str = "gpt-4",
    assigner_name: str = "claude-v1.3",
    num_clusters: int = 8,
    overlap_penalty: float = 0.5,
    not_cover_penalty: float = 1,
    random_seed: int = 0,
    iterative_max_rounds: int = 3,
    minimal_size: int = 20,
):
    """
    Given a node in the taxonomy, cluster the texts under this node. The results will be stored in place in the description2taxonomy field of the taxonomy.

    Parameters
    ----------
    taxonomy: Taxonomy
        The node in the taxonomy tree to cluster.
    root_goal: str
        The goal of the root of the entire taxonomy tree.
    proposer_model: str
        The model to use for the proposer.
    assigner_name: str
        The name of the assigner to use.
    num_clusters: int
        The number of clusters to cluster the texts into.
    overlap_penalty: float
        The penalty for overlapping clusters. \lambda in the paper.
    not_cover_penalty: float
        The penalty for not covering all the texts. by default 1
    random_seed: int
        The random seed to use.
    iterative_max_rounds: int
        The maximum number of rounds to run the iterative clustering algorithm.
    minimal_size: int
        The minimal size of a cluster. If the size of a sub-cluster is smaller than this, we will not cluster it further.
    """
    if len(taxonomy.texts) < minimal_size:
        return

    new_goal = (
        root_goal
        if "node" not in taxonomy.node_name
        else SUBGOAL_TEMPLATE.format(orig_goal=root_goal, parent=taxonomy.description)
    )
    problem = Problem(
        texts=taxonomy.texts,
        goal=new_goal,
        example_descriptions=[],
    )

    description2texts = run(
        problem=problem,
        exp_dir=f"experiments/{taxonomy.node_name}",
        proposer_model=proposer_model,
        proposer_template="templates/gpt_cluster_proposer_detailed.txt",
        assigner_name=assigner_name,
        assigner_for_proposed_descriptions_template="templates/gpt_assigner.txt",
        cluster_num_clusters=num_clusters,
        cluster_overlap_penalty=overlap_penalty,
        cluster_not_cover_penalty=not_cover_penalty,
        random_seed=random_seed,
        iterative_max_rounds=iterative_max_rounds,
        approve_cost_before_running=False
    )

    for selected_description, texts in description2texts.items():
        taxonomy.description2taxonomy[selected_description] = Taxonomy(
            description=selected_description,
            texts=texts,
            description2taxonomy={},
            node_name=f"{taxonomy.node_name}_node={hash(selected_description)}",
        )


def print_taxonomy(taxonomy: Taxonomy, indent: int = 0):
    """
    Print the taxonomy tree.

    Parameters
    ----------
    taxonomy: Taxonomy
        The root of the taxonomy tree to print.
    indent: int
        The indentation level.
    """
    print(" " * indent, taxonomy.node_name, taxonomy.description)
    for desc, taxonomy in taxonomy.description2taxonomy.items():
        print_taxonomy(taxonomy, indent=indent + 2)


def depth2_clustering(
    problem: Problem,
    problem_id: int = -1,
    proposer_model: str = "gpt-3.5-turbo",
    assigner_name: str = "gpt-3.5-turbo",
    num_clusters: int = 8,
    overlap_penalty: float = 0.5,
    not_cover_penalty: float = 1,
    random_seed: int = 0,
    iterative_max_rounds: int = 3,
    minimal_size: int = 20,
) -> Taxonomy:
    """
    Cluster the texts in the problem into a taxonomy tree. The clustering is done in a depth-2 manner.

    Parameters
    ----------
    problem: Problem
        The problem to cluster.
    problem_id: int
        The id of the problem.
    proposer_model: str
        The model to use for the proposer.
    assigner_name: str
        The name of the assigner to use.
    num_clusters: int
        The number of clusters to cluster the texts into.
    overlap_penalty: float
        The penalty for overlapping clusters. \lambda in the paper.
    not_cover_penalty: float
        The penalty for not covering all the texts. by default 1
    random_seed: int
        The random seed to use.
    iterative_max_rounds: int
        The maximum number of rounds to run the iterative clustering algorithm.

    """
    root_taxonomy = Taxonomy(
        node_name=f"root_id={problem_id}",
        description="root",
        texts=problem.texts,
        description2taxonomy={},
    )
    cluster_subtree(
        taxonomy=root_taxonomy,
        root_goal=problem.goal,
        proposer_model=proposer_model,
        assigner_name=assigner_name,
        num_clusters=num_clusters,
        overlap_penalty=overlap_penalty,
        not_cover_penalty=not_cover_penalty,
        random_seed=random_seed,
        iterative_max_rounds=iterative_max_rounds,
    )
    print_taxonomy(root_taxonomy)

    for desc, taxonomy in root_taxonomy.description2taxonomy.items():
        cluster_subtree(
            taxonomy=taxonomy,
            root_goal=problem.goal,
            proposer_model=proposer_model,
            assigner_name=assigner_name,
            num_clusters=num_clusters,
            overlap_penalty=overlap_penalty,
            not_cover_penalty=not_cover_penalty,
            random_seed=random_seed,
            iterative_max_rounds=iterative_max_rounds,
            minimal_size=minimal_size,
        )
        # print the entire taxonomy tree after clustering each subtree
        print_taxonomy(root_taxonomy)

    return root_taxonomy


if __name__ == "__main__":

    EXPERIMENT_DIR = "experiments/"
    cluster_problem_fs = os.listdir(EXPERIMENT_DIR)
    orig_problem_path = "processed_data/real_data/clustering_problems_for_paper.json"
    with open(orig_problem_path) as f:
        orig_problems = json.load(f)
    DEBUG = False
    if DEBUG:
        problem = orig_problems[-1]
        problem["texts"] = problem["texts"][:50]
        problem = Problem.from_dict(problem)
        save_path = f"taxonomy/taxonomy_debug.json"
        taxonomy_dict = depth2_clustering(
            problem,
            iterative_max_rounds=1,
            num_clusters=4,
            overlap_penalty=0.5,
            not_cover_penalty=1,
            random_seed=0,
            minimal_size=3,
            assigner_name="google/flan-t5-xl",
            proposer_model="gpt-3.5-turbo",
        ).to_dict()
        with open(save_path, "w") as f:
            json.dump(taxonomy_dict, f, indent=2)
    else:
        for problem_id, problem in enumerate(orig_problems):
            save_path = f"taxonomy/taxonomy_{problem_id}.json"
            if os.path.exists(save_path):
                continue
            problem["texts"] = problem["texts"][:400]

            problem = Problem.from_dict(problem)
            taxonomy_dict = depth2_clustering(problem, problem_id).to_dict()
            with open(save_path, "w") as f:
                json.dump(taxonomy_dict, f, indent=2)
