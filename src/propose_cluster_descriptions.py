import json
import random
from tqdm import trange
import utils
from utils import parse_template
from cluster_problem import ClusterProblem as Problem
from typing import List, Union
from dataclasses import dataclass
from dataclasses_json import dataclass_json


def construct_proposer_prompt(
    text_samples: List[str],
    goal: str,
    example_descriptions: List[str],
    num_descriptions_per_round: int,
    template: str,
):
    """
    Construct the prompt for the proposer model.

    Parameters
    ----------
    text_samples : List[str]
        A list of text samples to be included in the prompt.
    goal : str
        The goal or objective the proposer model should follow.
    example_descriptions : List[str], optional
        A list of example descriptions provided for formatting reference.
    num_descriptions_per_round : int
        The number of descriptions the model should suggest.
    template : str
        The template used for proposing, can be the actual string, or path to template file

    Returns
    -------
    str
        The formatted prompt for the proposer model.
    """
    text_samples = [f"{i}. {text}" for i, text in enumerate(text_samples)]
    samples_in_prompt = "\n".join(text_samples)
    example_description_in_prompt = ""
    if example_descriptions and len(example_descriptions) > 0:
        example_description_in_prompt = "Here are some example predicates; the generated predicates should have a similar granularity and perspective:\n"
        example_description_in_prompt += (
            "\n"
            + "\n".join(
                f'"{example_description.lower()}"'
                for example_description in example_descriptions
            )
            + "\n"
        )
    prompt = parse_template(template).format(
        goal=goal,
        samples_in_prompt=samples_in_prompt,
        example_description_in_prompt=example_description_in_prompt,
        num_descriptions_per_round=num_descriptions_per_round,
    )
    return prompt


@dataclass_json
@dataclass
class ClusterProposerResponse:
    """
    The response from the proposer model.

    Attributes
    ----------
    descriptions : List[str]
        A list of descriptions for each cluster.
    proposer_prompt : str
        The prompt used for the proposer model.
    text_subset : List[str]
        The text samples used in the prompt.
    raw_response : str
        The raw response from the proposer model, before parsing.
    """

    descriptions: List[str]
    proposer_prompt: str
    text_subset: List[str]
    raw_response: str


def propose_descriptions(
    problem: Problem,
    num_samples: int,
    model: str,
    template: str,
    example_descriptions: List[str],
    num_descriptions_per_round: int,
    random_seed: int,
    log_propose_prompt=False,
) -> ClusterProposerResponse:
    """
    Propose descriptions for a given problem.

    Parameters
    ----------
    problem : Problem
        The problem instance.
    num_samples : int
        The number of text samples to be included in the prompt. T in the paper.
    model : str
        The model to use for proposing descriptions.
    template : str
        The template used for proposing, can be the actual string, or path to template file
    example_descriptions : List[str]
        A list of example descriptions provided for formatting reference.
    num_descriptions_per_round : int
        The number of descriptions the model should suggest. J' in the paper.
    random_seed : int
        The random seed for sampling text samples.
    log_propose_prompt : bool
        Whether to log the prompt used for proposing.

    Returns
    -------
    ClusterProposerResponse
        The response from the proposer model. This includes the descriptions, the prompt, and the text samples used in the prompt.
    """
    # set the random seed
    random.seed(random_seed)

    # get the goal and text samples
    goal = problem.goal
    text_subset = random.sample(problem.texts, min(num_samples, len(problem.texts)))

    # construct the prompt based on the text samples and the goal
    proposer_prompt = construct_proposer_prompt(
        text_subset,
        goal,
        example_descriptions,
        num_descriptions_per_round,
        template,
    )

    # get the response from the model
    if log_propose_prompt == 0:
        print("Running the proposer model...")
        print(f"{proposer_prompt}")
    chat_gpt_query_model = utils.ChatGPTWrapperWithCost()
    raw_response = chat_gpt_query_model(
        prompt=proposer_prompt, model=model, temperature=0.2
    )
    if log_propose_prompt == 0:
        print("Proposer model response:")
        print(raw_response)
    if raw_response is None:
        return ClusterProposerResponse(
            descriptions=[],
            proposer_prompt=proposer_prompt,
            text_subset=text_subset,
            raw_response="",
        )
    text_response = raw_response[0]

    # parse the response to get the descriptions
    # each description is separated by a newline, surrounded by quotes according to the prompt
    descriptions = utils.parse_description_responses(text_response)

    # the later ones could very likely be of lower quality.
    descriptions = descriptions[:num_descriptions_per_round]
    # return the descriptions, the prompt, and the text samples used in the prompt
    return ClusterProposerResponse(
        descriptions=descriptions,
        proposer_prompt=proposer_prompt,
        text_subset=text_subset,
        raw_response=text_response,
    )


def propose_descriptions_multi_round(
    problem: Problem,
    num_samples: int,
    model: str,
    template: str,
    example_descriptions: List[str],
    num_descriptions_to_propose: int = 30,
    num_rounds_to_propose: int = None,
    num_descriptions_per_round: int = 8,
    random_seed: int = 0,
    return_descriptions_only: bool = False,
) -> Union[List[str], List[ClusterProposerResponse]]:
    """
    Propose descriptions for a given problem in multiple rounds. Similar to applying the propose_descriptions function multiple times, but for all rounds except the 1st round, we feed a random subset of 2 descriptions (or the length of the example_descriptions) from the 1st round as example_descriptions. This is to encourage the model to generate descriptions that are different but similar from the previous rounds.

    Parameters
    ----------
    problem : Problem
        The problem instance.
    num_samples : int
        The number of text samples to be included in the prompt.
    model : str
        The model to use for proposing descriptions.
    template : str
        The name of the template to use for constructing the prompt.
    example_descriptions : List[str]
        A list of example descriptions provided for formatting reference.
    num_descriptions_to_propose: int
        The number of descriptions to propose in total across all rounds.
    num_descriptions_per_round : int
        The number of descriptions the model should suggest. If set, overwrites num_descriptions_to_propose
    num_rounds_to_propose : int
        The number of rounds to run the proposer model.
    random_seed : int
        The random seed for sampling text samples. For each round, we will increment the random seed by 1.
    return_descriptions_only : bool
        Whether to return only the descriptions or the full ClusterProposerResponse across multiple rounds.

    Returns
    -------
    Union[List[str], List[ClusterProposerResponse]]
        if return_descriptions_only is True, return a list of descriptions across multiple rounds. Otherwise, return a list of ClusterProposerResponse across multiple rounds.
    """
    if example_descriptions is None:
        example_descriptions = []
    proposer_responses = []
    if num_rounds_to_propose is None:
        pbar_total = num_descriptions_to_propose
    else:
        pbar_total = num_rounds_to_propose
    num_descriptions = 0
    # propose descriptions in multiple rounds
    pbar = trange(pbar_total, desc="Proposing descriptions")
    all_descriptions_no_duplicates = []
    for i in pbar:

        # for the first round, we use the example_descriptions provided
        if i == 0:
            proposer_response = propose_descriptions(
                problem=problem,
                num_samples=num_samples,
                random_seed=random_seed,
                example_descriptions=example_descriptions,
                num_descriptions_per_round=num_descriptions_per_round,
                model=model,
                template=template,
                log_propose_prompt=True,
            )
        # for the rest of the rounds, we use a random subset of 2 descriptions (or the length of the example_descriptions) from the 1st round as example_descriptions
        else:
            example_descriptions = random.sample(
                proposer_responses[0].descriptions,
                len(example_descriptions) if len(example_descriptions) != 0 else 2,
            )
            proposer_response = propose_descriptions(
                problem=problem,
                num_samples=num_samples,
                random_seed=random_seed + i,
                example_descriptions=example_descriptions,
                num_descriptions_per_round=num_descriptions_per_round,
                model=model,
                template=template,
                log_propose_prompt=False,
            )

        # collect the responses, find the new descriptions, and update the progress bar
        proposer_responses.append(proposer_response)
        new_descriptions = [
            description
            for description in proposer_response.descriptions
            if description not in all_descriptions_no_duplicates
        ]
        all_descriptions_no_duplicates.extend(new_descriptions)

        num_descriptions += len(new_descriptions)
        if num_rounds_to_propose is not None:
            pbar.update(1)
        else:
            pbar.update(len(new_descriptions))
        if (
            num_rounds_to_propose is None
            and num_descriptions >= num_descriptions_to_propose
        ):
            break

    # if return_descriptions_only is True, return a list of descriptions across multiple rounds. Otherwise, return a list of ClusterProposerResponse across multiple rounds.
    if return_descriptions_only:
        return [
            proposer_response.descriptions for proposer_response in proposer_responses
        ]
    else:
        return proposer_responses


if __name__ == "__main__":

    # create a problem instance
    # data_path = "tmp_debug_data.json"
    data_path = "simplified_goal_debug_data.json"
    with open(data_path, "r") as f:
        problem = Problem.from_json(f.read())

    NUM_SAMPLES = 20
    NUM_DESCRIPTIONS_PER_ROUND = 8
    MODEL = "gpt-3.5-turbo"
    RETURN_DESCRIPTIONS_ONLY = False
    NUM_DESCRIPTIONS_TO_PROPOSE = 18
    RANDOM_SEEDS = [0]  # , 5, 10, 15, 20]
    TEMPLATE = "templates/gpt_cluster_proposer_w_example.txt"

    for RANDOM_SEED in RANDOM_SEEDS:
        # propose descriptions
        proposer_responses = propose_descriptions_multi_round(
            problem=problem,
            num_samples=NUM_SAMPLES,
            random_seed=RANDOM_SEED,
            example_descriptions=problem.example_descriptions,
            num_descriptions_per_round=NUM_DESCRIPTIONS_PER_ROUND,
            num_descriptions_to_propose=NUM_DESCRIPTIONS_TO_PROPOSE,
            model=MODEL,
            # num_rounds_to_propose=NUM_ROUNDS,
            return_descriptions_only=RETURN_DESCRIPTIONS_ONLY,
            template=TEMPLATE,
        )

        # print the resulting descriptions
        all_descriptions = []
        for proposer_response in proposer_responses:
            if RETURN_DESCRIPTIONS_ONLY:
                all_descriptions.extend(proposer_response)
            else:
                all_descriptions.extend(proposer_response.descriptions)
        print(all_descriptions)

        # save the proposer_responses
        save_path = f"tmp_debug_proposer_responses_seed{RANDOM_SEED}.json"
        with open(save_path, "w") as f:
            # convert the proposer_responses to json
            proposer_responses = [
                proposer_response.to_dict() for proposer_response in proposer_responses
            ]
            f.write(json.dumps(proposer_responses, indent=4))
