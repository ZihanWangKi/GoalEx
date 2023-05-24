import os
from tqdm import tqdm
import numpy as np
from typing import List
from dataclasses import dataclass
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from tqdm import tqdm
from utils import ChatGPTWrapperWithCost, gpt3wrapper_texts_batch_iter, parse_template
import json
from functools import partial


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_TARGET_LENGTH = 2
YES_NO_TOK_IDX = [150, 4273]
MAX_SOURCE_LENGTH = 1024
TEMPERATURE = 0.001
sm = torch.nn.Softmax(dim=-1)

GPT_TEMPLATE = "templates/gpt_assigner.txt"
GPT_MULTI_TEMPLATE_ONE_OUTPUT = "templates/gpt_multi_assigner_one_output.txt"

T5_TEMPLATE = "templates/t5_assigner.txt"
T5_MULTI_TEMPLATE_ONE_OUTPUT = "templates/t5_multi_assigner_one_output.txt"


@dataclass
class AssignerInput:
    """A assigner input, consisting of a candidate_explanation and a text."""

    candidate_explanation: str
    text: str


@dataclass
class MultiAssignerInput:
    """A assigner input, consisting of a list of candidate_explanation and a text."""

    candidate_explanation: List[str]
    text: str


def create_prompt_inputs_for_single_assigner(
    template: str, assigner_inputs: List[AssignerInput]
):
    """
    A helper function to create the prompt inputs for single assigner.

    Parameters
    ----------
    template: str
        The template used for assignment
    assigner_inputs : List[AssignerInput]
        A list of AssignerInput.

    Returns
    -------
    List[str]
        A list of prompts.
    """
    template = parse_template(template)
    prompts = []
    for assigner_dict in assigner_inputs:
        prompt = template.format(
            candidate_explanation=assigner_dict.candidate_explanation,
            text=assigner_dict.text,
        )
        prompts.append(prompt)
    return prompts


def create_prompt_inputs_for_multi_assigner(
    template: str,
    assigner_inputs: List[MultiAssignerInput],
    add_null_description: bool = True,
):
    """
    A helper function to create the prompt inputs for multi assigner.

    Parameters
    ----------
    template: str
        The template used for assignment
    assigner_inputs : List[MultiAssignerInput]
        A list of MultiAssignerInput.
    add_null_description : bool
        Whether to add a null description to the candidate_explanation.

    Returns
    -------
    List[str]
        A list of prompts.
    """
    template = parse_template(template)
    prompts = [
        template.format(
            descriptions_with_index="\n".join(
                [
                    f"{i}. {description}"
                    for i, description in enumerate(
                        (input.candidate_explanation + ["none of the above"])
                        if add_null_description
                        else input.candidate_explanation
                    )
                ]
            ),
            text=input.text,
        )
        for input in assigner_inputs
    ]
    return prompts


def parse_mutli_assigner_output(response, num_descriptions):
    """
    A parser for multi assigner, which parses the response into a list of 0/1, where 1 means the description is satisfied.
    The expected format of the response is a list of integers, stringified.

    Parameters
    ----------
    response : str
        The response from the model to be parsed.
    num_descriptions : int
        The number of descriptions, for the backup choice of empty parsed result.

    Returns
    -------
    List[int]
        A list of 0/1, where 1 means the description is satisfied.
    """
    try:
        # this can deal with some errors
        opening_bracket = response.find("[")
        closing_bracket = response.find("]")
        response = response[opening_bracket : closing_bracket + 1]
        answer = json.loads(response)
        assert isinstance(answer, list)
        answer = list(map(int, answer))
        matched = [0] * num_descriptions
        for i in answer:
            if i < num_descriptions:
                matched[i] = 1
        return matched
    except Exception as e:
        # empty is kind of frequent, so let's not alert in this case
        if response.strip() == "":
            return [0] * num_descriptions
        print(
            "Assigner failed to parse the response. Treating it as empty. Error message: ",
            e,
        )
        print("The response to parse:", response)
        return [0] * num_descriptions


class Assigner:
    """A assigner to assign a candidate_explanation given a text; abstract class."""

    def __init__(self, model_name, verbose=False):
        """
        Parameters
        ----------
        model_name : str
            The name of the model or the path to the model weights used for assignment
        verbose : bool
            Whether to print some logs (e.g., costs).
        """
        self.model_name = os.path.basename(model_name)
        self.verbose = verbose

    def obtain_single_assigner_scores(
        self,
        template: str,
        assigner_inputs: List[AssignerInput],
    ):
        """
        Given a list of AssignerInput, return a list of scores, which the i-th score is how well the i-th text satisfies the candidate_explanation.

        Parameters
        ----------
        template: str
            The template used for assignment
        assigner_inputs : List[AssignerInput]
            A list of AssignerInput.

        Returns
        -------
        List[float]
            A list of scores, which the i-th score is how well the i-th text satisfies the candidate_explanation.
        """
        raise NotImplementedError

    def obtain_multi_assigner_scores(
        self,
        template: str,
        assigner_inputs: List[MultiAssignerInput],
        add_null_description: bool = True,
    ):
        """
        Given a list of MultiAssignerInput, returns a list of lists of scores, which the i-th list of scores is how well the i-th text satisfies the i-th hypotheses.

        Parameters
        ----------
        template: str
            The template used for assignment
        assigner_inputs : List[MultiAssignerInput]
            A list of AssignerInput.
        add_null_description : bool
            Whether to add a null description to the candidate_explanation.

        Returns
        -------
        List[float]
            A list of scores, which the i-th list of scores is how well the i-th text satisfies the i-th hypotheses.
        """
        raise NotImplementedError


class T5Assigner(Assigner):
    """A assigner based on T5 model to assign a candidate_explanation given a text"""

    BATCH_SIZE = 10

    def __init__(
        self,
        model_name: str = "google/flan-t5-xl",
        verbose: bool = False,
        batch_size: int = BATCH_SIZE,
    ):
        """
        Initialize the assigner

        Parameters
        ----------
        model_name : str
            The name of the T5 model or the path to the T5 model weights used for assignment
        template : str
            The template used for assignment
        verbose : bool
            Whether to print some logs (e.g., model loading).
        batch_size : int
            The batch size used for assignment
        """
        super().__init__(model_name=model_name, verbose=verbose)
        self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
        if self.verbose:
            print(f"Loading model weights for T5 assigner {model_name}...", end="")
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.parallelize_across_device()
        if self.verbose:
            print("Done.")
        self.batch_size = batch_size

    def parallelize_across_device(self):
        """Parallelize the model across devices if multiple GPUs are available"""
        num_heads = len(self.model.encoder.block)
        num_device = torch.cuda.device_count()
        other_device_alloc = num_heads // num_device + 1
        first_device = num_heads - (num_device - 1) * other_device_alloc
        device_map = {}
        cur = 0
        end = max(cur + first_device, 1)
        device_map[0] = list(range(cur, end))
        cur = end
        for i in range(1, num_device):
            end = min(cur + other_device_alloc, num_heads)
            device_map[i] = list(range(cur, end))
            cur += other_device_alloc
        print("device_map", device_map)
        self.model.parallelize(device_map)

    def batch_inference(self, prompts: List[str], max_new_tokens=1):
        """
        Given a list of prompts, return a list of generated results in a batch manner.

        Parameters
        ----------
        prompts : List[str]
            A list of prompts.
        max_new_tokens : int
            The maximum number of tokens to be generated.

        Returns
        -------
        List[str]
            A list of generated results.

        """
        with torch.no_grad():
            self.model.eval()
            num_batches = (len(prompts) - 1) // self.batch_size + 1

            for batch_idx in range(num_batches):
                input_prompts = prompts[
                    batch_idx * self.batch_size : (batch_idx + 1) * self.batch_size
                ]
                inputs = self.tokenizer(
                    input_prompts,
                    return_tensors="pt",
                    padding="longest",
                    max_length=MAX_SOURCE_LENGTH,
                    truncation=True,
                ).to(device)
                generation_result = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    do_sample=True,
                    temperature=0.001,
                    max_new_tokens=10,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                yield generation_result

    def obtain_single_assigner_scores(
        self, template: str, assigner_inputs: List[AssignerInput]
    ) -> List[float]:
        prompts = create_prompt_inputs_for_single_assigner(template, assigner_inputs)
        if self.verbose:
            print("First prompt as an example to assigner:")
            print(prompts[0])
        for generation_result in self.batch_inference(prompts):
            scores = (
                sm(generation_result.scores[0][:, YES_NO_TOK_IDX])[:, 1]
                .detach()
                .cpu()
                .numpy()
                .tolist()
            )
            for s in scores:
                yield s

    def obtain_multi_assigner_scores(
        self,
        template: str,
        assigner_inputs: List[MultiAssignerInput],
        add_null_description: bool = True,
    ) -> List[List[float]]:
        num_descriptions = len(assigner_inputs[0].candidate_explanation)
        assert all(
            len(input.candidate_explanation) == num_descriptions
            for input in assigner_inputs
        )

        prompts = create_prompt_inputs_for_multi_assigner(
            template, assigner_inputs, add_null_description
        )
        if self.verbose:
            print("First prompt as an example to assigner:")
            print(prompts[0])
        for generation_result in self.batch_inference(prompts, max_new_tokens=10):
            responses = self.tokenizer.batch_decode(
                generation_result.sequences, skip_special_tokens=True
            )
            for response in responses:
                yield parse_mutli_assigner_output(response, num_descriptions)


class GPTAssigner(Assigner):
    def __init__(
        self,
        model_name: str,
        verbose: bool = False,
    ):
        """
        Initialize the assigner

        Parameters
        ----------
        model_name : str
            The name of the gpt model
        verbose : bool
            Whether to print some logs (e.g., cost).
        """
        super().__init__(model_name=model_name, verbose=verbose)
        self.model = model_name

    def obtain_single_assigner_scores(
        self, template: str, assigner_inputs: List[AssignerInput]
    ) -> List[float]:
        prompts = create_prompt_inputs_for_single_assigner(template, assigner_inputs)
        if self.model in ("gpt-4", "gpt-3.5-turbo"):
            chat_gpt = ChatGPTWrapperWithCost()
            for prompt in prompts:
                response = chat_gpt(prompt=prompt, model=self.model, temperature=0.0)
                yield 1 if "yes" in response[0].lower() else 0
            if self.verbose:
                print("GPTAssigner Cost", chat_gpt.cost, "$")
        elif self.model.startswith("text-davinci"):
            for text_response in gpt3wrapper_texts_batch_iter(
                prompt=prompts, model=self.model, temperature=0.0
            ):
                yield 1 if "yes" in text_response.lower() else 0
            if self.verbose:
                print("Haven't implemented the cost function for text-davinci yet.")

    def obtain_multi_assigner_scores(
        self,
        template: str,
        assigner_inputs: List[MultiAssignerInput],
        add_null_description: bool = True,
    ) -> List[List[float]]:
        num_descriptions = len(assigner_inputs[0].candidate_explanation)
        assert all(
            len(input.candidate_explanation) == num_descriptions
            for input in assigner_inputs
        )

        prompts = create_prompt_inputs_for_multi_assigner(
            template, assigner_inputs, add_null_description
        )
        if self.model in ("gpt-4", "gpt-3.5-turbo"):
            chat_gpt = ChatGPTWrapperWithCost()
            for prompt in prompts:
                responses = chat_gpt(prompt=prompt, model=self.model, temperature=0.0)
                yield parse_mutli_assigner_output(responses[0], num_descriptions)
            if self.verbose:
                print("GPTSingleAssigner Cost", chat_gpt.cost, "$")
        elif self.model.startswith("text-davinci"):
            for text_responses in gpt3wrapper_texts_batch_iter(
                prompt=prompts, model=self.model, temperature=0.0
            ):
                yield parse_mutli_assigner_output(text_responses, num_descriptions)
            if self.verbose:
                print("Haven't implemented the cost function for text-davinci yet.")


def assign_descriptions(
    descriptions: List[str],
    texts: List[str],
    assigner: Assigner,
    template: str,
    use_multi_assigner: bool = False,
    add_null_description: bool = True,
    progress_bar: bool = False,
) -> np.ndarray:
    """
    Given a list of descriptions and a list of texts, return a matrix of scores, which the i-th row and j-th column is how well the i-th text satisfies the j-th description.

    Parameters
    ----------
    descriptions : List[str]
        A list of descriptions to be assigned into.
    texts : List[str]
        A list of texts to be assigned.
    assigner : Assigner
        A assigner that can assign a list of AssignerInput. Could be either a T5Assigner or a GPTAssigner.
    template : str
        The template to be used for assignment.
    use_multi_assigner : bool, optional
        Whether to use a multi-assigner, by default False
    add_null_description : bool, optional
        Whether to add a null description when using multi assigner, by default True
    progress_bar : bool, optional
        Whether to show a progress bar, by default False

    Returns
    -------
    np.ndarray
        A matrix of scores, which the i-th row and j-th column is how well the i-th text satisfies the j-th description.
    """
    if not use_multi_assigner:
        assigner_inputs = []
        for text in texts:
            for description in descriptions:
                assigner_inputs.append(
                    AssignerInput(candidate_explanation=description, text=text)
                )
    else:
        assigner_inputs = []
        for text in texts:
            assigner_inputs.append(
                MultiAssignerInput(candidate_explanation=descriptions, text=text)
            )

    # obtain the scores
    scores = []

    if not use_multi_assigner:
        obtain_scores_fn = assigner.obtain_single_assigner_scores
    else:
        obtain_scores_fn = partial(
            assigner.obtain_multi_assigner_scores,
            add_null_description=add_null_description,
        )

    if progress_bar:
        pbar = tqdm(
            obtain_scores_fn(template=template, assigner_inputs=assigner_inputs),
            total=len(assigner_inputs),
            desc="Assigning",
        )
    else:
        pbar = obtain_scores_fn(template=template, assigner_inputs=assigner_inputs)

    for score in pbar:
        scores.append(score)

    # reshape the scores into a matrix
    # the i-th row and j-th column is how well the i-th text satisfies the j-th description
    scores = np.array(list(scores)).reshape(len(texts), len(descriptions))
    return scores


def get_assigner(assigner_name, verbose=False, **kwargs):
    gpt_assigner_names = ["gpt-4", "gpt-3.5-turbo"]
    if assigner_name in gpt_assigner_names:
        return GPTAssigner(assigner_name, verbose=verbose, **kwargs)
    else:
        return T5Assigner(assigner_name, verbose=verbose, **kwargs)


if __name__ == "__main__":

    texts = [
        "I like this film.",
        "I hate this film.",
        "I am neural against this film.",
        "I both like and hate this film.",
    ]
    descriptions = [
        "is positive in tone",
        "is negative in tone",
    ]

    for assigner in [
        "google/flan-t5-large",
        "gpt-3.5-turbo",
    ]:
        assigner = get_assigner(assigner, verbose=True)
        # single assigner
        scores = assign_descriptions(
            descriptions,
            texts,
            assigner,
            template="templates/gpt_assigner.txt",
            use_multi_assigner=False,
            add_null_description=False,
            progress_bar=True,
        )
        print(scores)

        # multi assigner
        scores = assign_descriptions(
            descriptions,
            texts,
            assigner,
            template="templates/gpt_multi_assigner.txt",
            use_multi_assigner=True,
            add_null_description=True,
            progress_bar=True,
        )
        print(scores)

        # multi assigner
        scores = assign_descriptions(
            descriptions,
            texts,
            assigner,
            template="templates/gpt_multi_assigner_one_output.txt",
            use_multi_assigner=True,
            add_null_description=False,
            progress_bar=True,
        )
        print(scores)
