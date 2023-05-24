from typing import List
from dataclasses import dataclass
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class ClusterProblem:
    goal: str
    texts: List[str]
    example_descriptions: List[str]


@dataclass_json
@dataclass
class ClusterProblemLabel:
    class_descriptions: List[str]
    labels: List[int]