from langchain_core.pydantic_v1 import BaseModel, Field, PrivateAttr, root_validator
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from typing import List, Dict
from enum import Enum
import uuid

class VariableType(Enum):
    decision = "decision"
    chance = "chance"
    utility = "utility"


test=[{'variable_name': 'Fire_Separation_Measures',
  'variable_type': 'decision',
  'values': ['Implement', 'Not_Implement']},
 {'variable_name': 'Fire_Spread',
  'variable_type': 'chance',
  'values': ['Rapid_Upward', 'Contained']},
 {'variable_name': 'Fire_Integrity_Compromise',
  'variable_type': 'chance',
  'values': ['Compromised', 'Intact']},
 {'variable_name': 'Utility',
  'variable_type': 'utility',
  'values': ['Prevent_Rapid_Upward_Spread']}]


extract_template = """\
For the following text, identify the fundamental goal inplied by the context (which may not directly appear in the text), and extract every variable, action or event. Output using the following format:

text: {text}

{format_instructions}
"""

def get_list_of_nodes():
    return a

class Node(BaseModel):
    variable_name: str = Field(description="Extract the core concept or variable related to decision-making, and output its name.")
    variable_type: VariableType = Field(description="What is the nature of the variable? If it is an aleatory variable that cannot be intervened, output 'chance'; if it can be intervened and represents a decision to be made, output 'decision'; if it reflects the fundamental goal of decision-maker in the setting, output 'utility'.")
    values: list = Field(description="Extract the possible values of the variable, complete with common knowledge if necessary, and output them as a Python list.")
    __id: uuid.UUID = PrivateAttr(default_factory=uuid.uuid4)

    def __init__(self, variable_name, variable_type, values):
        super().__init__(variable_name=variable_name, variable_type=variable_type, values=values)

    def __hash__(self):
        return self.__id

class List_of_Nodes(BaseModel):
    node_list: List[Node]

    def __init__(self, node_list):
        super().__init__(node_list=node_list)
    
    def find(self, variable_name):
        for node in self.node_list:
            if node.variable_name == variable_name:
                return node
        return None

class Arc(BaseModel):
    condition: str = Field(description="?")
    variable: str = Field(description="?")
    probabilities: Dict[str, Dict[str, float]] = Field(description="?")
    __id: uuid.UUID = PrivateAttr(default_factory=uuid.uuid4)

    def __init__(self, condition, variable, probabilities):
        super().__init__(condition=condition, variable=variable, probabilities=probabilities)
    
    def __hash__(self):
        return self.__id

    @root_validator()
    def check_probabilities(cls, info):
        probabilities = info.get("probabilities")
        list_of_nodes = get_list_of_nodes()
        node_condition = list_of_nodes.find(info.get("condition"))
        node_variable = list_of_nodes.find(info.get("variable"))
        for key, value in probabilities.items():
            if key not in node_condition.values:
                raise ValueError(f"Invalid value {key} for variable {node_condition.variable_name}.")
            if sum(value.values()) != 1:
                raise ValueError(f"Probabilities for variable {node_condition.variable_name} must sum to 1.")
            for key2, _ in value.items():
                if key2 not in node_variable.values:
                    raise ValueError(f"Invalid value {key2} for variable {node_variable.variable_name}.")
        return info

class List_of_Arcs(BaseModel):
    arc_list: List[Arc]

    def __init__(self, arc_list):
        super().__init__(arc_list=arc_list)

    def find(self, condition, variable):
        for arc in self.arc_list:
            if arc.condition == condition and arc.variable == variable:
                return arc
        return None
    

parser = JsonOutputParser(pydantic_object=List_of_Nodes)
parser.get_format_instructions()
prompt = PromptTemplate(
    template=extract_template,
    input_variables=["text"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)
print(prompt.format(text="The weather is sunny and warm."))
a=List_of_Nodes(node_list=test)
b=Arc(condition="Fire_Separation_Measures", variable="Fire_Spread", probabilities={"Implement": {"Rapid_Upward": 0.8, "Contained": 0.2},"Not_Implement": {"Rapid_Upward": 0.2, "Contained": 0.8}})