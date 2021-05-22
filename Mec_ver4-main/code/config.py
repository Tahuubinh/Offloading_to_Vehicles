import os
from pathlib import Path

LINK_PROJECT = Path(os.path.abspath(__file__))
LINK_PROJECT = LINK_PROJECT.parent.parent
#print(LINK_PROJECT)
DATA_DIR = os.path.join(LINK_PROJECT, "data")
RESULT_DIR = os.path.join(LINK_PROJECT, "result")
DATA_TASK = os.path.join(LINK_PROJECT, "data_task")
class Config:
    Pr = 46
    Pr2 = 24
    Wm = 10
    length_hidden_layer=4
    n_unit_in_layer=[16, 32, 32, 8]
    
