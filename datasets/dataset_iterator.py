
from datastructures import QuerySceneSequence, ResultsSceneSequence


class DatasetIterator():
    """
    This class is sets up and wraps every dataset.

    It constructs each dataset with kwargs, and then expects a standard API
    to load successive QuerySceneSequence and the associated ground truth
    ResultsSceneSequence for that dataset problem.
    """
    def __init__(self, name : str, args = dict()) -> None:
        
     