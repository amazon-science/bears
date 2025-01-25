from typing import Dict, List, Optional

from bears.util import is_list_like, is_null
from pydantic import model_validator

from bears.processor import SingleColumnProcessor, TextInputProcessor, TextOutputProcessor


class StringRemoval(SingleColumnProcessor, TextInputProcessor, TextOutputProcessor):
    """
    Removes certain strings from each text string using str.replace() (no regex matching).

    Params:
    - REMOVAL_LIST: the list of strings to remove.
    """

    class Params(SingleColumnProcessor.Params):
        removal_list: List[str]

        @model_validator(mode="before")
        @classmethod
        def set_params(cls, params: Dict) -> Dict:
            removal_list = params.get("removal_list")
            if not is_list_like(removal_list) or len(removal_list) == 0:
                raise ValueError("`removal_list` should be a non-empty list of strings")
            return params

    def transform_single(self, data: Optional[str]) -> Optional[str]:
        if is_null(data):
            return None
        for s in self.params.removal_list:
            data: str = data.replace(s, "")
        return data
