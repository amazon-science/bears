from typing import Dict, Set, Union

from autoenum import AutoEnum, auto


class MLType(AutoEnum):
    ## "Data" MLTypes:
    BOOL = auto()
    TEXT = auto()
    CATEGORICAL = auto()
    INT = auto()
    FLOAT = auto()
    VECTOR = auto()
    SPARSE_VECTOR = auto()
    TIMESTAMP = auto()
    TENSOR = auto()
    OBJECT = auto()

    ## "Asset" MLTypes:
    IMAGE = auto()
    AUDIO = auto()
    VIDEO = auto()

    ## "Document" MLTypes:
    PDF = auto()  ## For .txt documents, PDFs, etc

    ## Schema MLTypes:
    INDEX = auto()
    GROUND_TRUTH = auto()
    PREDICTED_LABEL = auto()
    PREDICTED_PROBABILITY = auto()
    PREDICTED = auto()

    ## Ground truth label(s):
    GROUND_TRUTH_LABEL = auto()  ## TODO: Delete this.
    GROUND_TRUTH_LABEL_LIST = auto()
    GROUND_TRUTH_LABEL_COMMA_SEPARATED = auto()
    GROUND_TRUTH_LABEL_COMMA_SEPARATED_OR_LIST = auto()
    ENCODED_LABEL = auto()
    ENCODED_LABEL_LIST = auto()
    ENCODED_LABEL_COMMA_SEPARATED = auto()
    ENCODED_LABEL_COMMA_SEPARATED_OR_LIST = auto()

    ## Predicted label(s):
    PREDICTED_LABEL_COMMA_SEPARATED_OR_LIST = auto()
    ENCODED_PREDICTED_LABEL = auto()

    ## Predicted probability score(s):
    PROBABILITY_SCORE = auto()
    PROBABILITY_SCORE_COMMA_SEPARATED_OR_LIST = auto()
    PREDICTED_CORRECT = auto()
    PREDICTION_IS_CONFIDENT = auto()
    ## Each element stores a list [predicted_label, predicted_score, is_confident]:
    PREDICTED_LABEL_PREDICTED_SCORE_IS_CONFIDENT_VECTOR = auto()


DATA_ML_TYPES: Set[MLType] = {
    MLType.BOOL,
    MLType.TEXT,
    MLType.CATEGORICAL,
    MLType.INT,
    MLType.FLOAT,
    MLType.VECTOR,
    MLType.SPARSE_VECTOR,
    MLType.TIMESTAMP,
    MLType.TENSOR,
}

ASSET_ML_TYPES: Set[MLType] = {
    MLType.IMAGE,
    MLType.AUDIO,
    MLType.VIDEO,
}

DOCUMENT_ML_TYPES: Set[MLType] = {
    MLType.PDF,
}

PREDICTED_ML_TYPES: Set[MLType] = {
    MLType.PREDICTED,
    MLType.PREDICTED_LABEL,
    MLType.PREDICTED_PROBABILITY,
}

GROUND_TRUTH_ML_TYPES: Set[MLType] = {
    MLType.GROUND_TRUTH,
    MLType.GROUND_TRUTH_LABEL,
}

MLTypeSchema = Dict[str, MLType]

MLTypeOrStr = Union[MLType, str]


DASK_APPLY_OUTPUT_MLTYPE_TO_META_MAP = {
    MLType.BOOL: bool,
    MLType.TEXT: str,
    MLType.INT: int,
    MLType.FLOAT: float,
    MLType.VECTOR: list,
}
