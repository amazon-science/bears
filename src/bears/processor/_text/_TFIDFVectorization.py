from ast import literal_eval
from typing import Dict, Optional, Union

import numpy as np
from bears import ScalableSeries
from bears.util import if_else, is_dict_like
from pydantic import model_validator
from scipy.sparse import csr_matrix as SparseCSRMatrix
from sklearn.feature_extraction.text import TfidfVectorizer

from bears.constants import MLType
from bears.processor import SingleColumnProcessor, TextInputProcessor
from bears.processor._vector._VectorDensifier import VectorDensifier


class TFIDFVectorization(SingleColumnProcessor, TextInputProcessor):
    """
    Performs TF-IDF Vectorization of a text column using sklearn's TFIDFVectorizer.
    Ref: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    Params:
    - OUTPUT_SPARSE: whether to output each row as a sparse row matrix (1 x N). If False, will output a 1d numpy array.
    - SKLEARN_PARAMS: dictionary of sklearn params to be unpacked as keyword arguments to the constructor
        sklearn.feature_extraction.text.TfidfVectorizer. Thus, keys are case-sensitive.
    """

    class Params(SingleColumnProcessor.Params):
        sklearn_params: Dict = {}
        output_sparse: bool = False

        @model_validator(mode="before")
        @classmethod
        def set_params(cls, params: Dict) -> Dict:
            cls.set_default_param_values(params)
            sklearn_tfidf_params: Dict = params["sklearn_params"]
            if not is_dict_like(sklearn_tfidf_params):
                raise ValueError("`sklearn_params` should be a dictionary")
            token_pattern: Optional = sklearn_tfidf_params.get("token_pattern")
            if token_pattern is not None:
                sklearn_tfidf_params["token_pattern"] = str(sklearn_tfidf_params.get("token_pattern"))
            ngram_range: Optional = sklearn_tfidf_params.get("ngram_range")
            if ngram_range is not None:
                if isinstance(ngram_range, str):
                    ngram_range = literal_eval(ngram_range)
                if isinstance(ngram_range, list):
                    ngram_range = tuple(ngram_range)
                assert isinstance(ngram_range, tuple)
                sklearn_tfidf_params["ngram_range"] = ngram_range
            params["sklearn_params"] = sklearn_tfidf_params
            return params

    output_mltype = MLType.VECTOR
    vectorizer: TfidfVectorizer = None
    vector_densifier: VectorDensifier = None

    @model_validator(mode="before")
    @classmethod
    def set_vectorizer(cls, params: Dict) -> Dict:
        cls.set_default_param_values(params)
        params["vectorizer"]: TfidfVectorizer = TfidfVectorizer(**params["params"].sklearn_params)
        params["vector_densifier"]: VectorDensifier = VectorDensifier()
        params["output_mltype"]: MLType = if_else(
            params["params"].output_sparse, MLType.SPARSE_VECTOR, MLType.VECTOR
        )
        return params

    def _fit_series(self, data: ScalableSeries):
        self.vectorizer.fit(data.pandas())  ## TODO: Super slow, replace with Dask TFIDF

    def transform_single(self, data: str) -> Union[SparseCSRMatrix, np.ndarray]:
        ## Will output a sparse matrix with only a single row.
        tfidf_vec: SparseCSRMatrix = self.vectorizer.transform([data])
        if not self.params.output_sparse:
            tfidf_vec: np.ndarray = self.vector_densifier.transform_single(tfidf_vec)
        return tfidf_vec
