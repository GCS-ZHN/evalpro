# coding=utf-8
"""
Perplexity module
"""

import evaluate
import datasets
import numpy as np

from typing import List, Dict, Any
from evalpro import utils
from scipy.special import log_softmax


class Perplexity(evaluate.EvaluationModule):
    '''
    Compute the perplexity of a language model. The perplexity is a measure
    of how well a probability model predicts a sample. The higher the perplexity,
    the worse the model is at predicting the sample. Mathematical equation:
    $$
        \\text{Perplexity} = \exp(-\\frac{1}{N} \sum_{i=1}^{N} \log p(x_i))
    $$
    where \(N\) is the number of tokens in the sample, and \(p(x_i)\) is
    the probability assigned by the model to the \(i\)-th token.

    Init Args:
        vocab_size (int): The size of the vocabulary of the language model.
    
    Sample Args:
        predictions (np.ndarray): The predicted logits of the language model.
            The shape of the array should be (N, vocab_size), where N is the
            number of tokens in the sample.
        references (List[int]): The reference tokens of the sample.

    Returns:
        perplexity_avg (float): The average perplexity of the predictions.
    
    Examples:
        >>> metric = Perplexity(vocab_size=10)
        >>> metric.add(
                prediction=np.log(np.diag(np.ones(10)) + 1e-10),
                reference=list(range(10))
            )
        >>> metric.compute()
        {'perplexity_avg': 1.0000000009000003}

    '''

    def __init__(self, vocab_size: int, *args, **kwargs):
        self.vocab_size = vocab_size
        super().__init__(*args, **kwargs)
 
    def _info(self) -> evaluate.MetricInfo:
        return evaluate.MetricInfo(
            description=utils.get_docstring_description(Perplexity),
            citation="",
            features=datasets.Features({
                'predictions': datasets.Array2D((None, self.vocab_size), dtype='float32'),
                'references': datasets.Sequence(datasets.Value('int32'))
            }),
            homepage="https://bookstall.github.io/2024/06/17/perplexity/"
        )
    
    def _calc_perplexity(self, logits: np.ndarray, target_idxs: List[int]) -> float:
        log_probs = log_softmax(logits, axis=-1)
        target_log_probs = log_probs[np.arange(len(target_idxs)), target_idxs]
        return np.exp(-np.mean(target_log_probs))

    def _compute(self, predictions: List[np.ndarray], references: List[List[int]]) -> Dict[str, Any]:
        perplexities = [self._calc_perplexity(pred, ref) for pred, ref in zip(predictions, references)]
        return {'perplexity': np.mean(perplexities)}
