
# coding=utf-8
"""
Sequence Recovery Rate module
"""

import datasets
import evaluate

from typing import List, Dict
from evalpro import utils


class SeqRecoveryRate(evaluate.EvaluationModule):
    """
    This new metric is designed to calculate the average amino acid 
    sequence recovery rate (AAR) of a set of predictions.

    Sample Args:
        predictions: list of predicted sequences to score. Each prediction
            should be a string with the same length as the corresponding reference.
        references: list of reference sequences for each prediction. Each

    Returns:
        seq_recovery_rate: the average sequence recovery rate of the predictions.

    Examples:
        >>> metric = SeqRecoveryRate()
        >>> metric.add(prediction='AQ', reference='AQ')
        >>> metric.add(prediction='AQ', reference='AT')
        {'seq_recovery_rate': 0.75}
    """
    def _info(self):
        return evaluate.MetricInfo(
            description=utils.get_docstring_description(SeqRecoveryRate),
            citation="",
            features=datasets.Features({
                'predictions': datasets.Value('string'),
                'references': datasets.Value('string'),
            })
        )
    
    def _compute_aar(self, prediction: str, reference: str) -> float:
        """Compute the average sequence recovery rate of a single prediction."""
        assert len(prediction) == len(reference)
        recovery_sum = sum([1 for p, r in zip(prediction, reference) if p == r])
        return recovery_sum / len(prediction)
    
    def _compute(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        aar_avg = sum([self._compute_aar(p, r) for p, r in zip(predictions, references)]) / len(predictions)
        return {
            'aar': aar_avg
        }
