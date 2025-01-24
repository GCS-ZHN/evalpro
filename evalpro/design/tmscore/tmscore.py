
# coding=utf-8
"""
Aligned TM-score module
"""

import tmtools
import evaluate
import datasets

from typing import List, Dict, Any
from evalpro import utils


class AlignedTMscore(evaluate.EvaluationModule):
    """
    This new metric is designed to calculate the average aligned TM-score
    of a set of predicted structures.

    Sample Args:
        predictions: list of predicted structures to score. Each prediction
            should be a dictionary with 'coords' and 'seq' fields. 'coords'
            is a 2D array with shape (N, 3) and 'seq' is a string with the
            same length as the coordinates.
            CA atom coordinates is recommended as it is the offical defoult
            in TMsore cli.
        references: list of reference structures for each prediction. 
            the same format as 'predictions'.

    Returns:
        rmsd_avg: the average RMSD of the predictions.
        tm_score_avg: the average TM-score of the predictions.

    Examples:
        >>> metric = AlignedTMscore()
        >>> metric.add(prediction={
                        'coords': torch.tensor([
                            [0, 1, 0],
                            [1, 1, 1],
                            [0, 1, 0]], dtype=torch.float32), 
                        'seq': 'ATA'}, 
                       reference={
                        'coords': torch.tensor([
                            [0, 1, 0],
                            [1, 1, 1],
                            [0, 1, 0]], dtype=torch.float32), 
                        'seq': 'ATA'})
        >>> metric.compute()
        {'rmsd_avg': 1.7206378853011898e-08, 'tm_score_avg': 1.0}
    """
    def _info(self) -> evaluate.MetricInfo:
        return evaluate.MetricInfo(
            description=utils.get_docstring_description(AlignedTMscore),
            citation="Zhang and Skolnick, 2004, Proteins, 57:702-710",
            features=datasets.Features({
                
                'predictions': {
                    'coords': datasets.Array2D(shape=(None, 3), dtype='float32'),
                    'seq': datasets.Value(dtype='string')
                },
                'references': {
                    'coords': datasets.Array2D(shape=(None, 3), dtype='float32'),
                    'seq': datasets.Value(dtype='string')
                }
            }),
            homepage="https://zhanglab.ccmb.med.umich.edu/TM-align/"
        )

    def _compute(self, predictions: List[Dict[str, Any]], references: List[Dict[str, Any]]) -> Dict[str, float]:
        rmsd, tm_score = 0, 0
        for pred, ref in zip(predictions, references):
            pred_coords = pred['coords']
            ref_coords = ref['coords']
            pred_seq = pred['seq']
            ref_seq = ref['seq']
            assert len(pred_coords) == len(pred_seq)
            assert len(ref_coords) == len(ref_seq)
            assert len(pred_coords) >= 3 and len(ref_coords) >= 3, "Sequence length should be at least 3"
            res = tmtools.tm_align(pred_coords, ref_coords, pred_seq, ref_seq)
            rmsd += res.rmsd
            tm_score += res.tm_norm_chain1
        return {
            'rmsd': rmsd / len(predictions),
            'tmscore': tm_score / len(predictions)
        }