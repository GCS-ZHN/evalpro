# coding=utf-8
"""
DockQ module
"""

import evaluate
import datasets
import numpy as np

from typing import List, Dict, Any
from evalpro import utils
from Bio.PDB.Model import Model
from DockQ.DockQ import run_on_all_native_interfaces, load_PDB
from functools import lru_cache

load_PDB = lru_cache(maxsize=None)(load_PDB)


class DockQ(evaluate.EvaluationModule):
    '''
    Compute the DockQ score for protein-protein docking predictions. The DockQ
    score is a measure of the quality of the docking prediction, with a higher
    score indicating a better prediction.

    *   Docking scoring for biomolecular models                    *
    *   DockQ score legend:                                        *
    *    0.00 <= DockQ <  0.23 - Incorrect                         *
    *    0.23 <= DockQ <  0.49 - Acceptable quality                *
    *    0.49 <= DockQ <  0.80 - Medium quality                    *
    *            DockQ >= 0.80 - High quality                      *
    *   Ref: Mirabello and Wallner, 'DockQ v2: Improved automatic  *
    *   quality measure for protein multimers, nucleic acids       *
    *   and small molecules'                                       *
    *                                                              *
    *   For comments, please email: bjorn.wallner@.liu.se          *

    Sample Args:
        predictions (str): The predicted PDB file path.
        references (str): The reference PDB file path.

    Returns:
        dockq_avg (float): The average DockQ score of the predictions.
        dockq_max (float): The maximum DockQ score of the predictions.
        dockq_min (float): The minimum DockQ score of the predictions.
        dockq_detail (List[Dict]): A list of dictionaries containing the DockQ
    '''
 
    def _info(self) -> evaluate.MetricInfo:
        return evaluate.MetricInfo(
            description=utils.get_docstring_description(DockQ),
            citation="",
            features=datasets.Features({
                'predictions': datasets.Value(dtype='string', id='predictions'),
                'references': datasets.Value(dtype='string', id='references')
            }),
            homepage="https://github.com/bjornwallner/DockQ"
        )
    
    def _calc_dockq(self, prediction: str, reference: str) -> Dict[str, Any]:
        """
        Calculate DockQ score for a single prediction-reference pair.
        """
        # Run DockQ on the prediction and reference models
        prediction: Model = load_PDB(prediction)
        reference: Model = load_PDB(reference)
        reference_chain_ids = list(map(lambda x: x.get_id(), reference.get_chains()))
        prediction_chain_ids = list(map(lambda x: x.get_id(), prediction.get_chains()))
        assert len(reference_chain_ids) == len(prediction_chain_ids), "Number of chains in prediction and reference must match."
        chain_map = dict(zip(reference_chain_ids, prediction_chain_ids))
        dockq_detail, dockq_sum = run_on_all_native_interfaces(prediction, reference, chain_map=chain_map)
        return {
            'dockq_avg': dockq_sum / len(dockq_detail),
            'dockq_max': np.max([dockq['DockQ'] for dockq in dockq_detail.values()]),
            'dockq_min': np.min([dockq['DockQ'] for dockq in dockq_detail.values()]),
            'dockq_detail': dockq_detail
        }

    def _compute(self, predictions: List[str], references: List[str]) -> Dict[str, Any]:
        
        """
        Compute DockQ score for a list of prediction-reference pairs.
        """
        results = []
        for prediction, reference in zip(predictions, references):
            result = self._calc_dockq(prediction, reference)
            results.append(result)
        
        # Aggregate results
        avg_dockq = np.mean([result['dockq_avg'] for result in results])
        max_dockq = np.max([result['dockq_max'] for result in results])
        min_dockq = np.min([result['dockq_min'] for result in results])
        
        return {
            'dockq_avg': avg_dockq,
            'dockq_max': max_dockq,
            'dockq_min': min_dockq,
            'dockq_detail': [result['dockq_detail'] for result in results]
        }
