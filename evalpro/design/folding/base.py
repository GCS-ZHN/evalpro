# coding=utf-8

"""
Abstract basic protein folding prediction
"""

from abc import ABC, abstractmethod
from pathlib import Path
from joblib import Parallel, delayed
from tqdm.auto import tqdm


class FoldingBase(ABC):

    @abstractmethod
    def predict(self, seq: str, pdb_path: Path, *args, **kwargs):
        raise NotImplementedError

    def predict_all(
            self, seqs: dict, pdb_path: Path, *args, 
            n_jobs: int = 1, progress_bar: bool = False,
            **kwargs):
        if n_jobs == 1:
            res = (self.predict(
                v,
                pdb_path / (k + '.pdb'),
                *args,
                **kwargs) for k,v in seqs.items())

        elif n_jobs > 1:
            parallel = Parallel(n_jobs=n_jobs, return_as='generator_unordered')
            res = parallel(delayed(self.predict)(
                v,
                pdb_path / (k + '.pdb'),
                *args,
                **kwargs) for k,v in seqs.items())
        else:
            raise ValueError('n_jobs must be >= 1')
        
        if progress_bar:
            res = tqdm(
                res,
                total=len(seqs),
                desc=self.__class__.__name__ + ': ')

        list(res)
