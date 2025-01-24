# coding=utf-8

def test_aligned_tmscore():
    import numpy as np
    from .tmscore import AlignedTMscore
    metric = AlignedTMscore()
    metric.add(
        prediction={
            'coords': np.array([
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0]], dtype=np.float32), 
            'seq': 'ATA'}, 
        reference={
            'coords': np.array([
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0]], dtype=np.float32),
            'seq': 'ATA'}
    )
    r = metric.compute()
    assert 0 <= r['rmsd'] < 1e-6
    assert r['tmscore'] == 1.0
