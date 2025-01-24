# coding=utf-8

"""
Seq Recovery Rate package test
"""


def test_seq_recovery_rate():
    from .aar import SeqRecoveryRate
    metric = SeqRecoveryRate()
    metric.add(
        prediction='QVQ',
        reference='QVQ'
    )
    r = metric.compute()
    assert r['aar'] == 1.0

    metric.add(
        prediction='QVQ',
        reference='QVW'
    )
    r = metric.compute()
    assert r['aar'] == 2/3

    metric.add(
        prediction='QVQ',
        reference='VWT'
    )
    r = metric.compute()
    assert r['aar'] == 0
