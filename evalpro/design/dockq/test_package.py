
def test_dockq():
    import numpy as np
    from .dockq import DockQ
    metric = DockQ()
    metric.add(prediction='test_data/6jjp_def.pdb', reference='test_data/6jjp_abc.pdb')
    result = metric.compute()
    assert np.isclose(result['dockq_avg'], 0.8417666354087948)
    assert np.isclose(result['dockq_max'], 0.8652124671491367)
    assert np.isclose(result['dockq_min'], 0.8280209264982691)
