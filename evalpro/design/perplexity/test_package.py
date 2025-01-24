# coding=utf-8

def test_perplexity():
    import numpy as np
    from .perplexity import Perplexity
    metric = Perplexity(vocab_size=10)
    metric.add(
        prediction=np.log(np.diag(np.ones(10)) + 1e-10),
        reference=list(range(10))
    )
    r = metric.compute()
    assert np.isclose(r['perplexity'], 1.0)

    # for logits with uniform distribution, 
    # the perplexity should be equal to the vocab_size
    for vocab_size in [1, 5, 10, 15, 30, 70, 100, 10243]:
        metric = Perplexity(vocab_size=vocab_size)
        metric.add(
            prediction=np.zeros((10, vocab_size)),
            reference=np.zeros(10)
        )
        r = metric.compute()
        assert np.isclose(r['perplexity'], vocab_size)
