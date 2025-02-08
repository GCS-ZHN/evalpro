# EVALPRO

EvalPro is a Python library for evaluating the performance of machine learning
protein models. It is designed to be easy to use and add metrics are implemented
based on HuggingFace's `evaluate` library.

## Design Principles

Based on Hugging Face's open-source `evaluate` and `datasets` packages, we build protein AI algorithm evaluation metrics such as [tmscore.py](evalpro/design/tmscore/tmscore.py).

New metrics are created by inheriting the `evaluate.EvaluationModule` class and implementing the `_info` and `_compute` methods.

The `_info` method returns an `evaluate.MetricInfo` object that describes various information about the new metric, with a focus on field type descriptions for predicted and reference values.

The `_compute` method is a batch processing function for input predictions and references, which is called by the public `compute` method.

> Note that type specifications apply to individual values, while `_compute` parameters are batch lists.

## Dependency Issues

- Pyarrow Compatibility

The datasets package uses Apache Arrow (pyarrow) for data storage. Current datasets versions have limited compatibility with newer pyarrow releases, requiring careful version control.
