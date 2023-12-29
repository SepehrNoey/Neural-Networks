from rsdl import Tensor
import numpy as np
from rsdl.activations import Softmax
from rsdl.tensors import _tensor_neg

def MeanSquaredError(preds: Tensor, actual: Tensor) -> Tensor:
    # TODO : implement mean squared error
    err = actual - preds
    return (err ** 2).sum() * (Tensor(1 / len(actual.data), requires_grad=actual.requires_grad, depends_on=actual.depends_on))

def CategoricalCrossEntropy(preds: Tensor, actual: Tensor) -> Tensor:
    # TODO : imlement categorical cross entropy 
    preds = Softmax(preds)
    log_softmaxed_preds = preds.log()
    cce_vector = actual * log_softmaxed_preds
    cce = _tensor_neg(cce_vector.sum())
    return cce



