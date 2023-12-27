from rsdl import Tensor
import numpy as np

def MeanSquaredError(preds: Tensor, actual: Tensor):
    # TODO : implement mean squared error
    err = actual - preds
    return (err ** 2).sum() * (Tensor(1 / len(actual.data), requires_grad=actual.requires_grad, depends_on=actual.depends_on))

def CategoricalCrossEntropy(preds: Tensor, actual: Tensor):
    # TODO : imlement categorical cross entropy 
    exps = preds.exp()
    sum_exps = exps.sum()
    softmaxed_preds = sum_exps * Tensor(data=(1 / sum_exps.data),
        requires_grad=sum_exps.requires_grad, depends_on=sum_exps.depends_on)
    log_softmaxed_preds = softmaxed_preds.log()
    cce_vector = - (actual * log_softmaxed_preds)
    cce = - (cce_vector.sum())
    return cce



