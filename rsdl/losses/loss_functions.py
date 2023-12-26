from rsdl import Tensor

def MeanSquaredError(preds: Tensor, actual: Tensor):
    # TODO : implement mean squared error
    err = actual - preds
    return (err ** 2).sum() / len(actual.data)

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



