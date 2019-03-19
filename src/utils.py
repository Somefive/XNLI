from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

def get_bleu(preds, labels, length):
    _preds, _labels, _length = preds.detach().numpy().argmax(axis=-1), labels.detach().numpy(), length.detach().numpy()
    refs, hypos = [], []
    for _p, _l, l in zip(_preds, _labels, _length):
        hypos.append(_p[:l])
        refs.append([_l[:l]])
    return corpus_bleu(refs, hypos, smoothing_function=SmoothingFunction().method1)