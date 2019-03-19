from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

def get_bleu(preds, labels, length):
    _preds, _labels, _length = preds.detach().numpy().argmax(axis=-1), labels.detach().numpy(), length.detach().numpy()
    refs, hypos = [], []
    for _p, _l, l in zip(_preds, _labels, _length):
        hypos.append(_p[:l])
        refs.append([_l[:l]])
    return corpus_bleu(refs, hypos, smoothing_function=SmoothingFunction().method1)

def extract_path(paths, prefix=None, suffix=None, groupby=1):
    if prefix is not None:
        paths = [prefix + path for path in paths]
    if suffix is not None:
        paths = [path + suffix for path in paths]
    if groupby > 1:
        assert len(paths) % groupby == 0, 'path number %d should be divided by group size %d' % (len(paths), groupby)
        paths = [paths[i:i+groupby] for i in range(0, len(paths), groupby)]
    return paths
            
