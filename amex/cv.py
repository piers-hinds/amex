import numpy as np


def cv(model, X, y, splitter, metric):
    scores = []
    for train_inds, val_inds in splitter.split(X):
        model.fit(X[train_inds], y[train_inds])
        preds = model.eval(X[val_inds])
        score = metric(y[val_inds], preds)
        scores.append(score)
    scores = np.array(scores)
    return scores.mean(), scores
