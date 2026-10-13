import pandas as pd
from sklearn.metrics import precision_recall_fscore_support


def per_class_report(y_true, y_pred, categories):
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, zero_division=0)
    rep = pd.DataFrame({"precision": p, "recall": r, "f1": f, "support": s},
                       index=categories)
    rep.loc["macro avg"] = [p.mean(), r.mean(), f.mean(), s.sum()]
    return rep.round(3)