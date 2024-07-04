# noinspection PyPep8Naming
def score_iso(clf, X):
    return clf.score_samples(X)


# noinspection PyPep8Naming
def score_lof(clf, X):
    if clf.novelty:
        result = clf.score_samples(X)
    else:
        result = clf.negative_outlier_factor_
    assert len(result) == len(X), (f"Size of result {len(result)} does not match size of points to score {len(X)}.\n"
                                   f" When running without novelty detection only the training dataset can be scored.\n"
                                   f"Novelty is currently set to {clf.novelty}")
    return result


# noinspection PyPep8Naming
def score_oc_svm(clf, X):
    return clf.score_samples(X)


# noinspection PyPep8Naming
def score_pid(clf, X):
    indices, outliers, scores_, pst, scores = clf.predict(
        X, err=0.1, pct=50)
    return scores


# noinspection PyPep8Naming
def score_eif(clf, X):
    return -clf.compute_paths(X)


# noinspection PyPep8Naming
def score_custom1(clf, X):
    return clf.score_samples(X)
