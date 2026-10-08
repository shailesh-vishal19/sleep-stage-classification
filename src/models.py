from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier


def get_models(seed=0):
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=seed),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            class_weight="balanced", random_state=seed),
    }
