import numpy as np
import pandas as pd
from sklearn import metrics


class DummyModel:
    def __init__(self):
        self.y = None

    def fit(self, X, y, **kwargs):
        self.y = np.mean(y)

    def predict(self, X, **kwargs):
        answer = np.ones(X.shape[0])
        answer[:] = self.y
        return answer


def test_validation():
    from src.utils import leave_one_out_validation

    N = 500
    SIZE = 10

    saved_results = {}
    for i in range(N):
        data = np.random.uniform()
        df = pd.DataFrame(
            {
                "Well": np.random.uniform(SIZE),
                "X": np.random.uniform(SIZE),
                "Y": np.random.uniform(SIZE),
                "NTG": [data for _ in range(SIZE)],
            }
        )
        model = DummyModel()
        res = leave_one_out_validation(
            saved_results,
            df,
            model,
            lambda x, y: np.sqrt(metrics.mean_squared_error(x, y)),
            save_res=True,
        )
        assert np.isclose(0.0, res)
        assert np.isclose(saved_results["common"][1], 0.0)


def test_models():
    from src.models import KNN_pipeline, linear_pipeline, random_forest_pipeline, catboost_pipeline

    N = 500
    SIZE = 100

    for i in range(N):
        column_1 = np.random.uniform(size=SIZE)
        column_2 = np.random.uniform(size=SIZE)
        column_3 = np.random.uniform(size=SIZE)
        trn_df = pd.DataFrame(
            {
                "Well": column_1[: SIZE // 2],
                "X": column_2[: SIZE // 2],
                "Y": column_3[: SIZE // 2],
                "NTG": column_1[: SIZE // 2] + column_2[: SIZE // 2] + column_3[: SIZE // 2],
            }
        )
        tst_df = pd.DataFrame(
            {
                "Well": column_1[SIZE // 2 :],
                "X": column_2[SIZE // 2 :],
                "Y": column_3[SIZE // 2 :],
                "NTG": column_1[SIZE // 2 :] + column_2[SIZE // 2 :] + column_3[SIZE // 2 :],
            }
        )
        preds, _ = linear_pipeline(trn_df, tst_df, target="NTG", use="linear")
        assert np.isclose(np.sqrt(metrics.mean_squared_error(preds, tst_df["NTG"])), 0.0)

    for i in range(N):
        column_1 = np.random.uniform(size=SIZE)
        column_2 = [x for x in range(SIZE)]
        column_3 = [x for x in range(SIZE)]
        trn_df = pd.DataFrame({"Well": column_1, "X": column_2, "Y": column_3, "NTG": column_1})
        preds, _ = KNN_pipeline(trn_df, trn_df, target="NTG", k=1, metric="euclidean")
        assert np.isclose(np.sqrt(metrics.mean_squared_error(preds, trn_df["NTG"])), 0.0)
        preds, _ = random_forest_pipeline(trn_df, trn_df, target="NTG")
        assert np.isclose(
            np.sqrt(metrics.mean_squared_error(preds, trn_df["NTG"])), 0.0, rtol=1e-1, atol=1e-1
        )
        preds, _ = catboost_pipeline(trn_df, trn_df, target="NTG")
        assert np.isclose(
            np.sqrt(metrics.mean_squared_error(preds, trn_df["NTG"])), 0.0, rtol=1e-1, atol=1e-1
        )
