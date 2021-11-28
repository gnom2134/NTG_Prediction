import numpy as np
import pandas as pd
from sklearn import metrics
from src.models import KNN_pipeline, linear_pipeline, random_forest_pipeline, catboost_pipeline
from src.utils import leave_one_out_validation


class DummyModel:
    def __init__(self):
        self._y = None

    def fit(self, _x, _y):
        self._y = np.mean(_y)

    def predict(self, _x):
        answer = np.ones(_x.shape[0])
        answer[:] = self._y
        return answer


def test_validation():
    p_n = 500
    p_size = 10

    saved_results = {}
    for _ in range(p_n):
        data = np.random.uniform()
        df = pd.DataFrame(
            {
                "Well": np.random.uniform(p_size),
                "X": np.random.uniform(p_size),
                "Y": np.random.uniform(p_size),
                "NTG": [data for _ in range(p_size)],
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
    p_n = 500
    p_size = 100

    for _ in range(p_n):
        column_1 = np.random.uniform(size=p_size)
        column_2 = np.random.uniform(size=p_size)
        column_3 = np.random.uniform(size=p_size)
        trn_df = pd.DataFrame(
            {
                "Well": column_1[: p_size // 2],
                "X": column_2[: p_size // 2],
                "Y": column_3[: p_size // 2],
                "NTG": column_1[: p_size // 2] + column_2[: p_size // 2] + column_3[: p_size // 2],
            }
        )
        tst_df = pd.DataFrame(
            {
                "Well": column_1[p_size // 2 :],
                "X": column_2[p_size // 2 :],
                "Y": column_3[p_size // 2 :],
                "NTG": column_1[p_size // 2 :] + column_2[p_size // 2 :] + column_3[p_size // 2 :],
            }
        )
        preds, _ = linear_pipeline(trn_df, tst_df, target="NTG", use="linear")
        assert np.isclose(np.sqrt(metrics.mean_squared_error(preds, tst_df["NTG"])), 0.0)

    for _ in range(p_n):
        column_1 = np.random.uniform(size=p_size)
        column_2 = [x for x in range(p_size)]
        column_3 = [x for x in range(p_size)]
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
