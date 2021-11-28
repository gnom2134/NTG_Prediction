import numpy as np
import pandas as pd
import warnings
from sklearn import metrics


from utils import NearestNeighborsFeats, plot_stuff, make_submission, leave_one_out_validation
from models import KNN_pipeline, linear_pipeline, random_forest_pipeline, catboost_pipeline, smart_ensemble


def find_best(preds, rmse_score):
    global best_pred, best_rmse

    if rmse_score < best_rmse:
        best_pred = preds
        best_rmse = rmse_score


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    trn_df = pd.read_csv("../data/Training_wells.csv")
    tst_df = pd.read_csv("../data/Empty_part.csv")

    k_list = [2, 5, 9]
    new_tst_df = tst_df.copy()
    new_trn_df = trn_df.copy()

    print("KNN features extracting...")
    for metric in ["minkowski", "cosine", "manhattan", "euclidean"]:
        NNF = NearestNeighborsFeats(n_jobs=1, k_list=k_list, metric=metric)
        NNF.fit(trn_df[trn_df.columns.drop(["NTG", "Well"])].values, trn_df["NTG"].values)

        test_knn_feats = NNF.predict(tst_df[tst_df.columns.drop(["Well"])].values)
        test_knn_feats_df = pd.DataFrame(
            test_knn_feats, columns=[metric + "_feature" + str(x) for x in range(test_knn_feats.shape[1])]
        )
        new_tst_df = pd.concat([new_tst_df, test_knn_feats_df], axis=1)

        train_knn_feats = NNF.predict(trn_df[trn_df.columns.drop(["Well", "NTG"])].values, train=True)
        train_knn_feats_df = pd.DataFrame(
            train_knn_feats, columns=[metric + "_feature" + str(x) for x in range(test_knn_feats.shape[1])]
        )
        new_trn_df = pd.concat([new_trn_df, train_knn_feats_df], axis=1)

    new_tst_df = new_tst_df.sort_values("Well")

    size = (tst_df["Y"].max() - tst_df["Y"].min() + 1, tst_df["X"].max() - tst_df["X"].min() + 1)

    best_pred = None
    best_rmse = 1e3

    predictions = []
    saved_results = {}

    print("Linear models...")
    for i in ["lasso", "ridge"]:
        pred, model = linear_pipeline(new_trn_df[new_trn_df.columns.drop(["Well"])], new_tst_df, use=i)
        val_rmse = leave_one_out_validation(
            saved_results, trn_df, model, lambda x, y: np.sqrt(metrics.mean_squared_error(x, y))
        )
        find_best(pred, val_rmse)
        plot_stuff(trn_df, tst_df, pred, size)
        predictions.append((pred, val_rmse, i))
        make_submission(tst_df, pred, i)

    print("KNN models...")
    for m in ["minkowski", "euclidean"]:
        for k in [3, 5]:
            pred, model = KNN_pipeline(trn_df[trn_df.columns.drop(["Well"])], tst_df, k=k, metric=m)
            val_rmse = leave_one_out_validation(
                saved_results, trn_df, model, lambda x, y: np.sqrt(metrics.mean_squared_error(x, y))
            )
            plot_stuff(trn_df, tst_df, pred, size)
            find_best(pred, val_rmse)
            predictions.append((pred, val_rmse, "knn"))
            make_submission(tst_df, pred, "knn_" + str(k))

    print("Random forest model...")
    pred, model = random_forest_pipeline(new_trn_df[new_trn_df.columns.drop(["Well"])], new_tst_df)
    val_rmse = leave_one_out_validation(
        saved_results, trn_df, model, lambda x, y: np.sqrt(metrics.mean_squared_error(x, y))
    )
    plot_stuff(trn_df, tst_df, pred, size)
    find_best(pred, val_rmse)
    predictions.append((pred, val_rmse, "random_forest"))
    make_submission(tst_df, pred, "rand_f")

    print("Catboost model...")
    pred, model = catboost_pipeline(new_trn_df[new_trn_df.columns.drop(["Well"])], new_tst_df, silent=True)
    val_rmse = leave_one_out_validation(
        saved_results, trn_df, model, lambda x, y: np.sqrt(metrics.mean_squared_error(x, y)), silent=True
    )
    plot_stuff(trn_df, tst_df, pred, size)
    find_best(pred, val_rmse)
    predictions.append((pred, val_rmse, "catboost"))
    make_submission(tst_df, pred, "catboost")

    predictions_map = {}
    for i in predictions:
        if i[2] not in predictions_map:
            predictions_map[i[2]] = (None, 0)
        if i[1] > predictions_map[i[2]][1]:
            predictions_map[i[2]] = (i[0], i[1])
    predictions = [(x[1][0], x[1][1]) for x in predictions_map.items()]

    predictions = sorted(predictions, key=lambda x: x[1])

    pred = predictions[0][0].copy()
    n = 3
    for i in range(1, n):
        pred += predictions[i][0]
    pred /= n

    plot_stuff(trn_df, tst_df, pred, size)
    find_best(pred, 0.1)
    make_submission(tst_df, pred, "ensemble")

    pred = predictions[0][0].copy()
    n = 3
    for i in [1, 3]:
        pred += predictions[i][0]
    pred /= n

    plot_stuff(trn_df, tst_df, pred, size)
    find_best(pred, 0.0)
    make_submission(tst_df, pred, "ensemble_2")

