import pandas as pd
import click

from utils import NearestNeighborsFeats


@click.command()
@click.option("--train", help="Path to the train file")
@click.option("--test", help="Path to the test file")
def preprocess_data(train, test):
    trn_df = pd.read_csv(train)
    tst_df = pd.read_csv(test)

    k_list = [2, 5, 9]

    new_tst_df = tst_df.copy()
    new_trn_df = trn_df.copy()

    for metric in ["minkowski", "cosine", "manhattan", "euclidean"]:
        NNF = NearestNeighborsFeats(n_jobs=1, k_list=k_list, metric=metric)
        NNF.fit(trn_df[trn_df.columns.drop(["NTG", "Well"])].values, trn_df["NTG"].values)

        test_knn_feats = NNF.predict(tst_df[tst_df.columns.drop(["Well"])].values)
        test_knn_feats_df = pd.DataFrame(
            test_knn_feats,
            columns=[metric + "_feature" + str(x) for x in range(test_knn_feats.shape[1])],
        )
        new_tst_df = pd.concat([new_tst_df, test_knn_feats_df], axis=1)

        train_knn_feats = NNF.predict(
            trn_df[trn_df.columns.drop(["Well", "NTG"])].values, train=True
        )
        train_knn_feats_df = pd.DataFrame(
            train_knn_feats,
            columns=[metric + "_feature" + str(x) for x in range(test_knn_feats.shape[1])],
        )
        new_trn_df = pd.concat([new_trn_df, train_knn_feats_df], axis=1)

    new_tst_df = new_tst_df.sort_values("Well")

    new_tst_df.to_csv("./data/preprocessed_test.csv", index=False)
    new_trn_df.to_csv("./data/preprocessed_train.csv", index=False)


if __name__ == "__main__":
    preprocess_data()
