from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import NearestNeighbors
from multiprocessing import Pool
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


class NearestNeighborsFeats(BaseEstimator, ClassifierMixin):
    """
    This class should implement KNN features extraction
    """
    def __init__(self, n_jobs, k_list, metric, n_classes=None, n_neighbors=None, eps=1e-6):
        self.n_jobs = n_jobs
        self.k_list = k_list
        self.metric = metric

        if n_neighbors is None:
            self.n_neighbors = max(k_list)
        else:
            self.n_neighbors = n_neighbors

        self.eps = eps
        self.n_classes_ = n_classes
        self.NN = None
        self.y_train = None
        self.X = None
        self.n_classes = None

    def fit(self, X, y):
        """
        Set's up the train set and self.NN object
        """
        self.NN = NearestNeighbors(
            n_neighbors=max(self.k_list),
            metric=self.metric,
            n_jobs=1,
            algorithm="brute" if self.metric == "cosine" else "auto",
        )
        self.NN.fit(X)

        self.y_train = y
        self.X = X

        self.n_classes = np.unique(y).shape[0] if self.n_classes_ is None else self.n_classes_

    def predict(self, X, train=False):
        """
        Produces KNN features for every object of a dataset X
        """
        if self.n_jobs == 1:
            test_feats = []
            for i in range(X.shape[0]):
                test_feats.append(self.get_features_for_one(X[i : i + 1], train=train))
        else:
            with Pool(self.n_jobs) as p:
                test_feats = p.map(self.get_features_for_one, list(X))

        return np.vstack(test_feats)

    def get_features_for_one(self, x, train=False):
        """
        Computes KNN features for a single object `x`
        """
        train = int(train)

        NN_output = self.NN.kneighbors(x)
        neighs = NN_output[1][0]

        neighs_dist = NN_output[0][0]

        neighs_y = self.y_train[neighs]
        return_list = []

        return_list += [[neighs_y[train], neighs_dist[train]]]

        for k in self.k_list:
            feature_list = []
            feature_list += [neighs_dist[k - 1]]
            feature_list += [neighs_dist[k - 1] / (neighs_dist[0] + self.eps)]
            feature_list += [np.mean(neighs_dist[train:k])]
            feature_list += [np.mean(neighs_y[train:k])]
            feature_list += [neighs_y[k - 1]]
            feature_list += [np.log(neighs_dist[k - 1]) * neighs_y[k - 1]]
            feature_list += [neighs_dist[k - 1] * neighs_y[k - 1]]
            feature_list += [neighs_y[k - 1] / neighs_dist[k - 1]]
            feature_list += list(
                np.mean(np.array(self.X[neighs[train:k]] - np.array(x[0])) * np.array([neighs_y[train:k]]).T, axis=0)
            )
            feature_list += list(
                np.mean(
                    np.array(self.X[neighs[train:k]] - np.array(x[0]) + self.eps) / np.array([neighs_y[train:k]]).T,
                    axis=0,
                )
            )
            feature_list += list(np.mean(np.array(self.X[neighs[train:k]]) * np.array([neighs_y[train:k]]).T, axis=0))
            feature_list += list(np.mean(np.array(self.X[neighs[train:k]]) / np.array([neighs_y[train:k]]).T, axis=0))

            return_list += [feature_list]

        knn_feats = np.hstack(return_list)

        return knn_feats


def make_submission(tst_df, preds, name):
    tst_df["NTG"] = preds
    tst_df.to_csv("submission_" + name + ".csv", index=False)


def leave_one_out_validation(saved_results, data, model, m, description="common", save_res=False, **kwargs):
    k_list = [2, 5, 9]
    result = []

    for index, row in tqdm(data.iterrows()):
        new_data = data.copy()
        new_data.drop(columns=["Well"], inplace=True)
        new_test = new_data.loc[[index], :].copy()
        new_data = new_data.drop(index=index)
        new_train_data = new_data.copy()
        new_test_data = new_test.copy()
        new_train_data.reset_index(drop=True, inplace=True)
        new_test_data.reset_index(drop=True, inplace=True)

        for metric in ["minkowski", "cosine", "manhattan", "euclidean"]:
            NNF = NearestNeighborsFeats(n_jobs=1, k_list=k_list, metric=metric)
            NNF.fit(new_data[new_data.columns.drop(["NTG"])].values, new_data["NTG"].values)

            test_knn_feats = NNF.predict(new_test[new_test.columns.drop(["NTG"])].values)
            test_knn_feats_df = pd.DataFrame(
                test_knn_feats, columns=[metric + "_feature" + str(x) for x in range(test_knn_feats.shape[1])]
            )
            new_test_data = pd.concat([new_test_data, test_knn_feats_df], axis=1)

            train_knn_feats = NNF.predict(new_data[new_data.columns.drop(["NTG"])].values, train=True)
            train_knn_feats_df = pd.DataFrame(
                train_knn_feats, columns=[metric + "_feature" + str(x) for x in range(test_knn_feats.shape[1])]
            )
            new_train_data = pd.concat([new_train_data, train_knn_feats_df], axis=1)

        model.fit(new_train_data[new_train_data.columns.drop(["NTG"])], new_train_data["NTG"], **kwargs)

        pred = model.predict(new_test_data[new_test_data.columns.drop(["NTG"])])
        result.append(pred[0])

    m_value = m(data.NTG.values, result)

    if save_res:
        if description not in saved_results:
            saved_results[description] = (None, 1e3)
        if m_value < saved_results[description][1]:
            saved_results[description] = (result, m_value)

    return m_value


def plot_stuff(trn_df, tst_df, preds, size):
    a = np.zeros(size)
    for i, x in enumerate(tst_df.values):
        a[x[2] - tst_df.Y.min(), x[1] - tst_df.X.min()] = preds[i]

    for i, x in enumerate(trn_df.values):
        a[x[2] - trn_df.Y.min(), x[1] - trn_df.X.min()] = x[3]

    plt.figure(figsize=(9, 9))
    plt.imshow(a)
    plt.show()
