from sklearn import (
    linear_model,
    pipeline,
    preprocessing,
    impute,
    feature_selection,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
import numpy as np
import pandas as pd


def linear_pipeline(trn_df, tst_df, target="NTG", use="lasso"):
    if use == "lasso":
        l = linear_model.LassoCV()
    elif use == "ridge":
        l = linear_model.RidgeCV(alphas=np.linspace(start=0.1, stop=10, num=1000), normalize=True)
    elif use == "elasticnet":
        l = linear_model.ElasticNetCV()
    elif use == "linear":
        l = linear_model.LinearRegression()
    else:
        raise RuntimeError("Wrong model name")

    model = pipeline.Pipeline(
        [
            ("fill_nan", impute.SimpleImputer(missing_values=np.nan, strategy="mean")),
            ("scaler", preprocessing.StandardScaler()),
            ("select features", feature_selection.VarianceThreshold()),
            ("linear_model", l),
        ]
    )

    model.fit(trn_df[trn_df.columns.drop(target)], trn_df[target])
    return model.predict(tst_df[trn_df.columns.drop(target)]), model


def KNN_pipeline(trn_df, tst_df, target='NTG', k=2, metric='minkowski'):
    model = KNeighborsRegressor(k, metric=metric)
    model.fit(trn_df[trn_df.columns.drop(target)], trn_df[target])
    return model.predict(tst_df[trn_df.columns.drop(target)]), model


def random_forest_pipeline(trn_df, tst_df, target='NTG', **kwargs):
    n_estimators = [int(x) for x in np.linspace(start=5, stop=20, num=2)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(5, 20, num=2)]
    max_depth.append(None)
    min_samples_split = [2, 3, 4, 5, 8, 10]
    min_samples_leaf = [1, 2, 3, 4, 5]
    bootstrap = [True, False]
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    rf = RandomForestRegressor(random_state=42)
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=5, verbose=2,
                                   n_jobs=-1, random_state=42)
    rf_random.fit(trn_df[trn_df.columns.drop(target)], trn_df[target])

    model = RandomForestRegressor(**rf_random.best_params_, random_state=42)
    model.fit(trn_df[trn_df.columns.drop(target)], trn_df[target])
    return model.predict(tst_df[trn_df.columns.drop(target)]), model


def catboost_pipeline(trn_df, tst_df, target='NTG', **kwargs):
    model = CatBoostRegressor(iterations=200,
                              learning_rate=0.1,
                              depth=4,
                              l2_leaf_reg=1,
                              min_data_in_leaf=3,
                              rsm=0.5,
                              early_stopping_rounds=5,
                              random_state=42
                              )

    model.fit(trn_df[trn_df.columns.drop(target)], trn_df[target], **kwargs)

    return model.predict(tst_df[trn_df.columns.drop(target)]), model


def smart_ensemble(trn_df, tst_df, size, *preds):
    meta_trn_df = pd.DataFrame(trn_df[['NTG', 'X', 'Y']])
    meta_tst_df = pd.DataFrame(tst_df[['X', 'Y']])

    max_x = int(meta_tst_df.X.max())
    max_y = int(meta_tst_df.Y.max())
    min_x = int(meta_tst_df.X.min())
    min_y = int(meta_tst_df.Y.min())

    cnt = 0

    for p in preds:
        a = np.zeros(size)
        for i, x in enumerate(tst_df.values):
            a[x[2] - tst_df.Y.min(), x[1] - tst_df.X.min()] = p[i]

        for i, x in enumerate(trn_df.values):
            a[x[2] - trn_df.Y.min(), x[1] - trn_df.X.min()] = x[3]

        for add_x, add_y in [(x, y) for x in [-2, -1, 0, 1, 2] for y in [-2, -1, 0, 1, 2]]:
            meta_trn_df['smart_feature_' + str(cnt)] = 0
            for index, row in meta_trn_df.iterrows():
                new_x = int(row['X'] + add_x)
                new_y = int(row['Y'] + add_y)

                if max_x < new_x or min_x > new_x or max_y < new_y or min_y > new_y:
                    meta_trn_df.loc[index, 'smart_feature_' + str(cnt)] = a[
                        int(row['Y']) - min_y, int(row['X']) - min_x]
                else:
                    meta_trn_df.loc[index, 'smart_feature_' + str(cnt)] = a[new_y - min_y, new_x - min_x]

            meta_tst_df['smart_feature_' + str(cnt)] = 0
            for index, row in meta_tst_df.iterrows():
                new_x = int(row['X'] + add_x)
                new_y = int(row['Y'] + add_y)

                if max_x < new_x or min_x > new_x or max_y < new_y or min_y > new_y:
                    meta_tst_df.loc[index, 'smart_feature_' + str(cnt)] = a[
                        int(row['Y']) - min_y, int(row['X']) - min_x]
                else:
                    meta_tst_df.loc[index, 'smart_feature_' + str(cnt)] = a[new_y - min_y, new_x - min_x]
            cnt += 1

    model = CatBoostRegressor(iterations=200,
                              learning_rate=0.1,
                              depth=3,
                              rsm=0.5,
                              l2_leaf_reg=1,
                              min_data_in_leaf=3,
                              early_stopping_rounds=5,
                              random_state=42
                              )

    model.fit(meta_trn_df[meta_trn_df.columns.drop('NTG')], meta_trn_df['NTG'], silent=True)

    return model.predict(meta_tst_df[meta_trn_df.columns.drop('NTG')]), model
