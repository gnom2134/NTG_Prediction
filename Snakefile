rule all:
    input:
        "subs/submission_ridge.csv",
        "subs/submission_lasso.csv",
        "subs/submission_knn_3.csv",
        "subs/submission_knn_5.csv",
        "subs/submission_rand_f.csv",
        "subs/submission_catboost.csv",
        "subs/submission_ensemble.csv",
        "subs/submission_ensemble_2.csv"

rule main:
    input:
        "data/train_df.csv",
        "data/Empty_part.csv",
        "data/preprocessed_train.csv",
        "data/preprocessed_test.csv"
    output:
        "subs/submission_ridge.csv",
        "subs/submission_lasso.csv",
        "subs/submission_knn_3.csv",
        "subs/submission_knn_5.csv",
        "subs/submission_rand_f.csv",
        "subs/submission_catboost.csv",
        "subs/submission_ensemble.csv",
        "subs/submission_ensemble_2.csv"
    shell:
        "python ./src/main.py --train {input[0]} --test {input[1]} --preprocessed_train {input[2]} --preprocessed_test {input[3]}"

rule preprocessing:
    input:
        "data/train_df.csv",
        "data/Empty_part.csv"
    output:
        "data/preprocessed_train.csv",
        "data/preprocessed_test.csv"
    shell:
        "python ./src/preprocessing.py --train {input[0]} --test {input[1]}"