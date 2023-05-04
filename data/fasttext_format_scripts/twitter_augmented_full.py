import pandas as pd
import FastTextSupervisedClassifier
from sklearn.model_selection import train_test_split
from utils import preprocess_none
from utils import preprocess_simple
from utils import preprocess_full
from utils import preprocess_RNN


if __name__ == '__main__':

    train_df = pd.read_csv('../twitter/augmented/full.csv', header=0)

    # preprocessing
    train_df["tweet_none"] = train_df["tweet"].copy().apply(lambda x: preprocess_none(str(x)))
    train_df["tweet_simple"] = train_df["tweet"].copy().apply(lambda x: preprocess_simple(str(x)))
    train_df["tweet_full"] = train_df["tweet"].copy().apply(lambda x: preprocess_full(str(x)))
    train_df["tweet_rnn"] = train_df["tweet"].copy().apply(lambda x: preprocess_RNN(str(x)))
    train_df["tweet_rnn_enriched"] = train_df["tweet"].copy().apply(lambda x: preprocess_RNN(str(x)))

    options = ["none", "simple", "full", "rnn", "rnn_enriched"]



    for o in options:
        X = train_df[f"tweet_{o}"]
        y = train_df["class"]

        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=0.1, random_state=42)

        train = pd.DataFrame()
        train["tweet"] = X_train
        train["class"] = y_train

        valid_df = pd.DataFrame()
        valid_df["tweet"] = X_valid
        valid_df["class"] = y_valid

        FastTextSupervisedClassifier.df_to_FT_labels(train, f"../twitter/augmented/full/{o}/FT_train.txt", "class",
                                                     "tweet")
        FastTextSupervisedClassifier.df_to_FT_labels(valid_df, f"../twitter/augmented/full/{o}/FT_valid.txt", "class",
                                                     "tweet")

        if o != "rnn_enriched":
            FastTextSupervisedClassifier.df_to_RNNLM_labels(train, f"../twitter/augmented/full/{o}/RNN_train.txt",
                                                            "class", "tweet", False)
        else:
            FastTextSupervisedClassifier.df_to_RNNLM_labels(train, f"../twitter/augmented/full/{o}/RNN_train.txt",
                                                            "class", "tweet", False, True)

        FastTextSupervisedClassifier.df_to_RNNLM_labels(valid_df, f"../twitter/augmented/full/{o}/RNN_valid.txt",
                                                        "class", "tweet", True)

        train.to_pickle(f"../twitter/augmented/full/{o}/train.p")
        valid_df.to_pickle(f"../twitter/augmented/full/{o}/valid.p")



