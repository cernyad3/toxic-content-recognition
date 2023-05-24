import pandas as pd
import data_utils
from sklearn.model_selection import train_test_split
from utils import preprocess_none
from utils import preprocess_simple
from utils import preprocess_full
from utils import preprocess_RNN


if __name__ == '__main__':

    aug_options = ["balanced", "full"]

    for ao in aug_options:
        train_df = pd.read_csv(f'../twitter/augmented/{ao}.csv', header=0)

        # preprocessing
        train_df["tweet_none"] = train_df["tweet"].copy().apply(lambda x: preprocess_none(str(x)))
        train_df["tweet_simple"] = train_df["tweet"].copy().apply(lambda x: preprocess_simple(str(x)))
        train_df["tweet_full"] = train_df["tweet"].copy().apply(lambda x: preprocess_full(str(x)))
        train_df["tweet_rnn"] = train_df["tweet"].copy().apply(lambda x: preprocess_RNN(str(x)))
        train_df["tweet_rnn_enriched"] = train_df["tweet"].copy().apply(lambda x: preprocess_RNN(str(x)))
        train_df["tweet_rnn_reverse"] = train_df["tweet"].copy().apply(lambda x: preprocess_RNN(str(x)))

        options = ["none", "simple", "full", "rnn", "rnn_enriched", "rnn_reverse"]

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

            prep = "rnn" if o.startswith("rnn") else o
            test_df = pd.read_pickle(f"../twitter/preprocessed/multi/{prep}/test.p")

            data_utils.df_to_FT_labels(train, f"../twitter/augmented/{ao}/{o}/FT_train.txt",
                                                         "class",
                                                         "tweet")
            data_utils.df_to_FT_labels(valid_df, f"../twitter/augmented/{ao}/{o}/FT_valid.txt",
                                                         "class",
                                                         "tweet")
            enrich = False
            reverse = False

            if o == "rnn_enriched":
                enrich = True

            if o == "rnn_reverse":
                reverse = True

            data_utils.df_to_RNNLM_labels(train, f"../twitter/augmented/{ao}/{o}/RNN_train.txt",
                                                            "class", "tweet", False, reverse=reverse, enrich=enrich)

            data_utils.df_to_RNNLM_labels(valid_df,
                                                            f"../twitter/augmented/{ao}/{o}/RNN_valid.txt",
                                                            "class", "tweet", True, reverse=reverse)




            colname = [i for i in list(test_df.columns) if i.startswith('tweet')][0]
            data_utils.df_to_RNNLM_labels(test_df, f"../twitter/augmented/{ao}/{o}/RNN_test.txt",
                                                        "class", colname, True, reverse=reverse)

            train.to_pickle(f"../twitter/augmented/{ao}/{o}/train.p")
            valid_df.to_pickle(f"../twitter/augmented/{ao}/{o}/valid.p")



