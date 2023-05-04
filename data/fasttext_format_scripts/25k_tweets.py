import pandas as pd
import FastTextSupervisedClassifier as fsc

from utils import preprocess_none
from utils import preprocess_simple
from utils import preprocess_full
from utils import preprocess_RNN

if __name__ == "__main__":
    df = pd.read_pickle('../twitter/labeled_data.p')

    # PREPROCESSING
    df["tweet"] = df["tweet"].copy().apply(preprocess_none)
    df["tweet_simple"] = df["tweet"].copy().apply(preprocess_simple)
    df["tweet_full"] = df["tweet"].copy().apply(preprocess_full)
    df["tweet_rnn"] = df["tweet"].copy().apply(preprocess_RNN)

    # Add binary classification column
    #df["class_binary"] = (df["class"] == 2).astype(int)

    fsc.split_df_dataset(df, 0.1, 0.2, 'tweet', 'class', '../twitter/preprocessed/multi/none')
    fsc.split_df_dataset(df, 0.1, 0.2, 'tweet_simple', 'class', '../twitter/preprocessed/multi/simple')
    fsc.split_df_dataset(df, 0.1, 0.2, 'tweet_full', 'class', '../twitter/preprocessed/multi/full')
    fsc.split_df_dataset(df, 0.1, 0.2, 'tweet_rnn', 'class', '../twitter/preprocessed/multi/rnn')

    #fsc.split_df_dataset(df, 0.1, 0.2, 'tweet', 'class_binary', '../twitter/preprocessed/binary')