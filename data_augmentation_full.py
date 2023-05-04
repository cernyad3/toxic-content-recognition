import pandas as pd


if __name__ == '__main__':
    print("hello")

    train_df = pd.read_pickle("data/twitter/preprocessed/multi/none/train.p")
    print((train_df[train_df["class"] == 0]).shape)

    test_df = pd.read_csv("data/twitter_100k/hatespeech_text_label_vote_RESTRICTED_100K.csv", names=["tweet", "class", "annotators"], sep='\t')

    test_df.loc[test_df["class"] == "hateful", "class"] = 0
    test_df.loc[test_df["class"] == "abusive", "class"] = 1
    test_df.loc[test_df["class"] == "normal", "class"] = 2

    test_df = test_df.drop(columns=["annotators"])
    test_df = test_df[test_df["class"] != "spam"]

    combined_df = pd.concat([test_df, train_df])
    combined_df = combined_df.sample(frac=1, random_state=0)
    combined_df = combined_df.sample(frac=1, random_state=0)


    print(combined_df[combined_df["class"] == 0].shape)
    print(combined_df[combined_df["class"] == 1].shape)
    print(combined_df[combined_df["class"] == 2].shape)
    print(combined_df.shape)
    print(combined_df.head(20))

    combined_df.to_csv("data/twitter/augmented/full.csv")







