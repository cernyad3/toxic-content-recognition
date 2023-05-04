import pandas as pd

if __name__ == '__main__':
    print("hello")

    train_df = pd.read_pickle("data/twitter/preprocessed/multi/none/train.p")
    print((train_df[train_df["class"] == 0]).shape)

    test_df = pd.read_csv("data/twitter_100k/hatespeech_text_label_vote_RESTRICTED_100K.csv", names=["tweet", "class", "annotators"], sep='\t')

    hateful_df = test_df[test_df["class"] == "hateful"].copy()
    hateful_df["class"] = 0
    hateful_df = hateful_df.drop(columns=["annotators"])
    hateful_df = pd.concat([hateful_df, train_df[train_df["class"] == 0].copy()])
    hateful_df = hateful_df.sample(frac=1, random_state=0)
    print(hateful_df.head(20))

    normal_df = test_df[test_df["class"] == "normal"].copy()
    normal_df["class"] = 2
    normal_df = normal_df.drop(columns=["annotators"])
    normal_df = normal_df.sample(n=hateful_df.shape[0]-train_df[train_df["class"]==2].shape[0], random_state=0)
    normal_df = pd.concat([normal_df, train_df[train_df["class"] == 2].copy()])
    normal_df = normal_df.sample(frac=1, random_state=0)

    offensive_df = train_df[train_df["class"] == 1].copy()
    offensive_df = offensive_df.sample(n=normal_df.shape[0], random_state=0)


    new_train_df = pd.concat([hateful_df, offensive_df, normal_df])
    new_train_df = new_train_df.sample(frac=1, random_state=0)
    print(new_train_df[new_train_df["class"] == 0].shape)
    print(new_train_df[new_train_df["class"] == 1].shape)
    print(new_train_df[new_train_df["class"] == 2].shape)
    print(new_train_df.shape)
    print(new_train_df.head(20))

    new_train_df.to_csv("data/twitter/augmented/balanced.csv")







