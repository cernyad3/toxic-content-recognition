import pandas as pd
import numpy as np

if __name__ == "__main__":

    paths = ["preprocessed/multi", "augmented/balanced", "augmented/full"]

    preps = ["full", "rnn"]

    for pr in preps:
        for p in paths:
            # print("-" * 60)
            # print(p, pr)

            df = pd.read_pickle(f'../twitter/{p}/{pr}/train.p')
            colname = [i for i in list(df.columns) if i.startswith('tweet')][0]

            ln = df.shape[0]

            ratios = df["class"].value_counts(normalize=True).apply(lambda x: round(x * 100, 2))
            rat_hate = ratios[0]
            rat_off = ratios[1]
            rat_norm = ratios[2]




            df["len"] = df[colname].apply(lambda x: len(str(x).split()))
            med = df["len"].median()

            print(f"{ln}\t{rat_hate}\t{rat_off}\t{rat_norm}\t{int(med)}")
            #print(df["len"].mean())

            # print("*" * 60)










