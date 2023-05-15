import fasttext

import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':

    model = fasttext.load_model("multi_model_classifier/FT_classifiers/original_full_10_false_better.ftz")


    args_obj = model.f.getArgs()
    for hparam in dir(args_obj):
        if not hparam.startswith('__'):
            print(f"{hparam} -> {getattr(args_obj, hparam)}")
    # FT built in classification
    df = pd.read_pickle(f'data/twitter/preprocessed/multi/full/test.p')
    res = [i for i in list(df.columns) if i.startswith('tweet')][0]
    df["prediction"] = df[res].apply(lambda x: model.predict(str(x).replace("\n", " "))[0][0][-1:])
    y_pred = df['prediction'].to_numpy().astype(int)
    y_test = df['class'].to_numpy().astype(int)
    print(classification_report(y_test, y_pred, digits=3))
    conf_mat = confusion_matrix(y_test, y_pred, normalize='true')
    print(conf_mat)

    df_cm = pd.DataFrame(conf_mat, index=["hate", "offensive", "neither"],
                         columns=["hate", "offensive", "neither"])
    # plt.figure(figsize=(10, 7))
    sns.set(font_scale=1.4)
    figure = sns.heatmap(df_cm, annot=True, cmap="Blues", cbar=False).get_figure()
    figure.savefig('single_model_heatmap.eps', format='eps')
    figure.savefig('single_model_heatmap.png', format='png')
