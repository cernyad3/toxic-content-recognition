import os
import pathlib
from typing import List, Sequence

import fasttext
import numpy as np
from fasttext.FastText import _FastText

def process_sentence(
    input_sentence: Sequence[str],
    classifiers: Sequence[_FastText]
) -> np.ndarray:
    vector_arr = []
    # get all vectors from classifiers and put them in an array
    for i, model in enumerate(classifiers):
        sentence_vector = model.get_sentence_vector(input_sentence)

        vector_arr.append(sentence_vector)
    final_vector = np.array([item for sublist in vector_arr for item in sublist])

    return final_vector


def load_classifiers() -> List[_FastText]:
    """
    Loads all the fastText classifiers that are in classifiers_folder_path.
    :return: list of loaded fastText classifiers
    """
    classifiers_folder_path = pathlib.Path("FT_classifiers")

    if (
        not classifiers_folder_path.exists()
        or len(list(classifiers_folder_path.glob("*.ftz"))) == 0
    ):
        print("No classifiers found.")
        classifiers_folder_path.mkdir(exist_ok=True)
        exit()

    classifiers = []
    classifier_files = [
        f
        for f in os.listdir(classifiers_folder_path)
        if os.path.isfile(os.path.join(classifiers_folder_path, f))
        and f.endswith(".ftz")
    ]

    try:
        # silence the deprecation warnings as the package does not properly use
        # the python 'warnings' package see
        # https://github.com/facebookresearch/fastText/issues/1056
        fasttext.FastText.eprint = lambda *args, **kwargs: None

        for name in classifier_files:
            classifiers.append(
                fasttext.load_model(os.path.join(classifiers_folder_path, name))
            )
    except Exception as _:
        pass

    return classifiers


def embed_sentences(input_sentences: Sequence[str]) -> List[np.ndarray]:
    classifiers = load_classifiers()
    vector_sentences = []
    for sentence in input_sentences:
        vector_sentence = process_sentence(sentence, classifiers)
        vector_sentences.append(vector_sentence)
    return vector_sentences

