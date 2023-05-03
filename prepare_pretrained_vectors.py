import fasttext
import fasttext.util

if __name__ == '__main__':
    model = fasttext.load_model("model/unsupervised/wiki.simple.bin")
    print(model.get_dimension())

    fasttext.util.reduce_model(model, 100)
    print(model.get_dimension())

    model.save_model("model/unsupervised/wiki.simple.reduced_100.bin")

    #model = fasttext.load_model("model/unsupervised/wiki.simple.reduced_100.bin")
    lines = []

    # get all words from model
    words = model.get_words()

    with open("model/unsupervised/wiki.simple.reduced_100.vec", 'w') as file_out:
        # the first line must contain number of total words and vector dimension
        file_out.write(str(len(words)) + " " + str(model.get_dimension()) + "\n")

        # line by line, you append vectors to VEC file
        for w in words:
            v = model.get_word_vector(w)
            vstr = ""
            for vi in v:
                vstr += " " + str(vi)
            try:
                file_out.write(w + vstr + '\n')
            except:
                pass