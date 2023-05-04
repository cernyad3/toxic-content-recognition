import re
import pandas as pd
from nltk.corpus import stopwords

def camel_case_split(matchobj):
    str = matchobj.group(0)
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', str)
    return " ".join([m.group(0) for m in matches])


def preprocess_tweet(tweet):
    STOPWORDS = set(stopwords.words('english'))

    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|''[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    hashtag_regex = '#[\w\-]+'


    parsed_text = re.sub(space_pattern, ' ', tweet)
    parsed_text = re.sub(giant_url_regex, '', parsed_text)
    parsed_text = re.sub(mention_regex, '', parsed_text)
    parsed_text = re.sub(hashtag_regex, camel_case_split, parsed_text)

    parsed_text = re.sub(r'\sRT\s', "", parsed_text)
    parsed_text = re.sub(r'^RT\s', "", parsed_text)

    #parsed_text = re.sub('[^A-Za-z!?,.\'\s@]+', '', parsed_text)
    #TODO solve case hello,world
    parsed_text = re.sub('[^A-Za-z\s]+', '', parsed_text)

    querywords = parsed_text.split()
    resultwords = [word for word in querywords if word.lower() not in STOPWORDS]
    parsed_text = ' '.join(resultwords)

    return parsed_text.strip().lower()



def preprocess_full(tweet):
    STOPWORDS = set(stopwords.words('english'))

    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|''[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    hashtag_regex = '#[\w\-]+'

    parsed_text = re.sub(space_pattern, ' ', tweet)
    parsed_text = re.sub(giant_url_regex, '', parsed_text)
    parsed_text = re.sub(mention_regex, '', parsed_text)
    parsed_text = re.sub(hashtag_regex, camel_case_split, parsed_text)

    parsed_text = re.sub(r'\sRT\s', "", parsed_text)
    parsed_text = re.sub(r'^RT\s', "", parsed_text)

    # parsed_text = re.sub('[^A-Za-z!?,.\'\s@]+', '', parsed_text)

    parsed_text = re.sub('[^A-Za-z\s]+', '', parsed_text)

    querywords = parsed_text.split()
    resultwords = [word for word in querywords if word.lower() not in STOPWORDS]
    parsed_text = ' '.join(resultwords)

    return parsed_text.strip().lower()

def preprocess_simple(tweet):
    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|''[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'

    parsed_text = re.sub(space_pattern, ' ', tweet)
    parsed_text = re.sub(giant_url_regex, '', parsed_text)
    parsed_text = re.sub(mention_regex, '', parsed_text)


    parsed_text = re.sub(r'\sRT\s', "", parsed_text)
    parsed_text = re.sub(r'^RT\s', "", parsed_text)

    parsed_text = re.sub('[^A-Za-z!?,.\'\s]+', '', parsed_text)


    return parsed_text.strip().lower()

def preprocess_RNN(tweet):
    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|''[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'

    parsed_text = re.sub(space_pattern, ' ', tweet)
    parsed_text = re.sub(giant_url_regex, '', parsed_text)
    parsed_text = re.sub(mention_regex, 'longuserplaceholderrrrrrrrrr', parsed_text)


    #parsed_text = re.sub(r'\sRT\s', "", parsed_text)
    #parsed_text = re.sub(r'^RT\s', "", parsed_text)

    #parsed_text = re.sub('[^A-Za-z!?,.\'\s]+', '', parsed_text)
    parsed_text = re.sub('[^A-Za-z\s]+', '', parsed_text)
    parsed_text = re.sub('longuserplaceholderrrrrrrrrr', '<user>', parsed_text)


    return parsed_text.strip().lower()



def preprocess_none(tweet):
    return tweet

if __name__ == '__main__':
    print(preprocess_tweet("@pussyriot hello @mkkm kks dawkdwak"))

