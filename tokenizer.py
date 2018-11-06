import nltk
import io
from nltk.corpus import stopwords
stop = stopwords.words('english')


def tokenize_file(file_name):
    with io.open(file_name) as f:
        return tokenize(f.read())


def tokenize(text):
    result = []
    sent_text = nltk.sent_tokenize(text)
    for sentence in sent_text:
        tokenized_text = nltk.word_tokenize(sentence)
        lowercase = [w.lower() for w in tokenized_text]
        filtered = [w for w in lowercase if w not in stop and len(w) > 1]
        result.append(filtered)
    return result
