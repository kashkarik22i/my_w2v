from gensim.models import Word2Vec
from gensim.test.utils import common_texts
from tokenizer import tokenize_file


def main():
    #data = tokenize_file("computer.txt")
    data = common_texts
    model = Word2Vec(data, sg=1, size=100, window=5, min_count=1, workers=1, min_alpha=0.025)
    model.save("their_w2v.model")
    print "computer: " + str(model.wv.most_similar("computer"))
    #print "flower: " + str(model.wv.most_similar("flower"))
    #print "data: " + str(model.wv.most_similar("data"))
    #print "earth: " + str(model.wv.most_similar("earth"))
    #print "condition: " + str(model.wv.most_similar("condition"))
    #print "artificial: " + str(model.wv.most_similar("artificial"))

if __name__ == "__main__":
    main()
