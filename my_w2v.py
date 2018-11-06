from tokenizer import tokenize_file
from gensim.test.utils import common_texts
import random
import math


def main():
    #data = tokenize_file("computer.txt")
    data = common_texts
    pairs = widows(data, widow=5)
    f = freq(data)
    net = Net(f, size=100, neg=5, iters=5, rate=0.005)
    net.train(pairs)

    print "computer: " + str(net.most_similar("computer"))
    # print "flower: " + str(net.most_similar("flower"))
    # print "data: " + str(net.most_similar("data"))
    # print "earth: " + str(net.most_similar("earth"))
    # print "condition: " + str(net.most_similar("condition"))
    # print "artificial: " + str(net.most_similar("artificial"))


def freq(sentences):
    f = {}
    for sentence in sentences:
        for word in sentence:
            if word not in f:
                f[word] = 0
            f[word] += 1
    return f


def widows(sentences, widow):
    res = []
    for sentence in sentences:
        res.extend(widows_for_sent(sentence, widow))
    return res


def widows_for_sent(sentence, window):
    pairs = []
    for i in range(len(sentence)):
        word = sentence[i]
        for j in range(i - window, i + window, 1):
            if i != j and 0 <= j < len(sentence):
                pairs.append((word, sentence[j]))
    return pairs


class Net(object):

    def __init__(self, freq, size, neg, iters, rate):
        self.rate = rate
        self.iters = iters
        self.neg = neg
        self.size = size
        self.freq = freq
        self.freq_sum = sum(f for f in freq.values())
        self.vocabulary = self.voc()
        self.vectors = []
        self.contexts = []

        self.init()

    def init(self):
        for i in range(len(self.vocabulary)):
            v = []
            c = []
            for j in range(self.size):
                v.append(random.uniform(0, 1))
                c.append(random.uniform(0, 1))
            self.vectors.append(v)
            self.contexts.append(c)

    def train(self, data):
        print "data size {}".format(len(data))
        for i in range(self.iters):
            print "iteration {}".format(i)
            j = 0
            for word, context in data:
                j += 1
                if j % 5 == 0:
                    print "{} samples".format(j)
                out = self.propagate(word)
                self.backprop(word, out, context)

    def backprop(self, word, out, expected):
        negs = self.sample_words()
        to_update = negs + [word]
        for word in to_update:
            self.update_word(word, out, expected)

    def update_word(self, word, out, expected):
        new_hidden = self.get_update_hidden(word, out)
        # old hidden should be used in here
        self.update_contexts(word, out, expected)
        self.vectors[self.vocabulary[word]] = new_hidden

    def get_update_hidden(self, word, out):
        index = self.vocabulary[word]
        h = self.vectors[index]
        e = [o for o in out]
        e[index] -= 1
        update = None
        for i in range(len(self.contexts)):
            if update is None:
                update = self.scalar(self.contexts[i], e[i])
            else:
                update = self.sum(update, self.scalar(self.contexts[i], e[i]))
        update = self.scalar(update, - self.rate)
        return self.sum(h, update)

    def update_contexts(self, word, out, expected):
        is_expected = 1 if word == expected else 0
        i = self.vocabulary[word]
        h = self.hidden(word)
        update = self.scalar(h, (is_expected - out[i]) * self.rate)
        self.contexts[i] = self.sum(self.contexts[i], update)

    @staticmethod
    def sum(v1, v2):
        return [x + y for x, y in zip(v1, v2)]

    @staticmethod
    def scalar(v, num):
        return [x * num for x in v]

    def voc(self):
        return {k: i for i, k in enumerate(list(self.freq.keys()))}

    def propagate(self, word):
        v = self.hidden(word)
        out = self.apply_contexts(v)
        return self.softmax(out)

    def apply_contexts(self, hidden):
        return [self.multiply(hidden, context) for context in self.contexts]

    def hidden(self, word):
        index = self.vocabulary[word]
        return self.vectors[index]

    @staticmethod
    def softmax(v):
        s = sum(math.exp(x) for x in v)
        return [math.exp(x) / s for x in v]

    @staticmethod
    def multiply(v1, v2):
        return sum([x * y for x, y in zip(v1, v2)])

    @staticmethod
    def cosine(v1, v2):
        v1s = math.sqrt(sum(x * x for x in v1))
        vn1 = [x / v1s for x in v1]
        v2s = math.sqrt(sum(x * x for x in v2))
        vn2 = [x / v2s for x in v2]
        return sum([x * y for x, y in zip(vn1, vn2)])

    def sample_words(self):
        return [self.sample_word() for _ in range(self.neg)]

    def sample_word(self):
        p = random.uniform(0, self.freq_sum)
        current = 0
        for word in self.vocabulary:
            f = self.freq[word]
            if current <= p < current + f:
                return word
            current += f

    def most_similar(self, word):
        pairs = []
        vec = self.hidden(word)
        for w in self.vocabulary:
            vec2 = self.hidden(w)
            pairs.append((w, self.cosine(vec, vec2)))
        return sorted(pairs, lambda x,y: 1 if x[1] < y[1] else -1)


if __name__ == "__main__":
    main()
