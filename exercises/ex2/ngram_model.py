import math

class NgramModel:
    def __init__(self, n):
        self.ngram_counts = {}
        self.unigram_counts = {}
        self.context_counts = {"": 0}
        self.n = n
        self.ngram_probabilities = {}
        self.unigram_probabilities = {}

    def train_model(self, training_file_path):
        with open(training_file_path, 'r') as f:
            training_file = f.read()
        for line in training_file.split("\n"):
            if len(line) > 1:
                words = line.split(" ")
                words.append("</s>")
                words.insert(0, "</s>")
                for i in range(1, len(words)):
                    bigram = "{} {}".format(words[i-1], words[i])
                    if bigram in self.ngram_counts:
                        self.ngram_counts[bigram] += 1
                    else:
                        self.ngram_counts[bigram] = 1
                    if words[i-1] in self.context_counts:
                        self.context_counts[words[i-1]] += 1
                    else:
                        self.context_counts[words[i-1]] = 1
                    if words[i] in self.unigram_counts:
                        self.unigram_counts[words[i]] += 1
                    else:
                        self.unigram_counts[words[i]] = 1
                    self.context_counts[""] += 1

    def calculate_probability(self):
        self.ngram_probabilities = {}
        self.unigram_probabilities = {}
        for bigram, count in self.ngram_counts.items():
            context = bigram.split(" ")[0]
            probability = float(self.ngram_counts[bigram]) / self.context_counts[context]
            self.ngram_probabilities[bigram] = probability
        for word, count in self.unigram_counts.items():
            probability = float(count) / self.context_counts[""]
            self.unigram_probabilities[word] = probability

    def write_model_to_file(self, output_file_path):
        self.calculate_probability()
        with open(output_file_path, 'w+') as f:
            for bigram, probability in self.ngram_probabilities.items():
                f.write("{}, {}\n".format(bigram, probability))
            for word, probability in self.unigram_probabilities.items():
                f.write("{}, {}\n".format(word, probability))

    def witten_bell_smoothing(self, word):
        u = set()
        for bigram, count in self.ngram_counts.items():
            context = bigram.split(" ")[0]
            if context == word:
                u.add(bigram.split(" ")[1])
        count = self.unigram_counts[word] if word in self.unigram_counts else 0
        return (1 - (len(u)/(len(u)+count))) if (len(u)+count) > 0 else .05

    def calculate_entropy(self, test_file_path):
        V = 1000000
        W = 0
        H = 0
        with open(test_file_path, 'r') as f:
            test_file = f.read()
        for line in test_file.split("\n"):
            if len(line) > 0:
                words = line.split(" ")
                words.append("</s>")
                words.insert(0, "</s>")
                for i in range(1, len(words)):
                    l_1 = self.witten_bell_smoothing(words[i-1])
                    l_2 = 1 - l_1
                    ngram = "{} {}".format(words[i-1], words[i])
                    P1 = l_1 * self.unigram_probabilities[words[i]] + l_2/V if words[i] in self.unigram_probabilities else l_2/V
                    P2 = l_2 * self.ngram_probabilities[ngram] + (1-l_2) * P1 if ngram in self.ngram_probabilities else (1-l_2) * P1
                    H += -math.log2(P2)
                    W += 1

        print("entropy = {}".format(H/W))
