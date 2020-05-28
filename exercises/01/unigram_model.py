import math

class UnigramModel:
    def __init__(self):
        self.counts = {}
        self.total_count = 0
        self.probabilities = {}

    def train_model(self, training_file_path):
        with open(training_file_path, 'r') as f:
            training_file = f.read()
        for line in training_file.split("\n"):
            if len(line) > 1:
                words = line.split(" ")
                words.append("</s>")
                for word in words:
                    if word != '' and not word.isspace():
                        if word in self.counts:
                            self.counts[word] += 1
                        else:
                            self.counts[word] = 1
                        self.total_count += 1

    def calculate_probability(self):
        self.probabilities = {}
        for word, count in self.counts.items():
            probability = float(count) / self.total_count
            self.probabilities[word] = probability

    def write_model_to_file(self, output_file_path):
        self.probabilities = {}
        with open(output_file_path, 'w+') as f:
            for word, count in self.counts.items():
                probability = float(count) / self.total_count
                self.probabilities[word] = probability
                f.write("{}, {}".format(word, probability))

    def calculate_entropy(self, test_file_path):
        l_1 = .95
        l_unk = 1 - l_1
        V = 1000000
        W = 0
        H = 0
        unk = 0
        with open(test_file_path, 'r') as f:
            test_file = f.read()
        for line in test_file.split("\n"):
            if len(line) > 0:
                words = line.split(" ")
                words.append("</s>")
                for word in words:
                    W += 1
                    P = l_unk / V
                    if word in self.probabilities:
                        P += l_1 * self.probabilities[word]
                    else:
                        unk += 1
                    H += -math.log2(P)

        print("entropy = {}".format(H/W))
        print("coverage = {}".format((W-unk)/W))



