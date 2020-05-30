import os
import sys
import math

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(PROJECT_ROOT, "exercises"))
from ex1.unigram_model import UnigramModel


class WordSegmentationModel:
    def __init__(self):
        self.model = UnigramModel()

    def train_model(self, training_file_path):
        self.model.load_model_from_file(training_file_path)

    def segment_word(self, file_path):
        with open(file_path, 'r') as f:
            data = f.read()
        for line in data.split("\n"):
            best_edge = {0: None}
            best_score= {0: 0}
            line_decoded = str(line)
            for word_end in range(1, len(line_decoded)+1):
                best_score[word_end] = 10 ** 6
                for word_start in range(word_end):
                    word = line_decoded[word_start:word_end]
                    if self.model.is_word_in_model(word):
                        prob = self.model.probability_of_word(word)
                        my_score = best_score[word_start] - math.log(prob)
                        if my_score < best_score[word_end]:
                            best_score[word_end] = my_score
                            best_edge[word_end] = (word_start, word_end)
            words = []
            next_edge = best_edge[len(best_edge) - 1]
            while next_edge:
                words.append(line[next_edge[0]:next_edge[1]])
                next_edge = best_edge[next_edge[0]]
            words.reverse()
            print(" ".join(words))