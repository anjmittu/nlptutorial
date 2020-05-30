import os
from unigram_model import UnigramModel

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

training_input = os.path.join(PROJECT_ROOT, "data/wiki-en-train.word")
test_input1 = os.path.join(PROJECT_ROOT, "test/ex1-train-input.txt")
test_input2 = os.path.join(PROJECT_ROOT, "test/ex1-test-input.txt")
test_output = os.path.join(PROJECT_ROOT, "exercises/ex1/output/test_output.csv")

# Train the model
model_test = UnigramModel()
model_test.train_model(test_input1)
model_test.write_model_to_file(test_output)

model_test.calculate_entropy(test_input2)
