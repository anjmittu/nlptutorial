import os
from ngram_model import NgramModel

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

test_input1 = os.path.join(PROJECT_ROOT, "test/ex1-train-input.txt")
test_input2 = os.path.join(PROJECT_ROOT, "test/ex1-test-input.txt")
test_output = os.path.join(PROJECT_ROOT, "exercises/ex2/output/test_output.txt")

# Train the model
model_test = NgramModel(2)
model_test.train_model(test_input1)
model_test.write_model_to_file(test_output)
model_test.calculate_entropy(test_input2)
