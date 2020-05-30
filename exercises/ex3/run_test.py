import os
from word_segmentation_model import WordSegmentationModel

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

training_model = os.path.join(PROJECT_ROOT, "test/04-model.txt")
test_input = os.path.join(PROJECT_ROOT, "test/04-input.txt")

# Train the model
model_test = WordSegmentationModel()
model_test.train_model(training_model)
model_test.segment_word(test_input)
