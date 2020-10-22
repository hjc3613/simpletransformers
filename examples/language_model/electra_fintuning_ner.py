from simpletransformers.language_modeling import LanguageModelingModel

model = LanguageModelingModel('electra', r'E:\checkpoint-120000',args={'output_dir':'discriminator_trained'}, use_cuda=True)
model = model.to('cpu')
model.save_discriminator()

from simpletransformers.ner import NERModel
import logging


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

train_file = "data/pos-tagging/pos-train.txt"
labels = ["O", "NOUN", "ADJ", "ADV", "VERB", "PRON"]

train_args = {
    "output_dir": "ner_output",
    "overwrite_output_dir": True,
}

model = NERModel("electra", "discriminator_trained/discriminator_model", args=train_args, labels=labels)

# Train the model
model.train_model(train_file)

# Evaluate the model
result, model_outputs, predictions = model.eval_model(train_file)

print(result)