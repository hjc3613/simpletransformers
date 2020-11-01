from simpletransformers.language_modeling import LanguageModelingModel
import torch
torch.cuda.empty_cache()
# model = LanguageModelingModel('electra', r'discriminator_trained\\discriminator_model',args={'output_dir':'discriminator_trained'}, use_cuda=True)
# model.save_discriminator()

from simpletransformers.ner import NERModel
import logging


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

train_file = "data_set/train.ner.small.txt"
eval_file = 'data_set/dev.ner.small.txt'
# labels = ["O", "NOUN", "ADJ", "ADV", "VERB", "PRON"]
with open('data_set/labels.txt', encoding='utf8') as f:
    labels = f.readlines()
labels = [i.strip() for i in labels]

train_args = {
    "output_dir": "ner_output",
    "overwrite_output_dir": True,
    "use_multiprocessing":False,
    "save_steps":0,
    "use_early_stopping":True,
    "early_stopping_patience":4,
    "evaluate_during_training":True,
    "reprocess_input_data":False,
    "use_cached_eval_features":True,
    "fp16":False,
    "num_train_epochs": 10,
    "evaluate_during_training_steps":1,
    "train_batch_size": 32,
}

model = NERModel("electra", "outputs/best_model", args=train_args, labels=labels, use_cuda=True, crf=True)

# Train the model
model.train_model(train_file, eval_data=eval_file)

# Evaluate the model
test_file = 'data_set/test.ner.small.txt'
result, model_outputs, predictions = model.eval_model(train_file)

print(result)

# from transformers import ElectraTokenizer, ElectraForPreTraining
# model_name = r'D:\git_learn\simpletransformers\examples\language_model\discriminator_trained\discriminator_model'
# model = ElectraForPreTraining.from_pretrained(model_name)
# tokenizer = ElectraTokenizer.from_pretrained(model_name)
# sentence = '发烧头[MASK]3天'
# sentence = '患者自发病来，神志清楚，精神好'
# input_ids = torch.tensor(tokenizer.encode(sentence, add_special_tokens=True)).unsqueeze(0)
# output = model(input_ids, return_dict=True)
# # print(list(zip(list(output.logits.detach().numpy()), tokenizer.tokenize(sentence))))
# logits = list(output.logits.detach().numpy())
# print(len(logits))
# print(logits)
# tokens = tokenizer.tokenize(sentence)
# print(len(tokens))
# print(tokens)
# print(list(zip(logits, ['CLS']+tokens+['SEP'])))

