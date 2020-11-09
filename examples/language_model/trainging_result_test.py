from simpletransformers.ner import NERModel

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
    "evaluate_during_training_steps":10000,
    "train_batch_size": 32,
    'cross_entropy_ignore_index':0,
    'classification_report':True
}

model = NERModel("electra", "ner_output/checkpoint-150000", args=train_args, labels=labels, use_cuda=True, crf=True)

result = model.predict([list('发烧头痛3天'), list('见盲肠底，升结肠近肝曲')], split_on_space=False)
print(result)