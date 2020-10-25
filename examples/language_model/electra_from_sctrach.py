import torch

torch.cuda.empty_cache()

from torch.utils.data import Dataset
from tokenizers import ByteLevelBPETokenizer
from pathlib import Path
from simpletransformers.language_modeling import LanguageModelingModel

import logging

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

train_args = {
    "reprocess_input_data": False,
    "overwrite_output_dir": True,
    "num_train_epochs": 3,
    "save_eval_checkpoints": True,
    "save_model_every_epoch": False,
    "learning_rate": 5e-4,
    "warmup_steps": 10000,
    "train_batch_size": 64,
    "eval_batch_size": 64,
    "gradient_accumulation_steps": 1,
    "block_size": 128,
    "max_seq_length": 256,
    "dataset_type": "simple",
    "wandb_kwargs": {"name": "Electra-SMALL"},
    "logging_steps": 100,
    "evaluate_during_training": True,
    "evaluate_during_training_steps": 50000,
    "evaluate_during_training_verbose": True,
    "use_cached_eval_features": True,
    "sliding_window": True,
    "vocab_size": 21128,
    "generator_config": {
        "embedding_size": 128,
        "hidden_size": 256,
        "num_hidden_layers": 3,
        "vocab_size":21128,
    },
    "discriminator_config": {
        "embedding_size": 128,
        "hidden_size": 256,
        "vocab_size": 21128,
    },
    "use_multiprocessing":False,
    "wandb_project":False,
    "fp16":False,
    "save_steps":20000,
    "tokenizer_name":'outputs',
    "model_name":'outputs/checkpoint-120000'

}

train_file = r"train.txt"
test_file = r"test.txt"

model = LanguageModelingModel(
    "electra",
    args=train_args,
    train_files=train_file,
    use_cuda=False,
    model_name="outputs/checkpoint-120000"
)


model.train_model(
    train_file, eval_file=test_file,
)

model.eval_model(test_file)
