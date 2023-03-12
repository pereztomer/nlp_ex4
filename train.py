from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from transformers import (AutoTokenizer,
                          AutoModelForSeq2SeqLM,
                          Seq2SeqTrainingArguments,
                          Seq2SeqTrainer,
                          DataCollatorForSeq2Seq)
from datasets import load_metric
import numpy as np
import wandb
from load_ds import load_ds_to_dict
import json
from sklearn.model_selection import KFold

from pynvml import *

# hyper parameters:
model_name = 't5-base'
max_seq_len = 250
run_name = f'{model_name}_{max_seq_len}_max_seq_len_modifiers_train_val_from_model_2'
prefix = "translate German to English: "
epochs = 45
batch_size = 4
# wandb.init(project=run_name)


def print_gpu_utilization():
    """
    GPU Usage tracking
    """
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used // 1024 ** 2} MB.")


def print_summary(result):
    """
    Training tracker
    """
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()


def get_model(model_checkpoint, datasets, source_lang, target_lang, fold_num):
    """
    This function
    Parameters
    ----------
    model_checkpoint:
    datasets:
    source_lang:
    target_lang:
    fold_num:

    Returns
    -------

    """

    def preprocess_function(examples):
        """
        Preprocesses given examples, so they can be used as model inputs
        """
        inputs = [prefix + ex[source_lang] for ex in examples["translation"]]
        targets = [ex[target_lang] for ex in examples["translation"]]
        model_inputs = tokenizer(
            inputs, max_length=max_seq_len, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets, max_length=max_seq_len, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    metric = load_metric("sacrebleu")

    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

    model_name = model_checkpoint.split("/")[-1]
    args = Seq2SeqTrainingArguments(  # Setting the arguments for the Seq2Seq model training
        f'kfold/{run_name}/fold_{fold_num}',
        gradient_accumulation_steps=2,
        save_strategy='epoch',
        logging_strategy="epoch",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=epochs,
        num_train_epochs=epochs,
        predict_with_generate=True,
        fp16=True,
        push_to_hub=False,
        report_to="wandb"
    )

    tokenized_datasets = datasets.map(preprocess_function, batched=True)

    # Data collators are objects that will form a batch by using a list of dataset elements as input.
    data_collator = DataCollatorForSeq2Seq(tokenizer,
                                           model=model)  # Data collator that will dynamically pad the inputs received, as well as the labels.

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()


def load_parsed_ds(file_path):
    ds = json.load(open(file_path))
    out_ds = []
    for val in ds:
        new_dict = {}
        new_dict['en'] = val['en']

        intro_sen = ''
        for value in val['parsing_tree']:
            root_index = value[1].index(0)
            root = value[0][root_index]
            # get modifiers:
            modifiers_to_add = ''
            modifiers_of_root_indexes = [idx for idx, value_inner in enumerate(value[1]) if value_inner == root_index]
            if len(modifiers_of_root_indexes) >= 2:
                modifiers_to_add = modifiers_to_add + value[0][modifiers_of_root_indexes[0]]
                modifiers_to_add = modifiers_to_add + ', ' + value[0][modifiers_of_root_indexes[1]]
            elif len(modifiers_of_root_indexes) == 1:
                modifiers_to_add = modifiers_to_add + value[0][modifiers_of_root_indexes[0]]
            intro_sen += f'sentence root: {root}, root modifiers: {modifiers_to_add}, '

        new_dict['de'] = intro_sen + ' German sentences to translate: ' + val['de']
        out_ds.append(new_dict)
    return {'translation': out_ds}


def train():
    train_path = './new_data/data_for_training/train_ds_dependency_parsed.json'
    train_ds = load_parsed_ds(train_path)
    val_path = './new_data/data_for_training/val_ds_dependency_parsed.json'
    val_ds = load_parsed_ds(val_path)
    complete_ds = train_ds['translation'] + val_ds['translation']

    kf = KFold(n_splits=5, random_state=500, shuffle=True)  # 5-Fold Cross Validation
    for i, (train_index, val_index) in enumerate(kf.split(complete_ds)):
        print(f"Fold {i}:")
        train_ds = []
        for index in train_index:
            train_ds.append(complete_ds[index])
        val_ds = []
        for index in val_index:
            val_ds.append(complete_ds[index])
        train_dataset = Dataset.from_dict({'translation': train_ds})
        validation_dataset = Dataset.from_dict({'translation': val_ds})
        datasets = DatasetDict({"train": train_dataset, "validation": validation_dataset})

        get_model(
            model_checkpoint="t5-base",
            datasets=datasets,
            source_lang="de",
            target_lang="en",
            fold_num=i
        )
        train_dataset = None
        validation_dataset = None
        datasets = None


def main():
    train()


if __name__ == '__main__':
    main()
