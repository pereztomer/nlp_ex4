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

# hyper parameters:
model_name = 't5-base'
max_seq_len = 128
run_name = f'{model_name}_{max_seq_len}_max_seq_len_short_sentences'
prefix = "translate German to English: "
epochs = 30
batch_size = 2
# wandb.init(project=run_name)


def get_model(model_checkpoint, datasets, source_lang, target_lang):
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    metric = load_metric("sacrebleu")

    model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

    model_name = model_checkpoint.split("/")[-1]
    args = Seq2SeqTrainingArguments(
        run_name,
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
        fp16=False,
        push_to_hub=False,
        report_to="wandb"
    )

    def preprocess_function(examples):
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

    tokenized_datasets = datasets.map(preprocess_function, batched=True)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

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
        decoded_labels = tokenizer.batch_decode(
            labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(
            decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds,
                                references=decoded_labels)
        result = {"bleu": result["score"]}

        prediction_lens = [np.count_nonzero(
            pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

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
    return model


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

    # train_dataset = Dataset.from_dict(load_ds_to_dict("data/train.labeled"))
    train_dataset = Dataset.from_dict(train_ds)
    validation_dataset = Dataset.from_dict(val_ds)
    datasets = DatasetDict({"train": train_dataset, "validation": validation_dataset})

    model = get_model(
        model_checkpoint="t5-base",
        datasets=datasets,
        source_lang="de",
        target_lang="en"
    )


def main():
    train()


if __name__ == '__main__':
    main()
