from datasets import Dataset
from transformers import AutoTokenizer
from load_ds import load_ds_labeled
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import DataCollatorForSeq2Seq
import evaluate
import numpy as np


# transformers on azure 4.6.0
# local version: 4.2.6
model_name = 't5-base'
max_seq_len = 128
run_name = f'{model_name}_{max_seq_len}_max_seq_len'
import wandb
wandb.init(project=run_name)

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def generate_compute_metrics(tokenizer, metric):
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    return compute_metrics


def define_preprocess_function(source_lang, target_lang, prefix, tokenizer):
    def preprocess_function(examples):
        inputs = [prefix + example[source_lang] for example in examples['data']]
        targets = [example[target_lang] for example in examples['data']]
        model_inputs = tokenizer(inputs, text_target=targets, max_length=max_seq_len, truncation=True)
        return model_inputs

    return preprocess_function


def main():
    train_ds_path = './data/train.labeled'
    val_ds_path = './data/val.labeled'
    train_ds = load_ds_labeled(file_path=train_ds_path)
    val_ds = load_ds_labeled(file_path=val_ds_path)

    source_lang = "gr"
    target_lang = "en"
    prefix = "translate German to English: "
    train_dataset = Dataset.from_dict({'data': train_ds})
    val_dataset = Dataset.from_dict({'data': val_ds})

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    preprocess_function = define_preprocess_function(source_lang=source_lang,
                                                     target_lang=target_lang,
                                                     prefix=prefix,
                                                     tokenizer=tokenizer)

    train_tokenized_ds = train_dataset.map(lambda batch: preprocess_function(batch), batched=True)
    val_tokenized_ds = val_dataset.map(lambda batch: preprocess_function(batch), batched=True)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    sacrebleu = evaluate.load("sacrebleu")

    compute_metrics = generate_compute_metrics(tokenizer=tokenizer, metric=sacrebleu)

    training_args = Seq2SeqTrainingArguments(
        output_dir=run_name,
        evaluation_strategy="epoch",
        save_strategy='epoch',
        logging_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=0.001,
        save_total_limit=15,
        load_best_model_at_end=True,
        #metric_for_best_model=sacrebleu,
        num_train_epochs=100,
        predict_with_generate=True,
        fp16=True,
        push_to_hub=False,
        report_to="wandb"
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized_ds,
        eval_dataset=val_tokenized_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(f'{run_name}/best_model')
    wandb.finish()


if __name__ == '__main__':
    main()
