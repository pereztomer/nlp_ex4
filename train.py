from datasets import Dataset
from transformers import AutoTokenizer
from load_ds import load_ds
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import DataCollatorForSeq2Seq
import evaluate
import numpy as np


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
        model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
        return model_inputs

    return preprocess_function


def main():
    train_ds_path = './data/train.labeled'
    val_ds_path = './data/val.labeled'
    train_ds = load_ds(ds_path=train_ds_path)
    val_ds = load_ds(ds_path=val_ds_path)

    source_lang = "gr"
    target_lang = "en"
    prefix = "translate German to English: "

    train_dataset = Dataset.from_dict({'data': train_ds})
    val_dataset = Dataset.from_dict({'data': val_ds})

    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    preprocess_function = define_preprocess_function(source_lang=source_lang,
                                                     target_lang=target_lang,
                                                     prefix=prefix,
                                                     tokenizer=tokenizer)

    train_tokenized_ds = train_dataset.map(lambda batch: preprocess_function(batch), batched=True)
    val_tokenized_ds = val_dataset.map(lambda batch: preprocess_function(batch), batched=True)

    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    sacrebleu = evaluate.load("sacrebleu")

    compute_metrics = generate_compute_metrics(tokenizer=tokenizer, metric=sacrebleu)

    training_args = Seq2SeqTrainingArguments(
        output_dir="my_awesome_opus_books_model",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=24,
        per_device_eval_batch_size=24,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=2,
        predict_with_generate=True,
        fp16=False,
        push_to_hub=False,
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


if __name__ == '__main__':
    main()
