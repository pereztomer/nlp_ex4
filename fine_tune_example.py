from datasets import load_dataset
from transformers import AutoTokenizer


def define_preprocess_function(source_lang, target_lang, prefix, tokenizer):
    def preprocess_function(examples):
        inputs = [prefix + example[source_lang] for example in examples["translation"]]
        targets = [example[target_lang] for example in examples["translation"]]
        model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
        return model_inputs

    return preprocess_function


def main():
    source_lang = "en"
    target_lang = "fr"
    prefix = "translate English to French: "

    books = load_dataset("opus_books", "en-fr")
    books = books["train"].train_test_split(test_size=0.2)
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    preprocess_function = define_preprocess_function(source_lang=source_lang,
                                                     target_lang=target_lang,
                                                     prefix=prefix,
                                                     tokenizer=tokenizer)
    tokenized_books = books.map(preprocess_function, batched=True)
    for key, value in tokenized_books.values():
        print(key)
        print(value)


if __name__ == '__main__':
    main()
