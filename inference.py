from transformers import pipeline
from load_ds import load_ds_unlabeled


def main():
    new_file_path = './data/val.labeled_self_made'
    unlabeled_ds = load_ds_unlabeled(path='./data/val.unlabeled')
    translator = pipeline("translation", model="./german_english_translator_baseline/checkpoint-8000")

    for idx, val in enumerate(unlabeled_ds):
        sen_to_translate = "translate German to English: "
        for sen in val['gr']:
            sen_to_translate += sen

        val['eg'] = translator(sen_to_translate)[0]['translation_text']

    with open(new_file_path, "w") as new_file:
        for idx, val in enumerate(unlabeled_ds):
            new_file.write('German:\n')
            for val_2 in val['gr']:
                new_file.write(val_2)
            new_file.write('English:\n')
            german_sen = val['eg'].split('.')
            for ger_sen in german_sen:
                if ger_sen != '':
                    new_file.write(ger_sen + '.\n')
            new_file.write('\n')


if __name__ == '__main__':
    main()
