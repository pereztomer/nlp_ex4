from transformers import pipeline
from load_ds import load_ds_unlabeled


def main():
    new_file_path = './data/val.labeled_self_made'
    unlabeled_ds = load_ds_unlabeled(path='./data/val.unlabeled')
    translator = pipeline("translation", model="./t5_small_200_max_seq_len/checkpoint-61500")

    for idx, val in enumerate(unlabeled_ds):
        sen_to_translate = "translate German to English: "
        for sen in val['gr']:
            sen_to_translate += sen

        val['eg'] = translator(sen_to_translate, max_length=420)[0]['translation_text']

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
