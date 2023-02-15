import torch
from transformers import pipeline
from load_ds import load_ds_unlabeled_modifiers


def main():
    model_name = 't5-base_128_max_seq_len_short_sentences'
    new_file_path = f'data/val.labeled_{model_name}'
    unlabeled_ds = load_ds_unlabeled_modifiers(path='./data/val.unlabeled')
    ds_processed = []
    for val in unlabeled_ds:
        intro_sen = ''
        new_dict = {}
        for root, modifiers in zip(val['Roots in English'], val['Modifiers in English']):
            modifiers_to_add = ''
            for m in modifiers:
                modifiers_to_add = modifiers_to_add + m + ', '
            intro_sen += f'sentence root: {root}, root modifiers: {modifiers_to_add}'

        zero_entry = intro_sen + ' German sentences to translate: '
        val['gr'].insert(0, zero_entry)

        ds_processed.append(new_dict)

    translator = pipeline("translation", model=f'{model_name}/checkpoint-28847', device='cuda:0')
    sen_to_translate_lst = []
    for idx, val in enumerate(unlabeled_ds):
        sen_to_translate = "translate German to English: "
        for sen in val['gr']:
            sen_to_translate += sen
        sen_to_translate_lst.append(sen_to_translate)

    translations = translator(sen_to_translate_lst, max_length=420)

    with open(new_file_path, "w") as new_file:
        for idx, (val, translated_eng_sen) in enumerate(zip(unlabeled_ds, translations)):
            new_file.write('German:\n')
            for val_2 in val['gr']:
                new_file.write(val_2)
            new_file.write('English:\n')
            english_split_sen = translated_eng_sen['translation_text'].split('.')
            for eng_sen in english_split_sen:
                if eng_sen != '':
                    new_file.write(eng_sen + '.\n')
            new_file.write('\n')


if __name__ == '__main__':
    main()
