from deep_translator import GoogleTranslator
from load_ds import load_ds_labeled
import json
from project_evaluate import compute_metrics, read_file
import re


def calculate_score(file1_en, file1_de, file2_en, file2_de):
    for idx, (sen1, sen2) in enumerate(zip(file1_en, file2_en)):
        if sen1.strip().lower() != sen2.strip().lower():
            raise ValueError('Different Sentences')
    score = compute_metrics(file1_de, file2_de)
    print(score)


def add_sigh(full_paragraph, sen, sign):
    sen_initial_index = full_paragraph.find(sen)
    if sen_initial_index + len(sen) == len(full_paragraph):
        return sen
    if sign in full_paragraph[sen_initial_index:sen_initial_index + len(sen)+1]:
        sen += sign
        return sen
    return sen


def split_ds_to_sen(ds_path):
    split_ds = []
    original_ds = load_ds_labeled(file_path=ds_path)
    for idx, val_dict in enumerate(original_ds):
        sen_splits = val_dict['en'].split('.')
        for inner_idx, sp in enumerate(sen_splits):
            if sp == '' or len(sp) <= 2:
                continue
            else:
                sp = add_sigh(full_paragraph=val_dict['en'], sen=sp, sign='.')
                sub_splits = sp.split('?')
                for sub_sp in sub_splits:
                    if sub_sp == '' or len(sub_sp) <= 2:
                        continue
                    else:
                        sub_sp = add_sigh(full_paragraph=val_dict['en'], sen=sub_sp, sign='?')
                        split_ds.append({'en': sub_sp})

    with open("./translation_tests/splits_ds_english.json", "w") as outfile:
        outfile.write(json.dumps(split_ds, indent=4))


def generate_translated_ds():
    original_ds = json.load(open('./translation_tests/splits_ds_english.json'))

    translated_ds = []
    for idx, val_dict in enumerate(original_ds):
        en_sen = val_dict['en']
        try:
            translated = GoogleTranslator(source='auto', target='de').translate(en_sen)
        except Exception as e:
            print('Failed sentence:')
            print(en_sen)
            print(e)
            translated = 'Failed to translate'

        new_dict = {'en': en_sen, 'gr': translated}
        translated_ds.append(new_dict)
        if idx % 250 == 0:
            print(f'finished: {idx} / 29290')

    with open("./translation_tests/complete_translation_for_split_ds.json", "w") as outfile:
        outfile.write(json.dumps(translated_ds, indent=4))


def compare_ds():
    translated_ds = json.load(open('./translation_tests/complete_translation.json'))
    file2_en, file2_de = [], []
    for value in translated_ds:
        en_sentence = value['en']
        german_sentence = value['gr']
        file2_en.append(en_sentence)
        file2_de.append(german_sentence)

    file1_en, file1_de = read_file(file_path='./data/train.labeled')
    calculate_score(file1_en, file1_de, file2_en, file2_de)


def main():
    ds_path = './data/train.labeled'
    split_ds_to_sen(ds_path=ds_path)


if __name__ == '__main__':
    main()
