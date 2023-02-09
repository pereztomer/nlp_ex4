from deep_translator import GoogleTranslator
from load_ds import load_ds_labeled
import json
from project_evaluate import compute_metrics, read_file


def calculate_score(file1_en, file1_de, file2_en, file2_de):
    for idx, (sen1, sen2) in enumerate(zip(file1_de, file2_de)):
        if sen1.strip().lower() != sen2.strip().lower():
            raise ValueError('Different Sentences')
    score = compute_metrics(file1_en, file2_en)
    print(score)


def generate_translated_ds():
    original_ds = load_ds_labeled(file_path='./data/train.labeled')
    translated_ds = []
    for idx, val_dict in enumerate(original_ds):
        en_sen = val_dict['en']
        translated = GoogleTranslator(source='auto', target='de').translate(en_sen)
        new_dict = {'en': en_sen, 'gr': translated}
        translated_ds.append(new_dict)
        if idx % 250 == 0:
            print(f'Done {idx}/10000 samples')

    with open("./translation_tests/complete_translation.json", "w") as outfile:
        outfile.write(json.dumps(translated_ds, indent=4))


def generate_sentence_ds():
    original_ds = load_ds_labeled(file_path='./data/train.labeled')
    translated_ds = []
    for idx, val_dict in enumerate(original_ds):
        sen_splits = val_dict['en'].split('.')
        for sp in sen_splits:
            if sp == '' or len(sp) == 1:
                continue
            else:
                sub_splits = sp.split('?')
                for sub_sp in sub_splits:
                    if sub_sp == '' or len(sub_sp) == 1:
                        continue
                    else:
                        translated = GoogleTranslator(source='auto', target='de').translate(sub_sp)
                        new_dict = {'en': sub_sp, 'gr': translated}
                        translated_ds.append(new_dict)
    print(len(translated_ds))


def main():
    translated_ds = json.load(open('./translation_tests/'))
    file2_en, file2_de = [], []
    for value in translated_ds:
        en_sentence = value['en']
        german_sentence = value['gr']
        file2_en.append(en_sentence)
        file2_de.append(german_sentence)

    file1_en, file1_de = read_file(file_path='./data/val.labeled')
    calculate_score(file1_en, file1_de, file2_en, file2_de)


if __name__ == '__main__':
    generate_translated_ds()
