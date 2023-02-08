from deep_translator import GoogleTranslator
from load_ds import load_ds_labeled


def main():
    original_ds = load_ds_labeled(file_path='./data/train.labeled')
    translated_ds = []
    for idx, val_dict in enumerate(original_ds):
        sen_splits = val_dict['en'].split('.')
        for sp in sen_splits:
            if sp == '' or len(sp) ==1:
                continue
            else:
                sub_splits = sp.split('?')
                for sub_sp in sub_splits:
                    if sub_sp == '' or len(sub_sp) ==1:
                        continue
                    else:
                        translated = GoogleTranslator(source='auto', target='de').translate(sub_sp)
                        new_dict = {'en': sub_sp, 'gr':translated}
                        translated_ds.append(new_dict)
    print(len(translated_ds))

if __name__ == '__main__':
    main()
