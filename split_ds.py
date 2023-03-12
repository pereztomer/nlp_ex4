from load_ds import load_ds_labeled
import json


def splits_paragraph(paragraph):
    sen_splits = paragraph.split('.')
    split_ds = []
    for sp in sen_splits:
        if sp == '' or len(sp) == 1:
            continue
        else:
            sub_splits = sp.split('?')
            for sub_sp in sub_splits:
                if sub_sp == '' or len(sub_sp) == 1:
                    continue
                else:
                    split_ds.append(sub_sp)
    return split_ds


def split_ds_to_sen_old():
    split_ds = []
    original_ds = load_ds_labeled(file_path='archive/data/train.labeled')
    for idx, val_dict in enumerate(original_ds):
        en_sentences = splits_paragraph(paragraph=val_dict['en'])
        ger_sentences = splits_paragraph(paragraph=val_dict['gr'])
        if len(en_sentences) == len(ger_sentences):
            for en_sen, gr_sen in zip(en_sentences, ger_sentences):
                temp_dict = {'en': en_sen, 'gr': gr_sen}
                split_ds.append(temp_dict)
        else:
            temp_dict = {'en': val_dict['en'], 'gr': val_dict['gr'], 'split_english': en_sentences}

            split_ds.append(temp_dict)

    print(len(split_ds))
    with open("archive/data/splits_ds_english_german.json", "w") as outfile:
        outfile.write(json.dumps(split_ds, indent=4))


def main():
    split_ds_to_sen_old()


if __name__ == '__main__':
    main()
