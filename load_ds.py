from project_evaluate import read_file


# def load_ds_labeled(ds_path):
#     ds = []
#     with open(ds_path) as my_file:
#         german_flag = False
#         english_flag = False
#         for line in my_file:
#             if line == 'German:\n':
#                 german_sample = ''
#                 german_flag = True
#                 english_flag = False
#             elif line == 'English:\n':
#                 english_sample = ''
#                 german_flag = False
#                 english_flag = True
#             elif line == '\n':
#                 new_sample = {'en': english_sample.replace('\n', ' '), 'gr': german_sample.replace('\n', ' ')}
#                 ds.append(new_sample)
#             elif german_flag:
#                 german_sample += line
#             elif english_flag:
#                 english_sample += line
#     return ds

def load_ds_labeled(file_path):
    out_ds = []
    ds = read_file(file_path=file_path)
    for english_se, german_sen in zip(ds[0], ds[1]):
        out_ds.append({'en': english_se, 'gr': german_sen})
    return out_ds


def load_ds_to_dict(path):
    data_dict = {'translation': []}
    sample = {}
    with open(path) as f:
        for line in f:

            if line == "German:\n":
                status = "de"
                sample[status] = ""

            elif line == "English:\n":
                status = "en"
                sample[status] = ""

            elif line != '\n':
                sample[status] += line

            else:
                data_dict['translation'].append(sample)
                sample = {}

    return data_dict


def load_ds_unlabeled(path):
    ds = []
    with open(path) as my_file:
        for line in my_file:
            if line == 'German:\n':
                german_sample = []
            elif line == '\n':
                new_sample = {'Roots in English': roots,
                              'Modifiers in English': modifiers,
                              'gr': german_sample}
                ds.append(new_sample)
            elif 'Roots in English:' in line:
                roots = line
            elif 'Modifiers in English:' in line:
                modifiers = line
            else:
                german_sample.append(line)

    return ds


def load_ds_unlabeled_modifiers(path):
    ds = []
    with open(path) as my_file:
        for line in my_file:
            if line == 'German:\n':
                german_sample = []
            elif line == '\n':
                new_sample = {'Roots in English': roots,
                              'Modifiers in English': modifiers_per_sentence_list,
                              'gr': german_sample}
                ds.append(new_sample)
            elif 'Roots in English:' in line:
                roots = line.replace('Roots in English:', '').replace('\n', '').replace(' ', '').split(',')
                if roots[-1] == '':
                    roots = roots[:-1]
            elif 'Modifiers in English:' in line:
                modifiers_per_sentence_list = []
                modifiers = line.replace('Modifiers in English: ', '').replace('\n', '').replace(' ', '').split('),')
                if modifiers[-1] == ' ':
                    modifiers = modifiers[:-1]
                for mod_sen in modifiers:
                    modifiers_in_sentence = mod_sen.replace('(', '').replace(')', '').replace(' ', '').split(',')
                    if modifiers_in_sentence[-1] == ' ':
                        modifiers_in_sentence = modifiers_in_sentence[:-1]
                    modifiers_per_sentence_list.append(modifiers_in_sentence)
            else:
                german_sample.append(line)

    return ds


def main():
    val_ds_path = 'archive/data/val.labeled'
    ds = load_ds_labeled(file_path=val_ds_path)
    print('hi')


if __name__ == '__main__':
    main()
