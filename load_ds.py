def load_ds(ds_path):
    ds = []
    with open(ds_path) as my_file:
        german_flag = False
        english_flag = False
        for line in my_file:
            if line == 'German:\n':
                german_sample = ''
                german_flag = True
                english_flag = False
            elif line == 'English:\n':
                english_sample = ''
                german_flag = False
                english_flag = True
            elif line == '\n':
                new_sample = {'en': english_sample.replace('\n', ' '), 'gr': german_sample.replace('\n', ' ')}
                ds.append(new_sample)
            elif german_flag:
                german_sample += line
            elif english_flag:
                english_sample += line
    return ds


def main():
    train_ds_path = './data/train.labeled'
    load_ds(ds_path=train_ds_path)


if __name__ == '__main__':
    main()
