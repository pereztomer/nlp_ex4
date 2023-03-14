from translate.storage.tmx import tmxfile


def read_tmx_file(file_path):
    ds_list = []
    with open(file_path, 'rb') as fin:
        tmx_file = tmxfile(fin, 'de-DE', 'en-GB')

    for node in tmx_file.unit_iter():
        ds_list.append({'de': node.source, 'en': node.target})

    return {'translation': ds_list}


if __name__ == '__main__':
    read_tmx_file()
