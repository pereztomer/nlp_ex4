import json
import os
import torch
from transformers import pipeline
from load_ds import load_ds_unlabeled_modifiers


def main():
    model_name = 't5-base_250_max_seq_len_modifiers_train_val_from_model_2'
    k_fold_num = 5
    models_checkpoints = [49500, 49500, 49500, 49500, 49500]
    path_2 = '/home/user/PycharmProjects/nlp_ex4/kfold/t5-base_250_max_seq_len_modifiers_train_val_from_model_2/fold_0/checkpoint-49500'
    path = 'kfold/t5-base_250_max_seq_len_modifiers_train_val_from_model_2/fold_0/checkpoint-49500'
    translator = pipeline("translation",
                          model=path,
                          device='cuda:0')

    exit()
    for index, model_ck in zip(range(k_fold_num), models_checkpoints):

        translator = pipeline("translation",
                              model=f'kfold/{model_name}/fold_{index}/checkpoint-{model_ck}',
                              device='cuda:0')
        val_set = json.load(open(f'kfold/{model_name}/fold_{index}/val.json'))
        for sample in val_set:
            print('hi')

    # sen_to_translate_lst = []
    # for idx, val in enumerate(unlabeled_ds):
    #     sen_to_translate = "translate German to English: "
    #     for sen in val['gr']:
    #         sen_to_translate += sen
    #     sen_to_translate_lst.append(sen_to_translate)
    #
    #
    # with open(new_file_path, "w") as new_file:
    #     for idx, (val, translated_eng_sen) in enumerate(zip(unlabeled_ds, translations)):
    #         new_file.write('German:\n')
    #         for counter, val_2 in enumerate(val['gr']):
    #             if counter == 0:
    #                 continue
    #             else:
    #                 new_file.write(val_2)
    #         new_file.write('English:\n')
    #         english_split_sen = translated_eng_sen['translation_text'].split('.')
    #         for eng_sen in english_split_sen:
    #             if eng_sen != '':
    #                 new_file.write(eng_sen + '.\n')
    #         new_file.write('\n')


if __name__ == '__main__':
    main()
