from project_evaluate import calculate_score


def main():
    calculate_score(file_path1='./new_data/val.labeled',
                    file_path2='new_data/val.labeled_t5-base_250_max_seq_len_modifiers_train_val_from_model_2')

    # calculate_score(file_path1='./new_data/val.labeled',
    #                 file_path2='new_data/val.unlabeled')


if __name__ == '__main__':
    main()
