from project_evaluate import calculate_score


def main():
    calculate_score(file_path1='./data/val.labeled',
                    file_path2='data/val.labeled_t5-base_128_max_seq_len_short_sentences_no_fp_16')


if __name__ == '__main__':
    main()
