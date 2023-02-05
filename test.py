from project_evaluate import calculate_score


def main():
    calculate_score(file_path1='./data/val.labeled', file_path2='./data/val.labeled_self_made')


if __name__ == '__main__':
    main()
