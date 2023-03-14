import json
import evaluate
import numpy as np
from transformers import pipeline


def main():
    model_name = 't5-base_250_max_seq_len_modifiers_train_val_from_model_2'
    k_fold_num = 5
    models_checkpoints = [49500, 49500, 49500, 49500, 49500]
    blue_list = []
    for index, model_ck in zip(range(k_fold_num), models_checkpoints):

        translator = pipeline("translation",
                              model=f'kfold/{model_name}/fold_{index}/checkpoint-{model_ck}',
                              device='cuda:0')
        val_set = json.load(open(f'kfold/{model_name}/fold_{index}/val.json'))
        original_eng_sentences_list = []
        ger_sentences_for_translation = []
        for i, sample in enumerate(val_set):
            original_eng_sen = sample['en']
            original_eng_sentences_list.append(original_eng_sen)
            german_sen_to_translate = sample['de']
            ger_sentences_for_translation.append(german_sen_to_translate)

        translations = translator(ger_sentences_for_translation, max_length=420)
        english_translations = []
        for val in translations:
            english_translations.append(val['translation_text'])

        fold_results = list(zip(ger_sentences_for_translation,
                                english_translations,
                                original_eng_sentences_list))

        with open(f"kfold/{model_name}/fold_{index}/fold_results.json", "w") as outfile:
            outfile.write(json.dumps(fold_results, indent=4))

        metric = evaluate.load("sacrebleu")
        result = metric.compute(predictions=original_eng_sentences_list,
                                references=english_translations)
        result = result['score']
        fold_blue = round(result, 2)
        blue_list.append(fold_blue)
        print(f'Fold {index}: {fold_blue}')

    print(f'avg fold blue: {np.average(blue_list)}, std fold blue: {np.std(blue_list)}')

    results = {'folds': blue_list, 'avg': np.average(blue_list), 'std': np.std(blue_list)}

    with open(f"kfold/{model_name}blue.json", "w") as outfile:
        outfile.write(json.dumps(results, indent=4))


if __name__ == '__main__':
    main()
