import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from chu_liu_edmonds import decode_mst
import json

from datasets.arrow_dataset import Dataset
from load_ds import load_ds_to_dict

from split_ds import splits_paragraph


def parse_comp_file(file_address):
    with open(file_address, encoding='utf-8') as f:
        sentences = []  # Contains the final sentences without tags
        sentence_positions = []
        sentences_real_len = []

        new_sentence = []
        new_sentence_pos = []

        for row in f:
            if row != '\n':
                token = row.split('\t')[1]
                token_pos = row.split('\t')[3]
                new_sentence.append(token)
                new_sentence_pos.append(token_pos)
            else:
                new_sentence.insert(0, 'ROOT')
                new_sentence_pos.insert(0, 'init')

                sentences.append(new_sentence)
                sentence_positions.append(new_sentence_pos)
                sentences_real_len.append(len(new_sentence))
                new_sentence = []
                new_sentence_pos = []

    return sentences, sentence_positions, sentences_real_len


def padding_(sentences, seq_len):
    features = np.zeros((len(sentences), seq_len), dtype=int)
    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, :len(review)] = np.array(review)[:seq_len]
    return features


class CustomDataset(Dataset):
    def __init__(self, sentences, positions, seq_len_vals):
        self.sentences = sentences
        self.positions = positions
        self.seq_len_values = seq_len_vals

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx], self.positions[idx], self.seq_len_values[idx]


def predict(model, data_loader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for x, pos, real_seq_len in data_loader:
            x = torch.squeeze(x)[:real_seq_len].to(device)
            pos = torch.squeeze(pos)[:real_seq_len].to(device)
            real_seq_len = torch.squeeze(real_seq_len).to(device)
            sample_score_matrix = model(padded_sentence=x,
                                        padded_pos=pos,
                                        real_seq_len=real_seq_len)

            mst, _ = decode_mst(sample_score_matrix.detach().cpu().numpy(), sample_score_matrix.shape[0],
                                has_labels=False)
            predictions.append(mst.tolist())
    return predictions


def write_file(file_address, predictions):
    predictions_counter = 0
    all_sentences = []
    new_sentence = []
    with open(file_address, encoding='utf-8') as f:
        for idx, row in enumerate(f):
            if row != '\n':
                new_sentence.append(row)
            else:
                all_sentences.append([new_sentence, predictions[predictions_counter]])
                new_sentence = []
                predictions_counter += 1

    with open('./data/comp_318295029_206230021.labeled', 'w') as f:
        for sen, pred in all_sentences:
            for row, word_pred in zip(sen, pred[1:]):
                row_lst = row.split('\t')
                row_lst[6] = word_pred
                row_string = ''
                for idx, x in enumerate(row_lst):
                    if idx == len(row_lst) - 1:
                        row_string = row_string + x
                    else:
                        row_string = row_string + str(x) + '\t'

                f.write(row_string)
                row_string = ''

            f.write('\n')


def main():
    train_dataset = Dataset.from_dict(load_ds_to_dict("/archive/data/val.labeled"))

    model = torch.load('comp_model_mlp_ex3').to('cuda')
    sentences_word2idx = model.sentences_word2idx
    pos_word2idx = model.pos_word2idx
    max_sen_len = 250
    generated_samples = []
    for counter, value in enumerate(train_dataset['translation']):
        if counter % 1000 == 0:
            print(f'{counter}/{len(train_dataset["translation"])}')
        ger_paragraph = value['de']
        eg_paragraph = value['en']
        eg_paragraph_temp = eg_paragraph.replace('\n', '')
        en_sentences = splits_paragraph(eg_paragraph_temp)
        split_en_paragraph_list = []
        for sen in en_sentences:
            words = sen.split(' ')
            words.insert(0, 'ROOT')

            embedded_words = [sentences_word2idx[word] if word in sentences_word2idx else 1 for word in words]
            sen_positions = [0] * len(words)
            sentence_real_len = len(words)

            model.eval()
            with torch.no_grad():
                x = torch.Tensor(embedded_words)
                x = x.int()
                x = x.to('cuda')
                pos = torch.Tensor(sen_positions)
                pos = pos.int()
                pos = pos.to('cuda')
                # real_seq_len = torch.Tensor(sentence_real_len).to('cuda')
                sample_score_matrix = model(padded_sentence=x,
                                            padded_pos=pos,
                                            real_seq_len=sentence_real_len)

                mst, _ = decode_mst(sample_score_matrix.detach().cpu().numpy(), sample_score_matrix.shape[0],
                                    has_labels=False)

            split_en_paragraph_list.append((words, mst.tolist()))

        paragraphs_dict = {'en': eg_paragraph, 'de': ger_paragraph, 'parsing_tree': split_en_paragraph_list}
        generated_samples.append(paragraphs_dict)

    with open("/archive/data/val_dependency_parsed.json", "w") as outfile:
        outfile.write(json.dumps(generated_samples, indent=4))


if __name__ == '__main__':
    main()
