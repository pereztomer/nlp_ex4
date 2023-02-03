from transformers import BertModel, BertTokenizer, AutoTokenizer, T5Model

from torch import nn


class CustomModel(nn.Module):
    def __init__(self, classes):
        super(CustomModel, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.out = nn.Linear(self.bert.config.hidden_size, classes)

    def forward(self, input):
        output = self.bert(input)
        pooler_output = output['pooler_output']
        last_hidden_state = output['last_hidden_state']
        exit()
        # out = self.out(output)
        # return out


def main():
    model = T5Model.from_pretrained('t5-base')
    model.encoder
    model.decoder
    # model = CustomModel(classes=5)
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # text = "What is the capital of France?"
    # inputs = tokenizer.encode(text,
    #                           return_tensors='pt',
    #                           add_special_tokens=True,
    #                           truncation=True,
    #                           padding="max_length")
    # print(inputs)
    # start, end = model(inputs)


if __name__ == '__main__':
    main()
