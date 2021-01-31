from pytorch_pretrained import BertModel, BertTokenizer
import torch

tokenizer = BertTokenizer.from_pretrained('ERNIE_pretrain')
bert = BertModel.from_pretrained('ERNIE_pretrain')

token = tokenizer.tokenize('今天天气很好')  # 切成字
token = ['[CLS]']+token
print(token)
token_ids = tokenizer.convert_tokens_to_ids(token)
print(token_ids)
token_ids = torch.LongTensor([token_ids])
torch.LongTensor(token_ids)
encoder_out, pooled = bert(token_ids, attention_mask=torch.LongTensor([[1]*7]), output_all_encoded_layers=False)
print(pooled)

# import transformers
# transformers.DataCollatorForLanguageModeling

