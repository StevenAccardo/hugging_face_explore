from transformers import BertTokenizer, AutoTokenizer

# Can use specific tokenizers or use the AutoTokenizer which will look up the configs for teh checkpoint that you pass it
# tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

results = tokenizer('Using a Transformer network is simple')

print(results)

{
  'input_ids': [101, 7993, 170, 11303, 1200, 2443, 1110, 3014, 102],
  'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0],
  'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1]
}

tokenizer.save_pretrained("directory_on_my_computer")