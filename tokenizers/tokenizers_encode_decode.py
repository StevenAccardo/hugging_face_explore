from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

sequence = 'Using a Transformer network is simple'
tokens = tokenizer.tokenize(sequence)

print(tokens)

# ['Using', 'a', 'transform', '##er', 'network', 'is', 'simple']

ids = tokenizer.convert_tokens_to_ids(tokens)

print(ids)

# [7993, 170, 11303, 1200, 2443, 1110, 3014]

decoded_string = tokenizer.decode([7993, 170, 11303, 1200, 2443, 1110, 3014])
print(decoded_string)

# 'Using a Transformer network is simple'