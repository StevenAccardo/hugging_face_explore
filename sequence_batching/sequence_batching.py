import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Pick the checkpoint we want to use
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
# Download the tokenizer config and vocab
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# Download the model config and the weights with a sequence classification head placed on the model
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."

# Create tokens
tokens = tokenizer.tokenize(sequence)

# Convert the tokens to input_ids using the pretrained model vocab
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)

input_ids = torch.tensor([ids])
print("Input IDs:", input_ids)
# Input IDs: [[ 1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172,  2607, 2026,  2878,  2166,  1012]]

output = model(input_ids)
print("Logits:", output.logits)
# Logits: [[-2.7276,  2.8789]]

# -----------------Using padding-----------------------------------------------------------

# Mocked input ids
sequence1_ids = [[200, 200, 200]]
sequence2_ids = [[200, 200]]

# Manually adds the padding token id 
batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id],
]

print(model(torch.tensor(sequence1_ids)).logits)
# [[ 1.5694, -1.3895]]

print(model(torch.tensor(sequence2_ids)).logits)
# [[ 0.5803, -0.4125]]

# Logits don't match for the 2nd sequence
print(model(torch.tensor(batched_ids)).logits)
# [[ 1.5694, -1.3895], [ 1.3373, -1.2163]]

# -----------------Using attention masks-----------------------------------------------------------

batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id],
]

# add attention mask to tell the model to ignore the padding tokens
attention_mask = [
    [1, 1, 1],
    [1, 1, 0],
]


outputs = model(torch.tensor(batched_ids), attention_mask=torch.tensor(attention_mask))
print(outputs.logits)
[[ 1.5694, -1.3895], [ 0.5803, -0.4125]]