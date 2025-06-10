from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import torch

# Checkpoints are the saved states of the underlying model
# They include the:
# 1. Model architecture config (e.g., number of layers, hidden size)
# 2. Model weights (learned parameters, usually hundreds of millions of values)
# 3. Tokenizer files (vocabulary, special tokens, etc.)
# 4. (Optional) Training state if saved during fine-tuning — includes optimizer state, learning rate, etc.
checkpoint = 'distilbert-base-uncased-finetuned-sst-2-english'
# This downloads the checkpoint info from the hugging face model hub
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

raw_inputs = [
    'I\'ve been waiting for a HuggingFace course my whole life.',
    'I hate this so much!',
]

# pass the raw_input text to our tokenizer
# Tell it to return our tensors for pytorch vs tensorflow or another library
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors='pt')
print(inputs)

# {
#     'input_ids': tensor([
#         [  101,  1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172, 2607,  2026,  2878,  2166,  1012,   102],
#         [  101,  1045,  5223,  2023,  2061,  2172,   999,   102,     0,     0,     0,     0,     0,     0,     0,     0]
#     ]), 
#     'attention_mask': tensor([
#         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
#         [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
#     ])
# }


checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
# Downloads the model data just like we did with the transformer above
# Ou
model = AutoModel.from_pretrained(checkpoint)
# accepts the dictionary that holds the tokens for each word
outputs = model(**inputs)
print(outputs.last_hidden_state.shape)

# Returns a 3d tensor
# batch size, sequence length, hidden size
# torch.Size([2, 16, 768])


checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
# The type of task that a model handles relies heavily on the head that is used
# Here we are assigning a head to the be used on the model output data
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(**inputs)

# Output dimensions are much less
# Output is logits
# Logits are the the raw, unnormalized scores outputted by the last layer of the model.
# Has 2 values and 2 lables to match the 2 sequences as input
print(outputs.logits.shape)
# torch.Size([2, 2])

print(outputs.logits)
# tensor(
#     [
#         [-1.5607,  1.6123],
#         [ 4.1692, -3.3464]
#     ],
#     grad_fn=<AddmmBackward>
# )

predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)

# These are probability scores for negative and positive, but first we need to find which is which from the model config
# tensor(
#     [
#         [4.0195e-02, 9.5980e-01],
#         [9.9946e-01, 5.4418e-04]
#     ],
#     grad_fn=<SoftmaxBackward>
# )

print(model.config.id2label)
# first index is neg, 2nd is pos
# {0: 'NEGATIVE', 1: 'POSITIVE'}

# First sentence: NEGATIVE: 0.0402, POSITIVE: 0.9598
# Second sentence: NEGATIVE: 0.9995, POSITIVE: 0.0005