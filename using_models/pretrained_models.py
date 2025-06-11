from transformers import BertModel
import torch

# This model is now initialized with all the weights of the checkpoint.
# It can be used directly for inference on the tasks it was trained on, and it can also be fine-tuned on a new task.
# By training with pretrained weights rather than from scratch, we can quickly achieve good results.
# Weights are cached on your local machine after the first download.
model = BertModel.from_pretrained('bert-base-cased')

# This saves the state of the model, even one you have been fine-tuning
# Results in 2 files:
# config.json which holds all the config data for the underlying model
# pytorch_model.bin aka the state dictionary which holds the models weights
model.save_pretrained('directory_on_my_computer')

sequences = ['Hello!', 'Cool.', 'Nice!']

# If we were using a tokenizer this would be the output of the tokenizer
# These are just for example right now.
encoded_sequences = [
    [101, 7592, 999, 102],
    [101, 4658, 1012, 102],
    [101, 3835, 999, 102],
]

model_inputs = torch.tensor(encoded_sequences)

output = model(model_inputs)

print(output)