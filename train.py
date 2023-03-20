import json
from nltk_utils import stem, bagOfWords, tokenize, removeStopWords
import numpy as np
from model import NeuralNet


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

with open('intents.json','r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:  
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))
        
all_words = [stem(w) for w in all_words]
all_words = removeStopWords(all_words)
all_words = sorted(set(all_words))
tags = sorted(set(tags))

# print('all_words :',all_words)
# print('tags :',tags)

X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    bag = bagOfWords(pattern_sentence, all_words)
    X_train.append(bag)

    label = tags.index(tag)
    y_train.append(label) #crossEntropy - so no one hotenvcodgin

X_train = np.array(X_train)
y_train = np.array(y_train)

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train
    
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.n_samples


# Hyperparameters
batch_size = 1
hidden_size = 8
output_size = len(tags)
input_size = len(X_train[0])
learning_rate = 0.001
num_epochs = 1000


print(input_size, len(all_words))
print(output_size, tags)

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, 
                            batch_size = batch_size,
                            shuffle=True,
                            num_workers=0)

device = 'cpu'
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        model = model.to(device)

        # Forward pass
        outputs = model(words)
        # if y would be one-hot, we must apply
        # labels = torch.max(labels, 1)[1]
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


print(f'final loss: {loss.item():.4f}')

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')