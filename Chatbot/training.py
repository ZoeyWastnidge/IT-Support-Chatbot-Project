import json
from nltkWork import tokenise, stem, bagOfWords
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, dataloader, dataset
from model import NeuralNet

with open('messageIntent.json', 'r') as file:
    messageIntent = json.load(file)

allWords = []
tags = []
xy = []

for intent in messageIntent['messageIntent']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        p = tokenise(pattern)
        allWords.extend(p)
        xy.append((p, tag))

ignoreWords = ['?', '!', '.', ',']
allWords = [stem(p) for p in allWords if p not in ignoreWords]
allWords = sorted(set(allWords))
tags = sorted(set(tags))

xTrain = []
yTrain = []
for (pSentence, tag) in xy:
    bag = bagOfWords(pSentence, allWords)
    xTrain.append(bag)

    label = tags.index(tag)
    yTrain.append(label)

xTrain = np.array(xTrain)
yTrain = np.array(yTrain)

class ConvoDataset(Dataset):
    def __init__(self):
        self.nSamples = len(xTrain)
        self.xData = xTrain
        self.yData = yTrain
    
    def __getitem__(self, index):
        return self.xData[index], self.yData[index]
    
    def __len__(self):
        return self.nSamples


#Hyperparameters
batchSize = 8
inputSize = len(xTrain[0])
outputSize = len(tags)
hiddenSize = 8
learningRate = 0.001
numEpochs = 1000

dataset = ConvoDataset()
trainLoader = DataLoader(dataset=dataset, batch_size=batchSize, shuffle=True, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(inputSize, hiddenSize, outputSize).to(device)


cross = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(model.parameters(), lr=learningRate)

#training loop
for epoch in range(numEpochs):
    for (words, labels) in trainLoader:
        words = words.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.long)

        outputs = model(words)
        loss = cross(outputs, labels)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    if (epoch +1) % 100 == 0:
        print(f'epoch {epoch+1}/{numEpochs}, loss={loss.item():.4f}')

print(f'final loss, loss={loss.item():.4f}')

data = {
    "modelState": model.state_dict(),
    "inputSize": inputSize,
    "outputSize": outputSize,
    "hiddenSize": hiddenSize,
    "allWords": allWords,
    "tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file has been saved to {FILE}')
