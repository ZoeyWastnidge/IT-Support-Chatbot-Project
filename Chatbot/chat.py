import random
import json
from re import X
import torch
from model import NeuralNet
from nltkWork import bagOfWords, tokenise

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')           #Uses CUDA to use the GPU for processing, defaults to the CPU otherwise

with open('messageIntent.json', 'r') as file:
    intent = json.load(file)

FILE = "data.pth"
data = torch.load(FILE)

inputSize = data["inputSize"]
hiddenSize = data["hiddenSize"]
outputSize = data["outputSize"]
allWords = data["allWords"]
tags = data["tags"]
modelState = data["modelState"]

model = NeuralNet(inputSize, hiddenSize, outputSize).to(device)                 #Creates the neural network from the file generated from training
model.load_state_dict(modelState)
model.eval()

#Actual chat begins

botName = "Athena"
print("Welcome to Athena, the technical support chatbot. No need to be formal, there is no need to use punctuation or apostrophes (I may not be able to understand you!) If you would like to exit, please type 'quit'")
while True:
    sentence = input('You: ')
    if sentence == "quit":
        break

    sentence = tokenise(sentence)                                               #Splits the sentence into tokens
    X = bagOfWords(sentence, allWords)                                          #Uses RegEx expression and creates the bag of words from the sentence and allWords from data.pth
    X = X.reshape(1, X.shape[0])                                                #Returns an array with the input in the specified shape of X with direction of 1 for use later
    X = torch.from_numpy(X).to(device)                                          #Converts the numpy array to a pytorch tensor on the GPU or CPU

    output = model(X)
    _, predicted = torch.max(output, dim=1)                                     #Returns max elements along the dimension 1
    tag = tags[predicted.item()]                                                #Finds the predicted tag in the training data

    probs = torch.softmax(output, dim=1)                                        #The final layer of the neural net, predicts probability of the tag predicted being correct
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:                                                      #If the probability score is high enough, iterates through the training data to find data with the predicted tag and formulates a response
        for inte in intent["messageIntent"]:
            if tag == inte["tag"]:
                print(f"{botName}: {random.choice(inte['responses'])}")
    else:
        print(f"{botName}: I do not understand. Please try again.")             #Appends unknown data to text file
        textFile = open("UnknownInputs.txt", "a")
        textFile.write(str(sentence))
        textFile.write("\n")
        textFile.close

