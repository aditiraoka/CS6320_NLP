import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import math
import random
import os
import time
from tqdm import tqdm
import json
import string
from argparse import ArgumentParser
import pickle
import matplotlib.pyplot as plt

unk = '<UNK>'
# Consult the PyTorch documentation for information on the functions used below:
# https://pytorch.org/docs/stable/torch.html
class RNN(nn.Module):
    def __init__(self, input_dim, h):  # Add relevant parameters
        super(RNN, self).__init__()
        self.h = h
        self.numOfLayer = 1
        self.rnn = nn.RNN(input_dim, h, self.numOfLayer, nonlinearity='tanh')
        self.W = nn.Linear(h, 5)
        self.softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, inputs):
        # [to fill] obtain hidden layer representation (https://pytorch.org/docs/stable/generated/torch.nn.RNN.html)
        _, hidden = self.rnn(inputs)

        # [to fill] obtain output layer representations
        outLayer = self.W(hidden)

        # [to fill] sum over output 
        sumOutLayer = torch.sum(outLayer, dim=1)

        # [to fill] obtain probability dist.
        predicted_vector = self.softmax(sumOutLayer)

        return predicted_vector


def load_data(train_data, val_data, test_data):
    with open(train_data) as training_f:
        training = json.load(training_f)
    with open(val_data) as valid_f:
        validation = json.load(valid_f)
    with open(test_data) as test_f:
        testing = json.load(test_f)
 
    tra = []
    val = []
    test = []
    for elt in training:
        tra.append((elt["text"].split(),int(elt["stars"]-1)))
    for elt in validation:
        val.append((elt["text"].split(),int(elt["stars"]-1)))
    for elt in testing:
        test.append((elt["text"].split(),int(elt["stars"]-1)))
    return tra, val, test


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required = True, help = "hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, required = True, help = "num of epochs to train")
    parser.add_argument("--train_data", required = True, help = "path to training data")
    parser.add_argument("--val_data", required = True, help = "path to validation data")
    parser.add_argument("--test_data", default = "to fill", help = "path to test data")
    parser.add_argument("--out_path", default = "./", help = "path to store output data")
    parser.add_argument('--do_train', action='store_true')
    args = parser.parse_args()

    print("========== Loading data ==========")
    train_data, valid_data, test_data = load_data(args.train_data, args.val_data, args.test_data) 
    # X_data is a list of pairs (document, y); y in {0,1,2,3,4}

    # Think about the type of function that an RNN describes. To apply it, you will need to convert the text data into vector representations.
    # Further, think about where the vectors will come from. There are 3 reasonable choices:
    # 1) Randomly assign the input to vectors and learn better embeddings during training; see the PyTorch documentation for guidance
    # 2) Assign the input to vectors using pretrained word embeddings. We recommend any of {Word2Vec, GloVe, FastText}. Then, you do not train/update these embeddings.
    # 3) You do the same as 2) but you train (this is called fine-tuning) the pretrained embeddings further.
    # Option 3 will be the most time consuming, so we do not recommend starting with this

    print("========== Vectorizing data ==========")
    model = RNN(50, args.hidden_dim)  # Fill in parameters
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    word_embedding = pickle.load(open('/content/drive/MyDrive/finalDraft/rnn/word_embedding.pkl', 'rb'))

    stopping_condition = False
    epoch = 0

    last_train_accuracy = 0
    last_validation_accuracy = 0
    trainingResults = {}
    valResults = {}
    trainingLoss = []
    valAccuracy = []
    epochList = []

    while not stopping_condition:
        random.shuffle(train_data)
        model.train()
        # You will need further code to operationalize training, ffnn.py may be helpful
        print("Training started for epoch {}".format(epoch + 1))
        epochList.append(epoch+1)
        train_data = train_data
        correct = 0
        total = 0
        minibatch_size = 16
        N = len(train_data)

        loss_total = 0
        loss_count = 0
        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                input_words, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                input_words = " ".join(input_words)

                # Remove punctuation
                input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()

                # Look up word embedding dictionary
                vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk'] for i in input_words ]

                # Transform the input into required shape
                vectors = np.array(vectors)
                vectors = torch.tensor(vectors).view(len(vectors), 1, -1)
                output = model(vectors)

                # Get loss
                example_loss = model.compute_Loss(output.view(1,-1), torch.tensor([gold_label]))

                # Get predicted label
                predicted_label = torch.argmax(output)

                correct += int(predicted_label == gold_label)
                # print(predicted_label, gold_label)
                total += 1
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss

            loss = loss / minibatch_size
            loss_total += loss.data
            loss_count += 1
            loss.backward()
            optimizer.step()
        print(loss_total/loss_count)
        print("Training completed for epoch {}".format(epoch + 1))
        print("Training accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        trainning_accuracy = correct/total
        
        trainingLoss.append(loss_total/loss_count)
        result = {
            #"hidden_dim": args.hidden_dim,
            "epoch": epoch + 1,
            "loss" : loss.item(),
            "accuracy": correct / total,

        }
        if args.hidden_dim not in trainingResults.keys():
            trainingResults[args.hidden_dim]=[epoch+1, result]
        else:
            trainingResults[args.hidden_dim].append([epoch+1, result])


        model.eval()
        correct = 0
        total = 0
        random.shuffle(valid_data)
        print("Validation started for epoch {}".format(epoch + 1))
        valid_data = valid_data

        for input_words, gold_label in tqdm(valid_data):
            input_words = " ".join(input_words)
            input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()
            vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk'] for i
                       in input_words]

            vectors = np.array(vectors)
            vectors = torch.tensor(vectors).view(len(vectors), 1, -1)
            output = model(vectors)
            predicted_label = torch.argmax(output)
            correct += int(predicted_label == gold_label)
            total += 1
            # print(predicted_label, gold_label)
        print("Validation completed for epoch {}".format(epoch + 1))
        print("Validation accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        validation_accuracy = correct/total

        if validation_accuracy < last_validation_accuracy and trainning_accuracy > last_train_accuracy:
            stopping_condition=True
            print("\nTraining done to avoid overfitting!")
            print("Best validation accuracy is:", last_validation_accuracy)
            '''
            results = []
            # You may want to save some additional information along with the accuracy, such as model settings or other metrics.
            results.append({
                "accuracy": last_validation_accuracy,
                "epoch": epoch,
                "hidden_dim": args.hidden_dim,
                 # Add more information as needed
            })
        
            with open("RNNout.txt", "a+") as outfile:
                json.dump(results, outfile, indent=2)
            '''
        else:
            last_validation_accuracy = validation_accuracy
            last_train_accuracy = trainning_accuracy

        epoch += 1
        tempResult = {
            "epoch": epoch,
            "accuracy": last_validation_accuracy,
            "hidden_dim": args.hidden_dim,
            #"loss" : loss.item(),
            #"time taken": timeTaken,
        }
        
        valAccuracy.append(last_validation_accuracy)
        #valResults.append({args.hidden_dim:[epoch, tempResult]})
        if args.hidden_dim not in valResults.keys():
            valResults[args.hidden_dim]=[epoch+1, tempResult]
        else:
            valResults[args.hidden_dim].append([epoch+1, tempResult])
            
    model.eval()
    
    test_correct = 0
    test_total = 0
    print("========== Test data ==========")
    with torch.no_grad():
        for input_words, gold_label in tqdm(test_data):
            input_words = " ".join(input_words)
            input_words = input_words.translate(input_words.maketrans("","", string.punctuation)).split()
            test_vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk'] for i in input_words]
            test_vectors = np.array(test_vectors)
            test_vectors = torch.tensor(test_vectors).view(len(test_vectors), 1, -1)
            test_output = model(test_vectors)
            test_label = torch.argmax(test_output)
            test_correct += int(test_label == gold_label)
            test_total += 1
    
    test_accuracy = test_correct / test_total
    print(f"Test Accuracy for the model: {test_accuracy}")
    
    f=args.out_path+"Results_"+ str(args.hidden_dim)+ "_" + str(args.epochs)+".txt"
    with open(f, "a+") as outfile:
        json.dump(["Training Results:"], outfile, indent=2)
        json.dump(trainingResults, outfile, indent=2)
        json.dump(["Validation Results:"], outfile, indent=2)
        json.dump(valResults, outfile, indent=2)
        json.dump(["Testing Results:"], outfile, indent=2)
        json.dump([test_accuracy], outfile, indent=2)

    #with open(args.out_path+"validationResults_"+str(args.hidden_dim)+"_"+str(args.epochs)+".txt", "a+") as outfile:
    #    json.dump(valResults, outfile, indent=2)
    
    #print(type(trainingLoss))
    #print(type(epochList))

    try:
        plt.plot(epochList, trainingLoss, label='Training Loss')
        #plt.title('Training Loss')
        plt.xlabel('epochs')
        plt.ylabel('training_loss')
        #plt.show()
        # Save the plot as a figure
        #plt.savefig(args.out_path+"Training Loss_"+str(args.hidden_dim)+"_"+str(args.epochs)+'.png')
    except Exception as e:
        print(str(e))

    try:
        #plt.clf()
        plt.plot(epochList, valAccuracy, color='yellow', label='val_accuracy Loss')
        plt.title('Learning Curve for RNN')
        #plt.xlabel('epochs')
        #plt.ylabel('val_accuracy')
        # Save the plot as a figure
        plt.savefig(args.out_path+"LearningCurve_"+str(args.hidden_dim)+"_"+str(args.epochs)+'.png')
    except Exception as e:
        print(str(e))

    

    # You may find it beneficial to keep track of training accuracy or training loss;

    # Think about how to update the model and what this entails. Consider ffnn.py and the PyTorch documentation for guidance
