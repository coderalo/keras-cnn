from ResNet import ResNet
from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np
import argparse
import json

parser = argparse.ArgumentParser(description='Process the data and parameters')
# LOG, MODEL 
parser.add_argument('--model_type', default="ResNet", help='the type of model [ResNet]')
parser.add_argument('--log_path', default="train.log", help='the path of log [train.log]')
parser.add_argument('--model_path', default="model.hdf5", help='the path of model [model.hdf5]')
parser.add_argument('--ans_path', default="answer.csv", help='the path of predict result [answer.csv]')
# TRAINING PARAMETERS
parser.add_argument('--learning_rate', type=float, default=0.1, help='the learning rate [0.1]')
parser.add_argument('--decay_rate', type=float, default=0.9, help='the decay rate of learning rate [0.9]')
parser.add_argument('--batch_size', type=int, default=32, help='the batch size [32]')
parser.add_argument('--nb_epoch', type=int, default=50, help='the number of training epochs [50]')
parser.add_argument('--momentum', type=float, default=0.9, help='the momentum rate [0.9]')
parser.add_argument('--val_split', type=float, default=0.2, help='the split of validation data [0.2]')
parser.add_argument('--model_data', default="model.json", help='the path of hyperparameters of the network [model.json]')
args = parser.parse_args()

with open(args.model_data, 'r') as file: model_data = json.load(file)
if args.model_type == "ResNet": net = ResNet(model_data)

(train_X, train_y), (test_X, test_y) = mnist.load_data()
train_X, test_X = np.expand_dims(train_X, 3), np.expand_dims(test_X, 3)
train_y = to_categorical(train_y)
net.train(
        images=train_X, 
        labels=train_y, 
        nb_epoch=args.nb_epoch,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        decay=args.decay_rate,
        momentum=args.momentum,
        val_split=args.val_split,
        log_path=args.log_path,
        model_path=args.model_path)

prediction, tokens = net.test(
        images=test_X,
        batch_size=args.batch_size,
        model_path=args.model_path)

cnt = 0
for idx, token in enumerate(tokens):
    if token == test_y[idx]: cnt += 1

print("Accuracy: {}%".format(float(cnt) / len(tokens) * 100))

with open(args.ans_path, 'w') as file:
    file.write("id,val\n")
    for idx, token in enumerate(tokens):
        file.write("{},{}\n".format(idx, token))
