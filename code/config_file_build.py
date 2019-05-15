import json
import os


os.chdir('../configs')
# batch sizes: 100, 200, 300, 400
# epochs: 20, 50, 150, 300
# learning rates: 0.00001, 0.0001, 0.001, 0.01
# optimizers: 'adam', 'rmsprop', 'sgd'
# hidden layers: 3, 4, 5, 6, 7
# dropout: 0.2, 0.4, 0.6, 0.8
# hidden activations: sigmoid, relu, tanh
# output activations: sigmoid, relu, tanh, linear
# scaling: 0.5, 0.6
params = {'batch_size': 300, 
'epochs': 50,
'learning_rate': 0.001,
'optimizer': 'adam',
'hidden_layers': 2,
'dropout': 0.2,
'hidden_act': 'sigmoid',
'output_act': 'relu',
'scaling': 0.5
}

with open('300_50_001_adam_2_02_sigmoid_relu_05.json', 'w') as json_file:
    json.dump(params, json_file)