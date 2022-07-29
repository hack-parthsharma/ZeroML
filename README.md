 

<div align="center">    
 
# broNet
  
</div>
 
## Description   
A neural net that tells whether a bit-vector has consecutive 1’s

## How to run   
First, install dependencies   
```bash
# clone project   
git clone https://github.com/hack-parthsharma/ZeroMl

# install project 
cd ZeroMl
pip install -r requirements.txt
 ```   
 Next, navigate to any file and run it.   
 ```bash
# module folder
cd ZeroMl

# run module 
python xor.py    
```

## Imports
This project is setup as a package which means you can now easily import any file into any other file like so:
```python
from broNet.train import train
from broNet.nn import NeuralNet
from broNet.layers import Linear, Tanh


# data
inputs = np.array([
    [0,0],
    [1,0],
    [0,1],
    [1,1]
])
targets = np.array([
    [1,0],
    [0,1],
    [0,1],
    [1,0]
])

# train
net = NeuralNet([
    Linear(input_size=2, output_size=2)
])
train(net, inputs, targets)

# test using the neuralnet!
for x, y in zip(inputs, targets):
    predicted = net.forward(x)
    print(x, predicted, y)
```


