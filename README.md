# Dart Neural Network Library

This project is a simple neural network implemented in dart, using the matrix library I also created in dart.

## Features
- Multi-layered neural network
- support for multiple Activation functions (ReLU, Sigmoid, Softmax, tanH)
- Capability of using other custom activations
# Usage
* Create a neural network
```
Network net = Network([1, 2, 2], [relu, softmax]);
```
* Train the network
```
Matrix input = randn(1, 1); //Example input
Matrix output = fill(0, 1, 2); //Example output
output.setAt(0, 1, value: 1); 
net.train([input], [output], 0.01, 1000); 
```
* Make predictions
```
Matrix result = net.predict(input);
print(result.toString());
```
## Example
```
void main() {
  Network net = Network([1, 2, 2], [relu, softmax]);

  Matrix input = randn(1, 1);
  Matrix output = fill(0, 1, 2);
  output.setAt(0, 1, value: 1);

  net.train([input], [output], 0.01, 1000);
  print(net._forward(input).toString());
  print(output.toString());
}
```
### Example output
The example below shows an example output, given the aove code. It makes a prediction which in this case is [0, 1] and is correct.
```
beginning training
epoch 0: 0.2510939097374368
epoch 1: 0.2533151884930789
epoch 2: 0.25649031497900676
epoch 3: 0.26041589978481183
epoch 4: 0.26491276849033973
[0.37788215326849345, 0.6221178467315066]

[0.0, 1.0] 

```

