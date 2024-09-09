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

  net.train([input], [output], 0.01, 5);
  print(net._forward(input).toString());
  print(output.toString());
}
```
### Example output
The example below shows an example output, given the aove code. It makes a prediction which in this case is [0, 1] and is correct.
```
beginning training
epoch 1: 0.2507944604037083
epoch 2: 0.25284205440405766
epoch 3: 0.25590826696030605
epoch 4: 0.2597775694935288
epoch 5: 0.26426020078006657
[0.3805839174145019, 0.6194160825854982] 

[0.0, 1.0] 
```

