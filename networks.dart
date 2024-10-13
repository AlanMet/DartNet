import 'matrices.dart';
import 'dart:math';

class Batch {
  List<Matrix> inputs;
  List<Matrix> outputs;
  Batch(this.inputs, this.outputs);
  int get length => inputs.length;

  void train(Network net, double lr, {double dropout = 0.2}) {
    List<Matrix> losses = [];
    for (var i = 0; i < length; i++) {
      net._forward(inputs[i], dropout: dropout);
      Matrix loss = net.lossFunction(net._activated.last, outputs[i]);
      losses.add(loss);
    }
    Matrix avgLoss = losses[0];
    for (var i = 1; i < length - 1; i++) {
      avgLoss += losses[i];
    }
    avgLoss /= length;
    net._backward(avgLoss);
    net._update(lr);
  }
}

class Network {
  int? seed;
  late List<int> _architecture;
  late List<Matrix Function(Matrix)> _activations;
  List<Matrix> _weights = [],
      _biases = [],
      _preActivated = [],
      _activated = [],
      _gradw = [],
      _gradb = [],
      _deltas = [];

  late Matrix Function(Matrix, Matrix) lossFunction;

  /// Create a new network
  /// - [architecture] The architecture of the network
  /// - [activations] The activation functions of the network
  /// - [seed] The seed for the random number generator
  /// - Returns The network
  /// - Example
  /// ```dart
  /// Network net = Network([1, 2, 2], [relu, softmax]);
  /// ```
  /// - Note
  /// The length of the architecture list should be equal to the length of the activations list minus one
  Network(List<int> architecture, List<Matrix Function(Matrix)> activations) {
    _architecture = architecture;
    _activations = activations;

    for (int x = 0; x < _architecture.length - 1; x++) {
      _weights.add(randn(_architecture[x], _architecture[x + 1]));
      _biases.add(zeros(1, _architecture[x + 1]));
    }

    lossFunction = mseDerivative;
  }

  Matrix XavierInit(int input, int ouput) {
    double limit = sqrt(6 / (input + ouput));
    return randn(input, ouput, start: -limit, end: limit);
  }

  /// Returns an independent copy of the network
  Network clone() {
    Network newNet = Network(_architecture, _activations);
    newNet._weights = _weights;
    newNet._biases = _biases;
    return newNet;
  }

  Matrix _activation(Matrix input, Matrix Function(Matrix) function) {
    return function(input);
  }

  Matrix _activationDeriv(Matrix input, Matrix Function(Matrix) function) {
    return derivative(function)(input);
  }

  Matrix _forward(Matrix input, {double dropout = 0.0}) {
    _preActivated = [];
    _activated = [];

    _preActivated.add(input);
    _activated.add(input);

    for (var i = 0; i < _architecture.length - 1; i++) {
      _preActivated.add(dot(_activated[i], _weights[i]) + _biases[i]);
      _activated.add(
          _activation(_preActivated[i + 1], _activations[i]).Dropout(dropout));
    }

    //print(_activated[_activated.length - 1].toString());
    return _activated[_activated.length - 1];
  }

  /// Predict the output of the network
  /// - [x] The input data
  /// - Returns The output of the network
  Matrix predict(Matrix x, {double dropout = 0.0}) {
    return _forward(x, dropout: dropout);
  }

  void setLoss(Matrix Function(Matrix x, Matrix y) function) {
    lossFunction = function;
  }

  double mse(Matrix x, Matrix y) {
    return mean(power(predict(x), 2));
  }

  Matrix mseDerivative(Matrix x, Matrix y) {
    return x - y;
  }

  double crossEntropy(Matrix x, Matrix y) {
    y.performFunction((value) => log(value));
    return mean((x * -1) * y);
  }

  Matrix crossEntropyDerivative(Matrix x, Matrix y) {
    return y - x;
  }

  double Function(Matrix x, Matrix y) getLossFunction() {
    if (lossFunction == mseDerivative) {
      return mse;
    } else if (lossFunction == crossEntropyDerivative) {
      return crossEntropy;
    } else {
      return mse;
    }
  }

  void _backward(Matrix loss) {
    _gradw = [];
    _gradb = [];
    _deltas = [];

    //dC/dz
    _deltas.add(loss);

    //dC/dz*dz/dw
    _gradw
        .add(dot(_activated[_activated.length - 2].transpose(), _deltas.last));

    //sum dC/dz
    _gradb.add(sum(_deltas.last, 0));

    for (var i = _architecture.length - 2; i > 0; i--) {
      //dc/dz * dz/da * da/dz
      _deltas.add(dot(_deltas.last, _weights[i].transpose()) *
          _activationDeriv(_preActivated[i], _activations[i]));
      //dC/dz * dz/da * da/dz * dz/dw
      _gradw.add(dot(_activated[i - 1].transpose(), _deltas.last));
      _gradb.add(sum(_deltas.last, 0));
    }

    _gradw = _gradw.reversed.toList();
    _gradb = _gradb.reversed.toList();
    _deltas = _deltas.reversed.toList();
  }

  /// Update the weights and biases of the network
  /// - [lr] The learning rate
  /// - Returns None
  /// to do : add other optimizers
  void _update(double lr) {
    _gradientDescent(lr);
  }

  /// Update the weights and biases of the network
  /// - [lr] The learning rate
  /// - Returns None
  void _gradientDescent(double lr) {
    for (var i = 0; i < _architecture.length - 1; i++) {
      _weights[i] -= _gradw[i] * lr;
      _biases[i] -= _gradb[i] * lr;
    }
  }

  /// Train the network
  /// - [inputs] The input data
  /// - [expected] The expected outputs
  /// - [lr] The learning rate
  /// - [epochs] The number of epochs
  void train(List<Matrix> inputs, List<Matrix> expected, double lr, int epochs,
      {double dropout = 0.2, bool verbose = true}) {
    print("beginning training");
    for (var i = 0; i < epochs; i++) {
      for (var x = 0; x < inputs.length; x++) {
        _forward(inputs[x], dropout: dropout);
        Matrix loss =
            lossFunction(_activated.last.clip(1e-6, 1e10), expected[x]);
        _backward(loss);
        _update(lr);
        if (verbose) {
          print(
              "epoch ${i + 1}: ${getLossFunction()(_activated.last, expected[i])}");
        }
      }
    }
  }
}

void main() {
  Network net = Network([1, 2, 10], [relu, softmax]);

  Matrix input = randn(1, 1);
  Matrix output = fill(0, 1, 10);
  output.setAt(0, 1, value: 1);

  net.train([input], [output], 0.1, 1000);
  print(net._forward(input).toString());
  print(output.toString());
}
