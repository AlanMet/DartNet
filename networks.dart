import 'matrices.dart';
import 'dart:math';
import 'dart:convert';
import 'dart:io';

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

  late List<Matrix> _mW; // Moving averages for weights
  late List<Matrix> _vW; // Squared gradients for weights
  late List<Matrix> _mB; // Moving averages for biases
  late List<Matrix> _vB; // Squared gradients for biases
  double _beta1 = 0.9; // Exponential decay rate for the first moment
  double _beta2 = 0.999; // Exponential decay rate for the second moment
  double _epsilon = 1e-8;

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
      Matrix Function(int, int) init;
      switch (activations[x]) {
        case relu || leakyRelu:
          init = HeInit;
        case tanH || sigmoid || softmax || linear:
          init = XavierInit;
        default:
          throw Exception('Unknown activation function: $activations[x]');
      }

      _weights.add(init(_architecture[x], _architecture[x + 1]));
      _biases.add(zeros(1, _architecture[x + 1]));
    }

    _mW = List.generate(_architecture.length - 1,
        (index) => zeros(_architecture[index], _architecture[index + 1]));
    _vW = List.generate(_architecture.length - 1,
        (index) => zeros(_architecture[index], _architecture[index + 1]));
    _mB = List.generate(_architecture.length - 1,
        (index) => zeros(1, _architecture[index + 1]));
    _vB = List.generate(_architecture.length - 1,
        (index) => zeros(1, _architecture[index + 1]));
  }

  Matrix clipGradients(Matrix gradients, double threshold) {
    return gradients.performFunction((g) => g > threshold
        ? threshold
        : g < -threshold
            ? -threshold
            : g);
  }

  /// Xavier initialization for a matrix
  /// - [input] The number of input neurons (i.e., number of columns in the input matrix)
  /// - [ouput] The number of output neurons (i.e., number of columns in the output matrix)
  /// - [seed] Optional seed for random number generation
  Matrix XavierInit(int input, int ouput) {
    double limit = sqrt(6 / (input + ouput));
    return randn(input, ouput, start: limit, end: -limit, seed: seed);
  }

  /// He initialization for a matrix
  /// - [input] The number of input neurons (i.e., number of columns in the input matrix)
  /// - [output] The number of output neurons (i.e., number of columns in the output matrix)
  /// - [seed] Optional seed for random number generation
  Matrix HeInit(int input, int output) {
    double limit = sqrt(2 / input);
    return randn(input, output, start: -limit, end: limit, seed: seed);
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

  double _mse(Matrix x, Matrix y) {
    return mean(power(predict(x), 2));
  }

  Matrix _forward(Matrix input) {
    _preActivated = [];
    _activated = [];

    _preActivated.add(input);
    _activated.add(input);

    for (var i = 0; i < _architecture.length - 1; i++) {
      _preActivated.add(dot(_activated[i], _weights[i]) + _biases[i]);
      _activated.add(_activation(_preActivated[i + 1], _activations[i]));
    }

    //print(_activated[_activated.length - 1].toString());
    return _activated[_activated.length - 1];
  }

  /// Predict the output of the network
  /// - [x] The input data
  /// - Returns The output of the network
  Matrix predict(Matrix x) {
    return _forward(x);
  }

  void _backward(Matrix x, Matrix y) {
    _gradw = [];
    _gradb = [];
    _deltas = [];

    //dC/dz
    _deltas.add(_activated.last - y);

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

  void _update(double lr, int t) {
    _updateAdam(lr, t);
  }

  void _updateAdam(double lr, int t) {
    for (var i = 0; i < _architecture.length - 1; i++) {
      // Update m and v for weights
      _mW[i] = (_mW[i] * _beta1) + (_gradw[i] * (1 - _beta1));
      _vW[i] = (_vW[i] * _beta2) + (power(_gradw[i], 2) * (1 - _beta2));

      // Bias-corrected estimates
      var mHatW = _mW[i] / (1 - pow(_beta1, t));
      var vHatW = _vW[i] / (1 - pow(_beta2, t));

      // Update weights
      _weights[i] -= (mHatW * lr) /
          (sqareRoot(vHatW) + fill(_epsilon, vHatW.getRow(), vHatW.getCol()));

      // Update m and v for biases
      _mB[i] = (_mB[i] * _beta1) + (_gradb[i] * (1 - _beta1));
      _vB[i] = (_vB[i] * _beta2) + (power(_gradb[i], 2) * (1 - _beta2));

      // Bias-corrected estimates
      var mHatB = _mB[i] / (1 - pow(_beta1, t));
      var vHatB = _vB[i] / (1 - pow(_beta2, t));

      // Update biases
      _biases[i] -= (mHatB * lr) /
          (sqareRoot(vHatB) + fill(_epsilon, vHatB.getRow(), vHatB.getCol()));
    }
  }

  /// Train the network
  /// - [inputs] The input data
  /// - [expected] The expected outputs
  /// - [lr] The learning rate
  /// - [epochs] The number of epochs
  void train(List<Matrix> inputs, List<Matrix> expected, double lr, int epochs,
      {bool verbose = false}) {
    int frequency = epochs ~/ 10000;
    int t = 0;

    print("beginning training");
    for (var i = 0; i < epochs; i++) {
      for (var x = 0; x < inputs.length; x++) {
        _forward(inputs[x]);
        _backward(inputs[x], expected[x]);
        for (Matrix matrix in _weights) {
          clipGradients(matrix, 5.0);
        }
        _update(lr, t);

        t += 1;
      }

      if (verbose && i % frequency == 0) {
        print("epoch ${i + 1}: ${_mse(inputs[0], expected[0])}");
      }
    }
  }

  String getActivationName(Function activation) {
    if (activation == relu) {
      return 'relu';
    } else if (activation == softmax) {
      return 'softmax';
    } else if (activation == sigmoid) {
      return 'sigmoid';
    } else if (activation == tanH) {
      return 'tanh';
    } else if (activation == linear) {
      return 'linear';
    } else if (activation == leakyRelu) {
      return 'leakyRelu';
    } else {
      throw Exception('Unknown activation function: $activation');
    }
  }

  Map<String, dynamic> toJson() {
    return {
      'architecture': _architecture,
      'weights': _weights.map((w) => w.getMatrix()).toList(),
      'biases': _biases.map((b) => b.getMatrix()).toList(),
      'activations': _activations
          .map((f) => getActivationName(f))
          .toList(), // Note: Store function names or types
    };
  }

  Network.fromJson(Map<String, dynamic> json) {
    _architecture = List<int>.from(json['architecture']);

    // You need to implement a method to convert stored activation function names back to functions
    _activations = (json['activations'] as List)
        .map((activation) => _activationFunctionFromString(activation))
        .toList();

    _weights = (json['weights'] as List).map((matrix) {
      // Make sure to convert each element to List<List<double>> correctly
      return Matrix.fromList((matrix as List<dynamic>).map((row) {
        return List<double>.from(row as List<dynamic>);
      }).toList());
    }).toList();

    _biases = (json['biases'] as List).map((matrix) {
      // Ensure to convert each bias matrix similarly
      return Matrix.fromList((matrix as List<dynamic>).map((row) {
        return List<double>.from(row as List<dynamic>);
      }).toList());
    }).toList();
  }

  Matrix Function(Matrix) _activationFunctionFromString(String name) {
    switch (name) {
      case 'relu':
        return relu;
      case 'softmax':
        return softmax;
      case 'sigmoid':
        return sigmoid;
      case 'tanh':
        return tanH;
      case 'linear':
        return linear;
      case 'leakyRelu':
        return leakyRelu;
      default:
        throw Exception('Unknown activation function: $name');
    }
  }

  void save(String filename) {
    final file = File(filename);
    file.writeAsStringSync(json.encode(toJson()));
  }

  /// Load the network from a file
  static Network load(String filename) {
    final file = File(filename);
    final jsonString = file.readAsStringSync();
    final Map<String, dynamic> json = jsonDecode(jsonString);
    return Network.fromJson(json);
  }
}

void main() {
  // Create a network with specified layers and activation functions
  Network net = Network([1, 2, 10], [relu, softmax]);

  // Create random input and target output
  Matrix input = randn(1, 1);
  Matrix output = fill(0, 1, 10);

  output.setAt(0, 1, value: 1); // Set the desired output for training

  // Train the network with the input and output
  net.train([input], [output], 0.1, 100);

  // Save the trained network to a file
  net.save('network.json');

  // Load the network from the saved file
  Network loadedNet = Network.load('network.json');

  // Make a prediction with the loaded network using the same input
  print(loadedNet.predict(input).toString());
}
