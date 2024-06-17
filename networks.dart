import 'matrices.dart';

class Network {
  int? seed;
  late List<int> _architecture;
  late List<double Function(double)> _activations;
  List<Matrix> _weights = [],
      _biases = [],
      _preActivated = [],
      _activated = [],
      _gradw = [],
      _gradb = [],
      _deltas = [];

  Network(List<int> architecture, List<double Function(double)> activations) {
    _architecture = architecture;
    _activations = activations;

    for (int x = 0; x < _architecture.length - 1; x++) {
      _weights.add(randn(_architecture[x], _architecture[x + 1]));
      print(_weights[x].toString());
      _biases.add(randn(1, _architecture[x + 1]));
      print(_biases[x].toString());
    }
  }

  Matrix _activation(Matrix input, double Function(double) function) {
    return input.performFunction(function);
  }

  Matrix _activation_deriv(Matrix input, double Function(double) function) {
    return input.performFunction(derivative(function));
  }

  double mse(Matrix x, Matrix y) {
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

    return _activated[_activated.length - 1];
  }

  Matrix predict(Matrix x) {
    return _forward(x);
  }

  void _backward(Matrix x, Matrix y) {
    _gradw = [];
    _gradb = [];
    _deltas = [];

    //not my algorithm
    _deltas.add(_activated[_activated.length - 1] - y);

    _gradw.add(dot(_activated[_activated.length - 2].transpose(),
        _deltas[_deltas.length - 1]));

    _gradb.add(sum(_deltas[_deltas.length - 1], 0));

    for (var i = _architecture.length - 2; i > 0; i--) {
      _deltas.add(dot(_deltas[_deltas.length - 1], _weights[i].transpose()) *
          _activation_deriv(_preActivated[i], _activations[i]));
      _gradw
          .add(dot(_activated[i - 1].transpose(), _deltas[_deltas.length - 1]));
      _gradb.add(sum(_deltas[_deltas.length - 1], 0));
    }

    _gradw = _gradw.reversed.toList();
    _gradb = _gradb.reversed.toList();
    _deltas = _deltas.reversed.toList();
  }

  void update(double lr) {
    for (var i = 0; i < _architecture.length - 1; i++) {
      _weights[i] -= _gradw[i] * lr;
      _biases[i] -= _gradb[i] * lr;
    }
  }

  void train(
      List<Matrix> inputs, List<Matrix> expected, double lr, int epochs) {
    for (var i = 0; i < epochs; i++) {
      _forward(inputs[i]);
      _backward(inputs[i], expected[i]);
      update(lr);
      if (i % 100 == 0) {
        print("epoch ${i}: ${mse(inputs[i], expected[i])}");
      }
    }
  }
}
