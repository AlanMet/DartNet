import 'dart:io';
import 'matrices.dart';

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

  Network(List<int> architecture, List<Matrix Function(Matrix)> activations) {
    _architecture = architecture;
    _activations = activations;

    for (int x = 0; x < _architecture.length - 1; x++) {
      _weights.add(randn(_architecture[x], _architecture[x + 1]));
      _biases.add(zeros(1, _architecture[x + 1]));
    }
  }

  Matrix _activation(Matrix input, Matrix Function(Matrix) function) {
    return function(input);
  }

  Matrix _activationDeriv(Matrix input, Matrix Function(Matrix) function) {
    return derivative(function)(input);
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

    //print(_activated[_activated.length - 1].toString());
    return _activated[_activated.length - 1];
  }

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

  void update(double lr) {
    for (var i = 0; i < _architecture.length - 1; i++) {
      _weights[i] -= _gradw[i] * lr;
      _biases[i] -= _gradb[i] * lr;
    }
  }

  void train(
      List<Matrix> inputs, List<Matrix> expected, double lr, int epochs) {
    print("beginning training");
    for (var i = 0; i < epochs; i++) {
      for (var x = 0; x < inputs.length; x++) {
        _forward(inputs[x]);
        _backward(inputs[x], expected[x]);
        update(lr);
      }
      print("epoch $i: ${mse(inputs[0], expected[0])}");
    }
  }

  void store(String name) {
    String filePath = '$name.txt';
    File file = File(filePath);

    if (!file.existsSync()) {
      file.createSync(recursive: true);
    }

    String content = "";
    for (var weight in _weights) {
      content += weight.toString();
    }

    file.writeAsStringSync(content);

    print('Matrix stored successfully in $filePath');
  }
}

void main() {
  Network net = Network([1, 2, 2], [relu, softmax]);

  Matrix input = randn(1, 1);
  Matrix output = fill(0, 1, 2);
  output.setAt(0, 1, value: 1);

  print(input.toString());
  print(net._forward(input).toString());
  net.train([input], [output], 0.01, 1000);
  print(net._forward(input).toString());
  print(output.toString());
}
