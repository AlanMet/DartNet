import 'matrices.dart';
import 'networks.dart';
import 'dart:math';

List<List<Matrix>> generateData(int amount) {
  List<Matrix> labels = [];
  List<Matrix> inputs = [];
  Random rand = Random();
  for (var i = 0; i < amount; i++) {
    double number = rand.nextDouble() * 2 - 1;
    Matrix input = fill(number, 1, 1);
    Matrix label = fill(sin(number), 1, 1);

    labels.add(label);
    inputs.add(input);
  }

  return [labels, inputs];
}

void getAccuracy(Network net) {
  List<List<Matrix>> data = generateData(10000);
  int total = 0;
  for (var i = 0; i < data[0].length; i++) {
    //checks if the prediction is within 0.01 of the actual value and considers the prediction correct if that is the case
    if ((net.predict(data[1][i]).getAt(0, 0) - data[0][i].getAt(0, 0)).abs() <
        0.01) {
      total += 1;
    }
  }
  print("Accuracy: ${(total / data[0].length) * 100}%");
}

void main() {
  Network net = Network([1, 10, 10, 1], [relu, tanH, tanH]);
  List<List<Matrix>> data = generateData(10000);

  net.train(data[1], data[0], 0.0001, 10000, verbose: true);
  getAccuracy(net);
}
