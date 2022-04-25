import 'dart:io';
import 'package:ml_algo/ml_algo.dart';
import 'package:ml_dataframe/ml_dataframe.dart';

void main() async {
  final fileName = 'diabetes_classifier.json';
  final file = File(fileName);
  final encodedModel = await file.readAsString();
  final classifier = KnnRegressor.fromJson(encodedModel);
  final data = await fromCsv('assets/inferenceData.csv');
  final prediction = classifier.predict(data);
  print(prediction.header);
  print(prediction.rows);
}
