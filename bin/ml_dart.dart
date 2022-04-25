import 'package:ml_algo/ml_algo.dart';
import 'package:ml_dataframe/ml_dataframe.dart';

void main() async {
  final samples = await fromCsv('assets/housing.csv', headerExists: true);
  const targetColumnName = 'MEDV';

  final splits = splitData(samples, [0.7]);
  final trainData = splits[0];
  final testData = splits[1];

  final validator = CrossValidator.kFold(samples, numberOfFolds: 5);

  // ignore: prefer_function_declarations_over_variables
  final createRegressor = (DataFrame samples) => KnnRegressor(
        samples,
        targetColumnName,
        2,
      );

  final scores = await validator.evaluate(createRegressor, MetricType.rmse);
  final accuracy = scores.mean();

  // ignore: avoid_print
  print('accuracy on k fold validation: ${accuracy.toStringAsFixed(2)}');

  final regressor = createRegressor(trainData);
  final finalScore = regressor.assess(testData, MetricType.rmse);
  
  // ignore: avoid_print
  print('Final score of model: ${finalScore.toStringAsFixed(2)}');

  await regressor.saveAsJson('diabetes_classifier.json');
}
