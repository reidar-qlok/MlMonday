using Microsoft.ML;
using Microsoft.ML.Data;

namespace MlMonday
{
    public class FaqData
    {
        [LoadColumn(0)]
        public string Question { get; set; }

        [LoadColumn(1)]
        public string Answer { get; set; }
    }

    public class FaqPrediction
    {
        [ColumnName("PredictedLabel")]
        public string Answer { get; set; }
    }

    public class Program
    {
        static void Main(string[] args)
        {
            string dataPath = "C:/Users/reidar/source/repos/MlMonday/MlMonday/inmodel.csv";
            var mlContext = new MLContext();

            var dataView = mlContext.Data.LoadFromTextFile<FaqData>(dataPath, separatorChar: ';', hasHeader: true);

            var dataProcessPipeline = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "Label", inputColumnName: nameof(FaqData.Answer))
                .Append(mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(FaqData.Question)))
                .AppendCacheCheckpoint(mlContext);

            var trainer = mlContext.MulticlassClassification.Trainers.OneVersusAll(mlContext.BinaryClassification.Trainers.LbfgsLogisticRegression());
            var trainingPipeline = dataProcessPipeline
                .Append(trainer)
                .Append(mlContext.Transforms.Conversion.MapKeyToValue(outputColumnName: "PredictedLabel", inputColumnName: "PredictedLabel"));

            var trainedModel = trainingPipeline.Fit(dataView);

            // Utvärdera modellen
            var testMetrics = mlContext.MulticlassClassification.Evaluate(trainedModel.Transform(dataView));

            System.Console.WriteLine($"Log-loss: {testMetrics.LogLoss}");
            System.Console.WriteLine($"Per class Log-loss: {string.Join(" , ", testMetrics.PerClassLogLoss.Select(l => l.ToString("N4")))}");

            mlContext.Model.Save(trainedModel, dataView.Schema, "model.zip");
            System.Console.WriteLine("Modellen är sparad.");
        }
    }
}