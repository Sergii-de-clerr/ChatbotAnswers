using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.IO;
using System.Linq;
using System.Windows.Forms;

namespace ChatbotAnswers
{
    public partial class Form1 : Form
    {
        MLContext mlContext;
        ITransformer mlmodel;
        public class QAData
        {
            [LoadColumn(2)] public string Question { get; set; }
            [LoadColumn(5)] public string Answer { get; set; }
        }

        public class QAOutput
        {
            [ColumnName("PredictedLabel")] public string PredictedAnswer { get; set; }
        }
        public Form1()
        {
            InitializeComponent();
            //InitialiseDataset();
            ImportDataset();
        }

        public void InitialiseDataset()
        {
            string dataPath = "trainSQuAD.csv";

            var mlContext = new MLContext();

            var lines = File.ReadAllLines(dataPath).Skip(1).Take(20000).ToArray();

            var qaData = lines.Select(line =>
            {
                var columns = line.Split(',');
                if (columns.Length >= 4)
                {
                    return new QAData
                    {
                        Question = columns[columns.Length - 4],
                        Answer = columns[columns.Length - 1]
                    };
                }
                else
                {
                    return null;
                }
            }).Where(fakedata => fakedata != null).ToList();

            var data = mlContext.Data.LoadFromEnumerable(qaData);

            var dataSplit = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);
            var trainData = dataSplit.TrainSet;
            var testData = dataSplit.TestSet;

            var pipeline = mlContext.Transforms.Text.FeaturizeText("QuestionFeaturized", "Question")
                .Append(mlContext.Transforms.Conversion.MapValueToKey("Label", "Answer"))
                .Append(mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(labelColumnName: "Label", featureColumnName: "QuestionFeaturized"))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            var model = pipeline.Fit(trainData);

            var modelPath = "savedVeryBigModel.zip";
            mlContext.Model.Save(model, data.Schema, modelPath);
        }

        public void ImportDataset()
        {
            mlContext = new MLContext();
            var modelPath = "savedBigModel.zip";
            mlmodel = mlContext.Model.Load(modelPath, out var inputSchema);

            //var predictions = mlmodel.Transform(testData);
            //var metrics = mlContext.MulticlassClassification.Evaluate(predictions, labelColumnName: "Label", scoreColumnName: "Score");
            //Console.WriteLine($"MicroAccuracy: {metrics.MicroAccuracy}, MacroAccuracy: {metrics.MacroAccuracy}");
        }

        private void button1_Click(object sender, EventArgs e)
        {
            var predictionEngine = mlContext.Model.CreatePredictionEngine<QAData, QAOutput>(mlmodel);

            var sample = new QAData
            {
                Question = textBox1.Text
            };

            var prediction = predictionEngine.Predict(sample);
            label2.Text = prediction.PredictedAnswer;
        }
    }
}
