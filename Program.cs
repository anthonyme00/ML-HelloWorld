using MachineLearning.NeuralNet;
using MachineLearning.Dataset;
using System;
using System.IO;
using System.Collections.Generic;

namespace MachineLearning
{
    class Program
    {
        static void Main(string[] args)
        {
            float learningStep = 0.005f;
            int epochCount = 100;
            //float costCutoff = 0.05f;
            int batchCount = 100;

            ImageLabelSet dataset = ImageLabelSet.LoadDataset("dataset/train-images.idx3-ubyte", "dataset/train-labels.idx1-ubyte");
            ImageLabelSet trainingSet = ImageLabelSet.LoadDataset("dataset/t10k-images.idx3-ubyte", "dataset/t10k-labels.idx1-ubyte");

            Console.WriteLine(string.Format("Images in dataset: {0}", dataset.Count));
            Console.WriteLine(string.Format("Row : {0}", dataset.RowCount));
            Console.WriteLine(string.Format("Column : {0}", dataset.ColumnCount));

            int bytesPerImage = dataset.RowCount * dataset.ColumnCount;
            Network network = new Network(bytesPerImage, 10, learningStep);
            network.AddLayer(new Layer(16, Activation.Relu));
            network.AddLayer(new Layer(16, Activation.Relu));
            network.Build();

            float[] learningData = new float[bytesPerImage];
            float[] expectedData = new float[10];
            float[] output = new float[10];

            Console.WriteLine(string.Format("Starting training; epoch count : {0}; batch size : {1}; learning speed : {2,4:F2}", epochCount, batchCount, learningStep));

            float cost = 100;
            Random rand = new Random();
            List<int> datasetList = new List<int>(dataset.Count);
            for (int epoch = 0; epoch < epochCount; epoch++)
            {
                for (int i = 0; i < dataset.Count; i++)
                {
                    datasetList.Add(i);
                }

                cost = 0;
                Console.WriteLine(string.Format("Starting epoch {0}", epoch));
                int trainedCount = 0;
                for (int i = 0; i < dataset.Count; i++)
                {
                    trainedCount++;
                    int randDataIndex = rand.Next(0, datasetList.Count);
                    int randData = datasetList[randDataIndex];
                    datasetList.RemoveAt(randDataIndex);
                    for (int j = 0; j < bytesPerImage; j++)
                    {
                        learningData[j] = (float)dataset.Data[randData * bytesPerImage + j] / 255.0f;
                    }

                    for (int j = 0; j < 10; j++)
                    {
                        if (j == (int)dataset.Labels[randData])
                        {
                            expectedData[j] = 1.0f;
                        }
                        else
                        {
                            expectedData[j] = 0.0f;
                        }
                    }

                    network.Predict(learningData);
                    network.Train(expectedData);
                    cost += network.CalculateCost(expectedData);

                    if ((i+1) % batchCount == 0 || (i+1) == dataset.Count)
                    {
                        network.ApplyChanges(learningStep);
                        Console.WriteLine(string.Format("{0,4:F2}% done", ((float)(i+1) /dataset.Count)*100f));
                        Console.WriteLine(string.Format("Current cost : {0}\nEpoch : {1}", cost / trainedCount, epoch));
                        trainedCount = 0;
                        cost = 0;
                    }
                }
            }
            while (true)
            {
                int id = rand.Next(0, trainingSet.Count);

                for (int y = 0; y < trainingSet.RowCount; y++)
                {
                    for (int x = 0; x < trainingSet.ColumnCount; x++)
                    {
                        float data = (float)trainingSet.Data[id * bytesPerImage + y * trainingSet.ColumnCount + x] / 255.0f;
                        learningData[y * trainingSet.ColumnCount + x] = data;
                        if (data > 0.9f)
                        {
                            Console.Write("$");
                        }
                        else if (data > 0.75f)
                        {
                            Console.Write("#");
                        }
                        else if (data > 0.6f)
                        {
                            Console.Write("Z");
                        }
                        else if (data > 0.5f)
                        {
                            Console.Write("t");
                        }
                        else if (data > 0.35f)
                        {
                            Console.Write("\\");
                        }
                        else if (data > 0.2f)
                        {
                            Console.Write("`");
                        }
                        else
                        {
                            Console.Write(" ");
                        }
                    }
                    Console.WriteLine("");
                }
                Console.WriteLine("Number is : " + trainingSet.Labels[id]);

                float[] prediction = network.Predict(learningData);
                for (int i = 0; i < prediction.Length; i++)
                {
                    Console.Write(string.Format("{0} - {1,4:F2}%; ", i, prediction[i] * 100));
                }
                Console.Write("\nTest again? (y/n)");
                ConsoleKeyInfo key = Console.ReadKey();
                Console.WriteLine("");
                while (key.KeyChar != 'y' && key.KeyChar != 'n')
                {
                    Console.Write(string.Format("Test again? (y/n)"));
                    key = Console.ReadKey();
                    Console.WriteLine("");
                }

                if (key.KeyChar == 'n')
                {
                    break;
                }
            }
        }
    }
}
