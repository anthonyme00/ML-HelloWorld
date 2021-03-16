using MachineLearning.NeuralNet;
using MachineLearning.Dataset;
using System;
using System.IO;

namespace MachineLearning
{
    class Program
    {
        static void Main(string[] args)
        {
            ImageLabelSet dataset = ImageLabelSet.LoadDataset("dataset/train-images.idx3-ubyte", "dataset/train-labels.idx1-ubyte");

            Console.WriteLine(dataset.Count);
            Console.WriteLine(dataset.RowCount);
            Console.WriteLine(dataset.ColumnCount);
            Console.WriteLine(dataset.Labels.Length);

            int bytesPerImage = dataset.RowCount * dataset.ColumnCount;
            Network network = new Network(bytesPerImage, 10, 0.1f);
            network.AddLayer(new Layer(32, Activation.Relu));
            network.AddLayer(new Layer(16, Activation.Relu));
            network.Build();

            float[] learningData = new float[bytesPerImage];
            float[] expectedData = new float[10];
            float[] output = new float[10];

            for (int i = 0; i < dataset.Count; i++)
            {
                for (int y = 0; y < dataset.RowCount; y++)
                {
                    for (int x = 0; x < dataset.ColumnCount; x++)
                    {
                        learningData[dataset.ColumnCount * y + x] = (float)dataset.Data[i * dataset.ColumnCount * dataset.RowCount + y * dataset.ColumnCount + x] / 255.0f;

                        if (dataset.Data[i * dataset.ColumnCount * dataset.RowCount + y * dataset.ColumnCount + x] > 127)
                        {
                            Console.Write("#");
                        }
                        else
                        {
                            Console.Write(" ");
                        }
                    }
                    Console.Write("\n");
                }
                Console.WriteLine("Number is: " + dataset.Labels[i]);

                for (int j = 0; j < 10; j++)
                {
                    if (j == (int)dataset.Labels[i])
                    {
                        expectedData[j] = 1.0f;
                    }
                    else
                    {
                        expectedData[j] = 0.0f;
                    }
                }

                network.Predict(learningData);
                output = network.Train(expectedData);
                for (int j = 0; j < 10; j++)
                {
                    Console.Write(string.Format("{0} - {1}%; ", j, output[j] * 100));
                }
                Console.WriteLine(string.Format("\nCost is: {0}\n", network.CalculateCost(expectedData)));
            }                     
        }
    }
}
