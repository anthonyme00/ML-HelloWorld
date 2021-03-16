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

            Console.WriteLine(string.Format("Images in dataset: {0}", dataset.Count));
            Console.WriteLine(string.Format("Row : {0}", dataset.RowCount));
            Console.WriteLine(string.Format("Column : {0}", dataset.ColumnCount));

            int bytesPerImage = dataset.RowCount * dataset.ColumnCount;
            Network network = new Network(bytesPerImage, 10, 0.1f);
            network.AddLayer(new Layer(16, Activation.Sigmoid));
            network.AddLayer(new Layer(16, Activation.Sigmoid));
            network.Build();

            float[] learningData = new float[bytesPerImage];
            float[] expectedData = new float[10];
            float[] output = new float[10];

            
            for (int epoch = 0; epoch < 100; epoch++)
            {
                Console.WriteLine(string.Format("Starting epoch {0}", epoch));
                float cost = 0;
                for (int i = 0; i < dataset.Count; i++)
                {
                    for (int j = 0; j < bytesPerImage; j++)
                    {
                        learningData[j] = (float)dataset.Data[i * bytesPerImage + j] / 255.0f;
                    }

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
                    network.Train(expectedData);
                    cost += network.CalculateCost(expectedData);

                    if (i % 5000 == 0)
                    {
                        Console.WriteLine(string.Format("{0,4:F2}% done", ((float)i /dataset.Count)*100f));
                    }
                }
                Console.WriteLine(string.Format("Current cost : {0}\nEpoch : {1}", cost/dataset.Count, epoch));
                network.ApplyChanges();
            }
                      
        }
    }
}
