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
            ImageLabelSet trainingSet = ImageLabelSet.LoadDataset("dataset/t10k-images.idx3-ubyte", "dataset/t10k-labels.idx1-ubyte");

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

                    if ((i+1) % 5000 == 0)
                    {
                        Console.WriteLine(string.Format("{0,4:F2}% done", ((float)(i+1) /dataset.Count)*100f));
                    }
                }
                Console.WriteLine(string.Format("Current cost : {0}\nEpoch : {1}", cost/dataset.Count, epoch));
                network.ApplyChanges();

                Console.Write(string.Format("Test prediction? (y/n) "));
                ConsoleKeyInfo key = Console.ReadKey();
                Console.WriteLine("");
                while (key.KeyChar != 'y' && key.KeyChar != 'n')
                {
                    Console.Write(string.Format("Test prediction? (y/n) "));
                    key = Console.ReadKey();
                    Console.WriteLine("");
                }

                if (key.KeyChar == 'y')
                {
                    while (true)
                    {
                        Random rand = new Random();
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
                        key = Console.ReadKey();
                        Console.WriteLine("");
                        while (key.KeyChar != 'y' && key.KeyChar != 'n')
                        {
                            Console.Write(string.Format("Test prediction? (y/n)"));
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
    }
}
