﻿using MachineLearning.NeuralNet;
using MachineLearning.Dataset;
using System;
using System.Collections.Generic;

namespace MachineLearning
{
    class Program
    {
        static void Main(string[] args)
        {
            float learningStep = 0.005f;
            int epochCount = 100;
            int batchCount = 100;

            ILabeledData trainingSet = MNISTDataset.LoadDataset("dataset/train-images.idx3-ubyte", "dataset/train-labels.idx1-ubyte");
            ILabeledData testingSet = MNISTDataset.LoadDataset("dataset/t10k-images.idx3-ubyte", "dataset/t10k-labels.idx1-ubyte");

            Console.WriteLine(string.Format("Training Dataset : \n{0}", (MNISTDataset)trainingSet));
            Console.WriteLine(string.Format("\nTesting Dataset : \n{0}", (MNISTDataset)testingSet));

            Network network = new Network(trainingSet);
            network.AddLayer(new Layer(16, Activation.Relu));
            network.AddLayer(new Layer(16, Activation.Relu));
            network.Build();
            Console.WriteLine(string.Format("Starting training\n\nepoch count : {0}\nbatch size : {1}\nlearning step : {2,4:F2}", epochCount, batchCount, learningStep));

            for (int epoch = 0; epoch < epochCount; epoch++)
            {
                Console.WriteLine(string.Format("Starting epoch {0}", epoch));
                network.Train(trainingSet, batchCount, learningStep, true);
            }
            Console.WriteLine("Finished training. Evaluating performance");

            int total = testingSet.GetDataCount();
            int correctPrediction = 0;
            float cost = 0;
            for (int i = 0; i < testingSet.GetDataCount(); i++)
            {
                float[] output = network.Predict(testingSet, i);
                cost += network.CalculateCost(testingSet.GetTarget(i));
                if (((MNISTDataset)testingSet).CheckPrediction(output, i))
                {
                    correctPrediction++;
                }
            }
            Console.WriteLine(string.Format("Cost : {0,4:F2}\nCorrect percentage : {1,4:F2}", cost, ((float)correctPrediction / total) * 100f));

            Random rand = new Random();
            while (true)
            {
                int id = rand.Next(0, trainingSet.GetDataCount());

                float[] input = trainingSet.GetInput(id);
                for (int y = 0; y < 28; y++)
                {
                    for (int x = 0; x < 28; x++)
                    {
                        float data = input[y * 28 + x];
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
                Console.WriteLine("Number is : " + ((MNISTDataset)trainingSet).GetLabel(id));

                float[] prediction = network.Predict(trainingSet, id);
                for (int i = 0; i < prediction.Length; i++)
                {
                    Console.Write(string.Format("{0} - {1,4:F2}%; ", i, prediction[i] * 100));
                }
                Console.WriteLine(string.Format("\nPredicted number is : {0}\n", ((MNISTDataset)trainingSet).PredictedNumber(prediction)));
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
