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

            for (int i = 0; i < dataset.Count; i++)
            {
                for (int y = 0; y < dataset.RowCount; y++)
                {
                    for (int x = 0; x < dataset.ColumnCount; x++)
                    {
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

                if (i > 5) break;
            }            

            Console.WriteLine(dataset.Labels.Length);
        }
    }
}
