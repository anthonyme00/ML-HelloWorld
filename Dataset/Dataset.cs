using System;
using System.IO;

namespace MachineLearning.Dataset
{
    class ImageLabelSet
    {
        public int Count { get; private set; }
        public int ColumnCount { get; private set; }
        public int RowCount { get; private set; }

        public byte[] Labels { get; private set; }
        public byte[] Data { get; private set; }

        public static ImageLabelSet LoadDataset(string imageFile, string labelFile)
        {
            ImageLabelSet dataset = new ImageLabelSet();

            using (FileStream fs = File.OpenRead(imageFile))
            {
                byte[] infoBuffer = new byte[4];

                fs.Read(infoBuffer, 0, 4);
                fs.Read(infoBuffer, 0, 4);
                if (BitConverter.IsLittleEndian)
                {
                    Array.Reverse(infoBuffer);
                }
                Int32 dataCount = BitConverter.ToInt32(infoBuffer);
                Console.WriteLine(string.Format("Images in dataset: {0}", dataCount));
                fs.Read(infoBuffer, 0, 4);
                if (BitConverter.IsLittleEndian)
                {
                    Array.Reverse(infoBuffer);
                }
                Int32 rowCount = BitConverter.ToInt32(infoBuffer);
                Console.WriteLine(string.Format("Number of rows: {0}", rowCount));
                fs.Read(infoBuffer, 0, 4);
                if (BitConverter.IsLittleEndian)
                {
                    Array.Reverse(infoBuffer);
                }
                Int32 colCount = BitConverter.ToInt32(infoBuffer);
                Console.WriteLine(string.Format("Number of cols: {0}", colCount));

                dataset.Count = dataCount;
                dataset.RowCount = rowCount;
                dataset.ColumnCount = colCount;

                byte[] dataBuffer = new byte[dataCount * rowCount * colCount];
                fs.Read(dataBuffer, 0, dataBuffer.Length);
                dataset.Data = dataBuffer;
            }

            using (FileStream fs = File.OpenRead(labelFile))
            {
                byte[] infoBuffer = new byte[4];

                fs.Read(infoBuffer, 0, 4);
                fs.Read(infoBuffer, 0, 4);
                if (BitConverter.IsLittleEndian)
                {
                    Array.Reverse(infoBuffer);
                }
                Int32 dataCount = BitConverter.ToInt32(infoBuffer);

                byte[] labelBuffer = new byte[dataCount];
                fs.Read(labelBuffer, 0, dataCount);
                dataset.Labels = labelBuffer;
            }

            return dataset;
        }
    }
}
