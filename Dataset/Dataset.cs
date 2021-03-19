using System;
using System.IO;

namespace MachineLearning.Dataset
{
    interface IInputData 
    {
        public int GetDataCount();
        public int GetDataDimension();
        public float[] GetInput(int i);
    }

    interface ILabelData
    {
        public int GetLabelCount();
        public int GetLabelDimension();
        public float[] GetTarget(int i);
    }

    interface ILabeledData : IInputData, ILabelData {}

    class MNISTDataset : ILabeledData
    {
        private int count;
        private int columnCount;
        private int rowCount;
        private int byteCount;

        private byte[] labels;
        private byte[] data;

        public static MNISTDataset LoadDataset(string imageFile, string labelFile)
        {
            MNISTDataset dataset = new MNISTDataset();

            Int32 dataCount = 0;
            using (FileStream fs = File.OpenRead(imageFile))
            {
                byte[] infoBuffer = new byte[4];

                fs.Read(infoBuffer, 0, 4);
                fs.Read(infoBuffer, 0, 4);
                if (BitConverter.IsLittleEndian)
                {
                    Array.Reverse(infoBuffer);
                }
                dataCount = BitConverter.ToInt32(infoBuffer);
                fs.Read(infoBuffer, 0, 4);
                if (BitConverter.IsLittleEndian)
                {
                    Array.Reverse(infoBuffer);
                }
                Int32 rowCount = BitConverter.ToInt32(infoBuffer);
                fs.Read(infoBuffer, 0, 4);
                if (BitConverter.IsLittleEndian)
                {
                    Array.Reverse(infoBuffer);
                }
                Int32 colCount = BitConverter.ToInt32(infoBuffer);

                dataset.count = dataCount;
                dataset.rowCount = rowCount;
                dataset.columnCount = colCount;

                byte[] dataBuffer = new byte[dataCount * rowCount * colCount];
                fs.Read(dataBuffer, 0, dataBuffer.Length);
                dataset.data = dataBuffer;
            }

            Int32 labelCount = 0;
            using (FileStream fs = File.OpenRead(labelFile))
            {
                byte[] infoBuffer = new byte[4];

                fs.Read(infoBuffer, 0, 4);
                fs.Read(infoBuffer, 0, 4);
                if (BitConverter.IsLittleEndian)
                {
                    Array.Reverse(infoBuffer);
                }
                labelCount = BitConverter.ToInt32(infoBuffer);

                byte[] labelBuffer = new byte[dataCount];
                fs.Read(labelBuffer, 0, dataCount);
                dataset.labels = labelBuffer;
            }

            if(dataCount != labelCount)
            {
                throw new Exception("Data count is not equal to label count!");
            }

            dataset.byteCount = dataset.columnCount * dataset.rowCount;

            return dataset;
        }

        public int GetDataCount()
        {
            return count;
        }

        public int GetDataDimension()
        {
            return byteCount;
        }

        public float[] GetInput(int i)
        {
            float[] inputArr = new float[byteCount];
            for (int j = 0; j < byteCount; j++)
            {
                inputArr[j] = (data[byteCount * i + j])/255.0f;
            }
            return inputArr;
        }

        public float[] GetTarget(int i)
        {
            int label = labels[i];
            float[] target = new float[10];
            target[label] = 1.0f;
            return target;
        }

        public int GetLabel(int i)
        {
            return labels[i];
        }

        public int GetLabelCount()
        {
            return count;
        }

        public int GetLabelDimension()
        {
            return 10;
        }

        public int PredictedNumber(float[] predicted)
        {
            int max = 0;
            float maxCertainty = 0.0f;

            for(int i = 0; i < predicted.Length; i++)
            {
                if (predicted[i] > maxCertainty)
                {
                    max = i;
                    maxCertainty = predicted[i];
                }
            }
            return max;
        }

        public bool CheckPrediction(float[] predicted, int dataIndex)
        {
            int num = PredictedNumber(predicted);

            return labels[dataIndex] == num;
        }

        public override string ToString()
        {
            return string.Format("Data count : {0}\nX Resolution : {1}\nY Resolution : {2}", count, columnCount, rowCount);
        }
    }
}
