using System;
using System.Collections.Generic;
using MachineLearning.Dataset;

namespace MachineLearning.NeuralNet
{
    class Network
    {
        private InputLayer m_inputLayer;
        private List<Layer> m_layers = new List<Layer>();
        private OutputLayer m_outputLayer;

        // Fit to a dataset
        public Network(ILabeledData data, CustomActivationSet customOutputActivation = null)
        {
            m_inputLayer = new InputLayer(data.GetDataDimension());
            m_outputLayer = new OutputLayer(data.GetLabelDimension(), customOutputActivation);
        }

        public Network(int inputSize, int outputSize, CustomActivationSet customOutputActivation = null)
        {
            m_inputLayer = new InputLayer(inputSize);
            m_outputLayer = new OutputLayer(outputSize, customOutputActivation);
        }
        
        public void AddLayer(Layer layer)
        {
            m_layers.Add(layer);
        }

        public void Build()
        {
            if (m_layers.Count == 0)
            {
                m_inputLayer.BuildConnection(null, m_outputLayer);
                m_outputLayer.BuildConnection(m_inputLayer, m_outputLayer);
            }
            else
            {
                m_inputLayer.BuildConnection(null, m_layers[0]);

                Layer previousLayer = m_inputLayer;
                Layer currentLayer = m_layers[0];
                for(int i = 1; i < m_layers.Count; i++)
                {
                    currentLayer.BuildConnection(previousLayer, m_layers[i]);
                    previousLayer = currentLayer;
                    currentLayer = m_layers[i];
                }
                currentLayer.BuildConnection(previousLayer, m_outputLayer);

                m_outputLayer.BuildConnection(currentLayer, null);
            }
        }

        public float CalculateCost(float[] expected)
        {
            return m_outputLayer.CalculateCost(expected);
        }

        /// <summary>
        /// Train the network for 1 epoch
        /// </summary>
        /// <param name="data">The dataset</param>
        /// <param name="batchSize">How many mini batch size before applying trained weights</param>
        /// <param name="learningRate">The learning speed, 0.005f is a good value</param>
        /// <param name="logTrainingData">Prints training data to the console</param>
        public void Train(ILabeledData data, int batchSize, float learningRate, bool logTrainingData)
        {
            if (data.GetDataCount() != data.GetLabelCount()) throw new System.Exception("Data and labels are not matched");

            Random rand = new Random();

            List<int> shuffledData = new List<int>(data.GetDataCount());
            for(int i = 0; i < data.GetDataCount(); i++)
            {
                shuffledData.Add(i);
            }

            float cost = 0;
            int dataCount = 0;
            int completeness = 1;
            while (shuffledData.Count > 0)
            {
                
                for (int j = 0; j < batchSize; j++)
                {
                    if (shuffledData.Count == 0) break;
                    int randomIndex = rand.Next(0, shuffledData.Count);
                    int dataIndex = shuffledData[randomIndex];
                    shuffledData.RemoveAt(randomIndex);

                    m_inputLayer.FeedInput(data.GetInput(dataIndex));
                    m_inputLayer.Predict();
                    m_outputLayer.Backpropagate(data.GetTarget(dataIndex), learningRate);

                    cost += m_outputLayer.CalculateCost(data.GetTarget(dataIndex));
                    dataCount++;
                }

                ApplyChanges(learningRate);

                float percentComplete = (((float)(data.GetDataCount() - shuffledData.Count)) / data.GetDataCount()) * 100f;
                if (logTrainingData && percentComplete/10.0f >= completeness)
                {
                    completeness++;
                    Console.WriteLine(string.Format("{0,4:F2}% Done. Current Cost: {1,6:F4}", percentComplete, cost / dataCount));
                    cost = 0;
                    dataCount = 0;
                }
            }
        }

        public float[] Predict(IInputData data, int index)
        {
            m_inputLayer.FeedInput(data.GetInput(index));
            m_inputLayer.Predict();
            return m_outputLayer.GetOutput();
        }

        private void ApplyChanges(float learningRate)
        {
            m_inputLayer.ApplyChanges(learningRate);
            foreach(Layer layer in m_layers)
            {
                layer.ApplyChanges(learningRate);
            }
        }
    }
}
