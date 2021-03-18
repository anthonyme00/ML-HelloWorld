using System;
using System.Collections.Generic;
using System.Text;

namespace MachineLearning.NeuralNet
{
    class Network
    {
        private InputLayer m_inputLayer;
        private List<Layer> m_layers = new List<Layer>();
        private OutputLayer m_outputLayer;
        public float learningSpeed = 0.5f;

        public Network(int inputSize, int outputSize, float learningSpeed)
        {
            m_inputLayer = new InputLayer(inputSize);
            m_outputLayer = new OutputLayer(outputSize);
            this.learningSpeed = learningSpeed;
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

        public void Train(float[] expected)
        {
            m_outputLayer.Backpropagate(expected, learningSpeed);
        }

        public void ApplyChanges()
        {
            m_inputLayer.ApplyChanges();
            foreach(Layer layer in m_layers)
            {
                layer.ApplyChanges();
            }
        }

        public float[] Predict(float[] data)
        {
            m_inputLayer.FeedInput(data);
            m_inputLayer.Predict();
            return m_outputLayer.GetOutput();
        }
    }
}
