using System;
using System.Security.Cryptography;

namespace MachineLearning.NeuralNet
{
    public class Layer
    {
        public int nodeCount { get; private set; }
        public ActivationSet activation { get; private set; }

        protected float[] inputs;
        protected float[] weights;
        protected float[] biases;

        protected float[] deltaWeights;
        protected float[] deltaBiases;
        protected int trainingCount;

        protected Layer previousLayer;
        protected Layer nextLayer;

        private static RandomNumberGenerator seedGenerator;

        public virtual float[] GetOutput()
        {
            float[] outputs = new float[nodeCount];
            inputs.CopyTo(outputs, 0);

            for (int i = 0; i < nodeCount; i++)
            {
                outputs[i] = activation.activationFunc(outputs[i]);
            }

            return outputs;
        }

        public Layer(int nodeCount, ActivationSet activation)
        {
            if (seedGenerator == null)
            {
                seedGenerator = RandomNumberGenerator.Create();
            }
            this.nodeCount = nodeCount;
            this.activation = activation;
            this.inputs = new float[nodeCount];
        }

        public virtual void BuildConnection(Layer previousLayer, Layer nextLayer)
        {
            if (nextLayer != null)
            {
                this.biases = new float[nextLayer.nodeCount];
                this.weights = new float[nextLayer.nodeCount * this.nodeCount];
                this.deltaBiases = new float[nextLayer.nodeCount];
                this.deltaWeights = new float[nextLayer.nodeCount * this.nodeCount];

                byte[] seed = new byte[4];
                seedGenerator.GetBytes(seed);
                Random rand = new Random(BitConverter.ToInt32(seed, 0));

                for (int i = 0; i < nextLayer.nodeCount; i++)
                {
                    biases[i] = (float)(rand.NextDouble() - 0.5);
                    for (int j = 0; j < nodeCount; j++)
                    {
                        weights[i * nodeCount + j] = (float)(rand.NextDouble()-0.5);
                    }
                }
            }
            
            this.previousLayer = previousLayer;
            this.nextLayer = nextLayer;
        }        

        public virtual void ApplyChanges()
        {
            if (trainingCount == 0) return;
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] += deltaWeights[i] / trainingCount;
                deltaWeights[i] = 0;
            }
            for (int i = 0; i < biases.Length; i++)
            {
                biases[i] += deltaBiases[i] / trainingCount;
                deltaBiases[i] = 0;
            }
            trainingCount = 0;
        }

        public virtual void Predict()
        {
            if (nextLayer == null) return;
            for (int nodeIndex = 0; nodeIndex < nextLayer.nodeCount; nodeIndex++)
            {
                nextLayer.inputs[nodeIndex] = biases[nodeIndex];
            }
            for (int nodeIndex = 0; nodeIndex < nodeCount; nodeIndex++)
            {
                for (int inputIndex = 0; inputIndex < nextLayer.nodeCount; inputIndex++)
                {
                    int weightIndex = NodeToNodeWeightIndex(nodeIndex, inputIndex);
                    nextLayer.inputs[inputIndex] += activation.activationFunc(inputs[nodeIndex]) * weights[weightIndex];
                }
            }
            nextLayer.Predict();
        }

        public virtual void Backpropagate(float[] activationToCostChange, float learningRate)
        {
            if (activationToCostChange.Length != nextLayer.nodeCount)
            {
                throw new Exception("Input exceeds or is lower than node count!");
            }
            float[] newActivationValue = new float[activationToCostChange.Length];

            trainingCount++;

            float[] output = GetOutput();
            for (int i = 0; i < nodeCount; i++)
            {
                for (int j = 0; j < nextLayer.nodeCount; j++)
                {
                    deltaWeights[NodeToNodeWeightIndex(i, j)] -= output[i] * nextLayer.activation.derivativeFunc(nextLayer.inputs[j]) * activationToCostChange[j] * learningRate;
                }
            }
            for (int i = 0; i < nextLayer.nodeCount; i++)
            {
                deltaBiases[i] -= nextLayer.activation.derivativeFunc(nextLayer.inputs[i]) * activationToCostChange[i] * learningRate;
            }

            if (previousLayer == null) return;

            float[] updatedChange = new float[nodeCount];

            for (int i = 0; i < nodeCount; i++)
            {
                for (int j = 0; j < nextLayer.nodeCount; j++)
                {
                    updatedChange[i] += weights[NodeToNodeWeightIndex(i, j)] * nextLayer.activation.derivativeFunc(nextLayer.inputs[j]) * activationToCostChange[j];
                }
            }

            previousLayer.Backpropagate(updatedChange, learningRate);
        }

        protected int NodeToNodeWeightIndex(int thisLayerIndex, int nextLayerIndex)
        {
            return thisLayerIndex * nextLayer.nodeCount + nextLayerIndex;
        }
    }

    public class InputLayer : Layer
    {
        public InputLayer(int nodeCount) : base(nodeCount, Activation.Sigmoid) { }
        
        public virtual void FeedInput(float[] input)
        {
            if (input.Length != nodeCount)
            {
                throw new Exception("Input exceeds or is lower than node count!");
            }
            else
            {
                input.CopyTo(inputs, 0);
            }
        }
    }

    public class OutputLayer : Layer
    {
        public OutputLayer(int nodeCount) : base(nodeCount, Activation.Sigmoid) { }

        public virtual float CalculateCost(float[] intendedValue)
        {
            if (intendedValue.Length != nodeCount)
            {
                throw new Exception("Input exceeds or is lower than node count!");
            }

            float[] output = GetOutput();
            float cost = 0;

            for (int i = 0; i < nodeCount; i++)
            {
                float costOfI = output[i] - intendedValue[i];
                cost += costOfI * costOfI;
            }

            return cost;
        }

        public override void Backpropagate(float[] targetValue, float learningRate)
        {
            if (targetValue.Length != nodeCount)
            {
                throw new Exception("Input exceeds or is lower than node count!");
            }

            float[] output = GetOutput();
            float[] layerChange = new float[nodeCount];
            for (int i = 0; i < nodeCount; i++)
            {
                float costOfLayer = output[i] - targetValue[i];
                layerChange[i] = 2 * costOfLayer;
            }

            previousLayer.Backpropagate(layerChange, learningRate);
        }
    }
}
