using System;
using System.Security.Cryptography;

namespace MachineLearning.NeuralNet
{
    public class Layer
    {
        public int nodeCount { get; private set; }
        public Func<float, float> activationFunction { get; private set; }

        protected float[] inputs;
        private float[] weights;
        private float[] biases;

        private float[] deltaWeights;
        private float[] deltaBiases;
        private int trainingCount;
        
        private Layer previousLayer;
        private Layer nextLayer;

        private static RandomNumberGenerator seedGenerator;

        public virtual float[] GetOutput()
        {
            float[] outputs = new float[nodeCount];
            inputs.CopyTo(outputs, 0);

            for (int i = 0; i < nodeCount; i++)
            {
                outputs[i] = activationFunction(outputs[i]);
            }

            return outputs;
        }

        public Layer(int nodeCount, Func<float, float> activationFunction)
        {
            if (seedGenerator == null)
            {
                seedGenerator = RandomNumberGenerator.Create();
            }
            this.nodeCount = nodeCount;
            this.activationFunction = activationFunction;
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
                    biases[i] = (float)(rand.NextDouble()-0.5) * 2f;
                    for (int j = 0; j < nodeCount; j++)
                    {
                        weights[i * nodeCount + j] = (float)(rand.NextDouble()-0.5)*2f;
                    }
                }
            }
            
            this.previousLayer = previousLayer;
            this.nextLayer = nextLayer;
        }

        public virtual void Backpropagate(float[] targetValue, float learningRate)
        {
            if (previousLayer == null) return;
            if (targetValue.Length != nodeCount)
            {
                throw new Exception("Input exceeds or is lower than node count!");
            }
            previousLayer.trainingCount++;

            float[] backpropagateTargetValue = new float[previousLayer.nodeCount];
            float[] myOutputs = GetOutput();

            for (int j = 0; j < nodeCount; j++)
            {
                float error = targetValue[j] - myOutputs[j];
                for (int i = 0; i < previousLayer.nodeCount; i++)
                {
                    int weightIndex = previousLayer.NodeToNodeWeightIndex(i, j);
                    float delta = previousLayer.weights[weightIndex] * error * learningRate;
                    previousLayer.deltaWeights[weightIndex] += delta;
                    backpropagateTargetValue[i] += delta/nodeCount;
                }

                previousLayer.deltaBiases[j] += error * learningRate;
            }

            previousLayer.Backpropagate(backpropagateTargetValue, learningRate);
        }

        public virtual void ApplyChanges()
        {
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] += deltaWeights[i]/trainingCount;
            }
            for (int i = 0; i < biases.Length; i++)
            {
                biases[i] += deltaBiases[i]/trainingCount;
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
                    int weightIndex = NodeToNodeWeightIndex(nodeIndex, inputIndex);//nodeIndex * nextLayer.nodeCount + inputIndex;
                    nextLayer.inputs[inputIndex] += inputs[nodeIndex] * weights[weightIndex];
                }
            }
            for (int nodeIndex = 0; nodeIndex < nextLayer.nodeCount; nodeIndex++)
            {
                nextLayer.inputs[nodeIndex] = nextLayer.activationFunction(nextLayer.inputs[nodeIndex]);
            }
            nextLayer.Predict();
        }

        private int NodeToNodeWeightIndex(int thisLayerIndex, int nextLayerIndex)
        {
            return thisLayerIndex * nextLayer.nodeCount + nextLayerIndex;
        }
    }

    public class InputLayer : Layer
    {
        public InputLayer(int nodeCount) : base(nodeCount, Activation.Relu) { }
        
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
        public OutputLayer(int nodeCount) : base(nodeCount, Activation.LinearClamped) { }

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
    }
}
