using System;
using static System.MathF;

namespace MachineLearning.NeuralNet
{
    public delegate float[] CustomActivation(float[] inputs);
    public delegate float[][] CustomDerivative(float[] inputs);
    public class CustomActivationSet
    {
        public CustomActivation activationFunc;
        public CustomDerivative derivativeFunc;
        public Func<float[], float[], float> costFunc;
    }

    public struct ActivationSet
    {
        public Func<float, float> activationFunc;
        public Func<float, float> derivativeFunc;
    }

    static class Activation
    {
        public static ActivationSet Pass = new ActivationSet() 
        { 
            activationFunc = (x) => { return x; }, 
            derivativeFunc = (x) => { return 1; } 
        };
        public static ActivationSet Relu = new ActivationSet()
        {
            activationFunc = (x) => { return Max(0, x); },
            derivativeFunc = (x) => { return x > 0 ? 1 : 0; }
        };
        public static ActivationSet Sigmoid = new ActivationSet()
        {
            activationFunc = (x) => { return (1.0f / (1.0f + Exp(-x))); },
            // For sigmoid, the derivative can be simplified to f(x)*(1-f(x))
            derivativeFunc = (x) => { return (1.0f / (1.0f + Exp(-x))) * (1.0f - (1.0f / (1.0f + Exp(-x)))); }
        };
        public static ActivationSet LinearClamped = new ActivationSet()
        {
            activationFunc = (x) => { return Max(0, Min(1, x)); },
            derivativeFunc = (x) => { return (x > 0 && x < 1) ? 1 : 0; }
        };

        public static CustomActivationSet SoftMax = new CustomActivationSet()
        {
            activationFunc = (inputs) =>
            {
                float[] activated = new float[inputs.Length];
                for (int i = 0; i < activated.Length; i++)
                {
                    activated[i] = Exp(inputs[i]);
                    float divisor = 0;
                    for (int j = 0; j < inputs.Length; j++)
                    {
                        divisor += Exp(inputs[j]);
                    }
                    activated[i] /= divisor;
                }
                return activated;
            },
            // https://aerinykim.medium.com/how-to-implement-the-softmax-derivative-independently-from-any-loss-function-ae6d44363a9d
            derivativeFunc = (inputs) =>
            {
                float[] activated = new float[inputs.Length];
                for (int i = 0; i < activated.Length; i++)
                {
                    activated[i] = Exp(inputs[i]);
                    float divisor = 0;
                    for (int j = 0; j < inputs.Length; j++)
                    {
                        divisor += Exp(inputs[j]);
                    }
                    activated[i] /= divisor;
                }


                float[][] jacobian_m = new float[inputs.Length][];
                for (int i = 0; i < inputs.Length; i++)
                {
                    jacobian_m[i] = new float[inputs.Length];
                    for (int j = 0; j < inputs.Length; j++)
                    {
                        if (i == j)
                        {
                            jacobian_m[i][j] = activated[i] * (1.0f - activated[i]);
                        }
                        else
                        {
                            jacobian_m[i][j] = -activated[i] * activated[j];
                        }
                    }
                }

                return jacobian_m;
            },
            costFunc = (inputs, targets) =>
            {
                float ret = 0;
                for (int i = 0; i < inputs.Length; i++)
                {
                    ret += targets[i] * Log(inputs[i] + (float)1e-8);
                }
                ret *= -1f/inputs.Length;
                return ret;
            }
        };
    }
}
