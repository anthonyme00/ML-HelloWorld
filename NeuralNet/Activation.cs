using System;
using static System.MathF;

namespace MachineLearning.NeuralNet
{
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
    }
}
