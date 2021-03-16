using System;
using static System.MathF;

namespace MachineLearning.NeuralNet
{
    static class Activation
    {
        public static Func<float, float> Relu = (x) => { return Max(0, x); };
        public static Func<float, float> Sigmoid = (x) => { return (1.0f / (1.0f + Exp(-x))); };
        public static Func<float, float> LinearClamped = (x) => { return Max(0, Min(1, x)); };
    }
}
