# ML-HelloWorld
A backpropagation neural network implementation C# for learning the hello world of machine learning (MNIST Dataset).
Should be flexible enough for other datasets with small modifications.

Built from scratch only using the .NET Core library.

# Running
- Run `git clone https://github.com/anthonyme00/ML-HelloWorld/`
- Get the MNIST data from http://yann.lecun.com/exdb/mnist/
- Open the .sln file with Visual Studio (Programmed on Visual Studio Community 2019, later version might work too)
- Build the project
- Navigate to the binary folder (usually it's bin/Release/netcoreapp3.1), Create a new folder "dataset"
- Extract the MNIST data and put it inside the "dataset" folder
- Folder should look like 
  ```
  dataset/
  --  t10k-images.idx3-ubyte
  --  t10k-labels.idx1-ubyte
  --  train-images.idx3-ubyte
  --  train-labels.idx1-ubyte
  ```
- Run

# Note
The accuracy of this particular implementation seems to hover around 70%-90% depending on the learningStep variable.
An epoch of around 120 seems enough, and batch count of 100 seems to be good enough.

For the hidden layers, Relu activation is much much faster compared to sigmoid. at around 1200 epoch, the cost still hovers around 0.9.

Feel free to play around with the settings. Change it in Program.cs, it should be fairly self explanatory.
