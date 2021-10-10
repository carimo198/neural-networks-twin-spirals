# neural-networks-twin-spirals
This is the repository for the neural networks I developed to train two models on the famous Two Spirals Problem (Lang &amp; Witbrock 1988). The code from spiral_main.py loads the training data from spirals.csv, and applies the specified model and produces a graph of the resulting function, along with the data. The code for the models (using PyTorch) are in spiral.py which include:

- ***PolarNet*** - the cartesian (x,y) input in spirals.csv is converted to polar coordinates (r,a) with r=sqrt(x*x + y*y), a=atan2(y,x). Next, (r,a) is fed into a fully connected neural network with one hidden layer using tanh activation, followed by a single output using sigmoid activation. Use the following code in the command line to run the model: python3 spiral_main.py --net polar --hid 10
- ***RawNet*** - which operates on the raw input (x,y) without converting to polar coordinates and consisting of two fully connected hidden layers with tanh activation, plus the output layer, with sigmoid activation. Different hyperparameter values were experimented with. Use the following code to run the model: python3 spiral_main.py --net raw. You can use --hid, --init, --epoch, to experiment with different hyperparameter values.

Refer to spirals_report.pdf for a discussion on the two implemented models.
