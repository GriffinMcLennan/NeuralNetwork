# Neural Network Class

## Creating a Neural Network:

* First create an instance of the Neural Network class specifying the learning rate and how often to print the cost function.

    ```Java
    NeuralNetwork nn = new NeuralNetwork(learning_rate, print_every_x_epochs);
    ```
* Next an input layer must be added specifying the input shape, output shape of the current layer, and the activation function.

    ```Java
    nn.addInputLayer(inputShape, outputShape, activation_function);
    ```
* As many layers as desired can now be created using the addLayer method:

    ```Java
    nn.addLayer(outputShape, activation_function);
    ```

    The output layer is also added via the addLayer method.

    ## Activation Functions:

    * sigmoid
    * tanh
    * relu
    * linear

## Training the Neural Network:

* Use the _train_ method to pass the input and output data along with parameters such as the number of epochs and the algorithm.

    ```Java
    nn.train(input_data, output_data, num_epochs, algorithm)
    ```


    ### Algorithms:

    * Gradient Descent, ```set algorithm="gd"```
    * Stochastic Gradient Descent, ```set algorithm="sgd"```

## Saving the Neural Network:

* Once the neural network has been trained the weights can be stored in a text file for future use via the _saveNetwork_ method and specifying the name of the file (extension added automatically). 

    ```Java
    nn.saveNetwork(neural_network_name)
    ```

## Loading a saved Neural Network:

* Once you've saved a neural network you can reload the neural netework using the _loadNetwork_ method:

    ```Java
    NeuralNetwork nn = new NeuralNetwork(learning_rate, print_every_x_epochs)
    nn.loadNetwork(neural_network_name)
    ```
* As shown in the example above the individual layers don't need to be added before calling _loadNetwork_.

## Examples:

 * Examples of setting up and training various Neural Networks can be found in the _examples.java_ file.
 
## Demo:

![](https://raw.githubusercontent.com/GriffinMcLennan/NeuralNetwork/master/demo.gif)
