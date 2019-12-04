public class examples
{
    public static void main(String[] args)
    {
        /* XOR function example*/

        
        NeuralNetwork nn = new NeuralNetwork(0.1, 200);
        nn.addInputLayer(2, 5, "sigmoid");
        nn.addLayer(1, "relu");

        double[][] input_batch = new double[][]{
            {0, 0},
            {0, 1},
            {1, 0},
            {1, 1}
        };

        double[][] output_batch = new double[][]{
            {0},
            {1},
            {1},
            {0}
        };

        
        nn.train(input_batch, output_batch, 2000, "sgd");
        

        /******  Larger Network example  ******/
        
        /*
        NeuralNetwork nn = new NeuralNetwork(0.01, 5);
        nn.addInputLayer(5, 20, "tanh");
        nn.addLayer(20, "tanh");
        nn.addLayer(20, "tanh");
        nn.addLayer(20, "tanh");
        nn.addLayer(1, "tanh");

        double[][] input_batch = new double[][]{
            {-0.3, 0.5, 1, 0.3, 0.6},
            {0.1, 0.8, -0.9, -0.3, 0.2},
            {-0.6, -0.3, -0.5, -0.9, -0.3},
            {0.25, 0.45, -0.1, -0.1, 0.44},
            {0.66, 0.91, 0.33, -0.3, -0.5},
            {0.90, 0.31, -0.1, -0.2, -0.6},
            {-0.6, -0.7, 0.8, 0.9, 1}
        };

        double[][] output_batch = new double[][]{
            {0},
            {0.5},
            {0.7},
            {0.4},
            {-0.3},
            {-0.5},
            {0.9}
        };

        nn.train(input_batch, output_batch, 2000, "sgd");
        */

        /******  Linear output layer example  ******/

        /*
        NeuralNetwork nn = new NeuralNetwork(0.01, 200);
        nn.addInputLayer(3, 5, "relu");
        nn.addLayer(3, "relu");
        nn.addLayer(2, "linear");

        double[][] input_batch = new double[][]{
            {0, 0.5, -0.5},
            {0.9, -0.9, 0.1},
            {0.6, 0.1, -0.4},
            {-0.5, -0.1, 0.25},
            {-0.13, 0.52, 0.66}
        };

        double[][] output_batch = new double[][]{
            {0, 10},
            {2, 6},
            {3, -1},
            {5, -7},
            {8, 8}
        };

        nn.train(input_batch, output_batch, 2000, "sgd");
        */

        /******  Learn A XOR B XOR C example  ******/

        /*
        NeuralNetwork nn = new NeuralNetwork(0.01, 200);
        nn.addInputLayer(3, 8, "relu");
        nn.addLayer(5, "relu");
        nn.addLayer(1, "linear");

        double[][] input_batch = new double[][]{
            {0, 0, 0},
            {0, 0, 1},
            {0, 1, 0},
            {0, 1, 1},
            {1, 0, 0},
            {1, 0, 1},
            {1, 1, 0},
            {1, 1, 1}
        };

        double[][] output_batch = new double[][]{
            {0},
            {1},
            {1},
            {0},
            {1},
            {0},
            {0},
            {1}
        };

        nn.train(input_batch, output_batch, 2200, "sgd");
        /*

        /* Prints outputs for the example */
        System.out.println("\n\nNote small numbers will appear in scientific notation.");
        for (double[] input : input_batch)
            nn.FeedForward(input, true);

    }
}