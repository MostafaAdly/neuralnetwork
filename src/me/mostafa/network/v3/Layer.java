package me.mostafa.network.v3;

import lombok.Data;

@Data
public class Layer {

    private Neuron[] neurons;
    private double[][] weights; // holds current neurons to previous neurons weights

    public Layer(int neurons, int inNeurons) {
        this.neurons = new Neuron[neurons];
        this.weights = MathTools.createRandomArray(neurons, inNeurons, NetworkConstants.WEIGHTS_LOWER_BOUND, NetworkConstants.WEIGHTS_UPPER_BOUND);
        for (int i = 0; i < neurons; i++)
            this.neurons[i] = new Neuron(NetworkConstants.BIASES_LOWER_BOUND, NetworkConstants.BIASES_UPPER_BOUND);
    }

    public double[] outputsToArray() {
        double[] outputs = new double[neurons.length];
        for (int i = 0; i < outputs.length; i++)
            outputs[i] = neurons[i].getValue();
        return outputs;
    }
}
