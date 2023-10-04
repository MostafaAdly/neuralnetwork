package me.mostafa.network.v3;

import lombok.Data;

import java.util.UUID;

@Data
public class Neuron {

    private String id = UUID.randomUUID().toString().split("-")[0];
    private double value, bias, error, derivative;
//    private ArrayList<Connection> incomingConnections = new ArrayList<>(), outgoingConnections = new ArrayList<>();

    public Neuron(double lower_bias_bound, double upper_bias_bound) {
        this.bias = MathTools.randomValue(lower_bias_bound, upper_bias_bound);
    }
}
