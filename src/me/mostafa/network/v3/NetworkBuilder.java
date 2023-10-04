package me.mostafa.network.v3;

import java.util.LinkedList;

public class NetworkBuilder {

    private String ACTIVATION_FUNCTION = "sigmoid", NETWORK_ID;
    private LinkedList<Integer> network_sizes = new LinkedList<>();

    public NetworkBuilder addLayer(int neurons) {
        this.network_sizes.add(neurons);
        return this;
    }

    public NetworkBuilder addLayers(int amountOfLayers, int dense) {
        for (int i = 0; i < amountOfLayers; i++)
            this.network_sizes.add(dense);
        return this;
    }

    public NetworkBuilder setActivationFunction(String function) {
        this.ACTIVATION_FUNCTION = function;
        return this;
    }

    public Network build() {
        Network network = new Network(network_sizes.toArray(new Integer[0]));
        network.setACTIVATION_FUNCTION(ACTIVATION_FUNCTION);
        return network;
    }
}
