package me.mostafa.network.v3;


import lombok.Data;

@Data
public class NetworkData {

    private Layer[] layers;
    private String NETWORK_VERSION, NETWORK_ID;

    public NetworkData(Network network) {
        this.layers = network.getLayers();
        this.NETWORK_VERSION = network.getNETWORK_VERSION();
        this.NETWORK_ID = network.getNETWORK_ID();
    }

    public Network load(){
        Network network = new Network(1, 1, 1);
        network.setLayers(layers);
        network.setNETWORK_VERSION(getNETWORK_VERSION());
        network.setNETWORK_ID(getNETWORK_ID());
        return network;
    }
}
