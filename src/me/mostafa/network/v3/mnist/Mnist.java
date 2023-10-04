package me.mostafa.network.v3.mnist;


import me.mostafa.network.v3.MathTools;
import me.mostafa.network.v3.Network;
import me.mostafa.network.v3.NetworkBuilder;
import me.mostafa.network.v3.TrainingSet;

import java.io.File;

/**
 * Created by Luecx on 10.08.2017.
 */
public class Mnist {

    public static void main(String[] args) {
        Network network = new NetworkBuilder().addLayer(784).addLayer(70).addLayer(35).addLayer(10).setActivationFunction("sigmoid").build();
        createTrainSet(network.getTrainingSet(), 0, 4999);
        trainData(network, 100, 50, 100);

        network.save();

        network.setTrainingSet(new TrainingSet(network));
        createTrainSet(network.getTrainingSet(), 5000, 9999);
        testTrainSet(network, 200);
    }

    public static void createTrainSet(TrainingSet set, int start, int end) {

        try {

            String path = new File("").getAbsolutePath();

            MnistImageFile m = new MnistImageFile(path + "/res/trainImage.idx3-ubyte", "rw");
            MnistLabelFile l = new MnistLabelFile(path + "/res/trainLabel.idx1-ubyte", "rw");

            for (int i = start; i <= end; i++) {
                if (i % 100 == 0) {
                    System.out.println("prepared: " + i);
                }

                double[] input = new double[28 * 28];
                double[] output = new double[10];

                output[l.readLabel()] = 1d;
                for (int j = 0; j < 28 * 28; j++) {
                    input[j] = (double) m.read() / (double) 256;
                }

                set.addDataSet(input, output);
                m.next();
                l.next();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static void trainData(Network net, int epochs, int loops, int batch_size) {
        for (int e = 0; e < epochs; e++) {
            double mse = net.train(0.3, loops, batch_size);
            System.out.println(">>>>>>>>>>>>>>>>>>>>>>>>>   " + e + "   <<<<<<<<<<<<<<<<<<<<<<<<<< " + mse);
        }
    }

    public static void testTrainSet(Network net, int printSteps) {
        int correct = 0;
        for (int i = 0; i < net.getTrainingSet().size(); i++) {

            double highest = MathTools.indexOfHighestValue(net.run(net.getTrainingSet().getInput(i)));
            double actualHighest = MathTools.indexOfHighestValue(net.getTrainingSet().getOutput(i));
            if (highest == actualHighest) {

                correct++;
            }
            if (i % printSteps == 0) {
                System.out.println(i + ": " + (double) correct / (double) (i + 1));
            }
        }
        System.out.println("Testing finished, RESULT: " + correct + " / " + net.getTrainingSet().size() + "  -> " + (((double) correct / (double) net.getTrainingSet().size()) * 100) + " %");
    }
}
