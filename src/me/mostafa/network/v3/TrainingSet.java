package me.mostafa.network.v3;

import java.util.HashMap;
import java.util.Objects;
import java.util.Random;

public class TrainingSet {

    private final Network networkV2;
    private final HashMap<double[], double[]> trainingData = new HashMap<>();

    public TrainingSet(Network network) {
        this.networkV2 = network;
    }

    public void addDataSet(double[] input, double[] target) {
        if (input.length != networkV2.getLayers()[0].getNeurons().length || target.length != networkV2.getLayers()[networkV2.getLayers().length - 1].getNeurons().length)
            return;
        trainingData.put(input, target);
    }

    public TrainingSet getBatch(int size) {
        if (size < 1 || size > size()) return this;
        TrainingSet batch = new TrainingSet(networkV2);
        for (int i : Objects.requireNonNull(MathTools.randomValues(0, size - 1, size)))
            batch.addDataSet(getInput(i), getOutput(i));
        return batch;
    }

    public double[] getInput(int index) {
        return (double[]) trainingData.keySet().toArray()[index];
    }

    public double[] getOutput(int index) {
        return (double[]) trainingData.values().toArray()[index];
    }

    public int size() {
        return this.trainingData.size();
    }

    public void createRandomDataSet(int amount) {
        // CREATING TRAINING SET
        Random random = new Random();
        for (int i = 1; i <= amount; i++) {
//            double[] inputs = {random.nextDouble() * random.nextInt(100), random.nextDouble() * random.nextInt(100)}, targets = new double[]{inputs[0] > inputs[1] ? 1 : 0, inputs[0] > inputs[1] ? 0 : 1};
            double[] inputs = {random.nextInt(1000), random.nextInt(1000)}, targets = {inputs[0] > 500 ? 1 : 0};
            addDataSet(inputs, targets);
        }
    }

}
