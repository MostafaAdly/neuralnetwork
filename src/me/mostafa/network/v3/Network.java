package me.mostafa.network.v3;

import com.google.gson.Gson;
import lombok.Data;

import java.io.File;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;

@Data
public class Network {

    private Layer[] layers;
    private Gson gson = new Gson();
    private TrainingSet trainingSet = new TrainingSet(this);
    private String NETWORK_VERSION = "1.0", NETWORK_ID = UUID.randomUUID().toString().split("-")[0], ACTIVATION_FUNCTION = "sigmoid";


    public static void main(String[] args) {
        Network net = new NetworkBuilder().addLayer(2).addLayers(5, 20).addLayer(1).build();
        net.getTrainingSet().createRandomDataSet(50000);
        net.train(0.3, 100000, 5, true);
        System.out.println(Arrays.toString(net.run(2, 5)));
        System.out.println(Arrays.toString(net.run(602, 8)));
    }

    public String trainThenSaveNetwork() {
        Network net = new Network(2, 20, 20, 20, 20, 20, 2);


        // CREATING TRAINING SET
        Random random = new Random();
        for (int i = 1; i <= 100000; i++) {
            double[] inputs = {random.nextDouble() * random.nextInt(100), random.nextDouble() * random.nextInt(100)}, targets = new double[]{inputs[0] > inputs[1] ? 1 : 0, inputs[0] > inputs[1] ? 0 : 1};
            net.getTrainingSet().addDataSet(inputs, targets);
        }
        // START TRAINING
        net.train(0.8, 5000, 5, true);

        // SAVING
        return net.save("saved_networks/");
    }

    public Network(Integer... network_sizes) {
        layers = new Layer[network_sizes.length];

        // creating layers
        for (int index = 0; index < network_sizes.length; index++)
            layers[index] = new Layer(network_sizes[index], index == 0 ? 0 : network_sizes[index - 1]);
    }

    public double MSE(boolean feedForward, double[] input, double[] target) {
        if (input.length != layers[0].getNeurons().length || target.length != layers[layers.length - 1].getNeurons().length)
            return 0;
        if (feedForward)
            run(input);
        double v = 0;
        for (int i = 0; i < target.length; i++) {
            double neuronValue = layers[layers.length - 1].getNeurons()[i].getValue();
            v += (target[i] - neuronValue) * (target[i] - neuronValue);
        }
        return v / (2d * target.length);
    }

    public double train(double[] inputs, double[] targets, double learningRate, int iterations) {
        if (inputs.length != layers[0].getNeurons().length || targets.length != layers[layers.length - 1].getNeurons().length)
            return 0;
        for (int i = 0; i < iterations; i++) {
            run(inputs);
            backPropagation(targets);
            updateWeights(learningRate);
        }
        return MSE(false, inputs, targets);
    }

    public double train(TrainingSet set, double learningRate, int batchSize) {
        double mse = -1;
        TrainingSet batch = set.getBatch(batchSize);
        for (int j = 0; j < batch.size(); j++)
            mse = train(batch.getInput(j), batch.getOutput(j), learningRate, NetworkConstants.ITERATIONS_PER_BATCH);
        return mse;

    }

    public double train(double learningRate, int iterations, int batchSize) {
        return train(learningRate, iterations, batchSize, false);
    }

    public double train(double learningRate, int iterations, boolean log) {
        return train(learningRate, iterations, (int) Math.sqrt(iterations), log);
    }

    public double train(double learningRate, int iterations, int batchSize, boolean log) {
        double mse = -1;
        long time = System.currentTimeMillis();
        for (int i = 1; i <= iterations; i++) {
            mse = train(getTrainingSet().getBatch(batchSize), learningRate, batchSize);
            if (log && i % 500 == 0) {
                double timeTook = (System.currentTimeMillis() - time) / 1000d, timeLeft = ((timeTook * (iterations - i) / i));
                log("Training: epoch[" + i + " : (" + String.format("%.3f", ((double) i / (double) iterations) * 100d) + "%)], mse[" +
                        MathTools.formatMSE(mse) + "], batch_size[" + batchSize + "], time_took[" + String.format("%.2f", timeTook) + "sec], time_left[" + String.format("%.2f", timeLeft) + "sec]");
            }
        }
        return mse;
    }

    public double[] run(double... inputs) {
        if (inputs.length != layers[0].getNeurons().length) return null;
        for (int i = 0; i < layers[0].getNeurons().length; i++)
            layers[0].getNeurons()[i].setValue(inputs[i]);

        for (int layer = 1; layer < layers.length; layer++) {
            Layer currentLayer = layers[layer];
            for (int neuron = 0; neuron < currentLayer.getNeurons().length; neuron++) {
                Neuron currentNeuron = currentLayer.getNeurons()[neuron];
                double total = currentNeuron.getBias();
                for (int prevNeuron = 0; prevNeuron < currentLayer.getWeights()[neuron].length; prevNeuron++)
                    total += layers[layer - 1].getNeurons()[prevNeuron].getValue() * currentLayer.getWeights()[neuron][prevNeuron];
                currentNeuron.setValue(MathTools.activate(total, ACTIVATION_FUNCTION));
                currentNeuron.setDerivative(currentNeuron.getValue() * (1 - currentNeuron.getValue()));
            }
        }
        return layers[layers.length - 1].outputsToArray();
    }

    private void backPropagation(double[] targets) {
        if (targets.length != layers[layers.length - 1].getNeurons().length) return;
        for (int i = 0; i < layers[layers.length - 1].getNeurons().length; i++) {
            Neuron neuron = layers[layers.length - 1].getNeurons()[i];
            neuron.setError((neuron.getValue() - targets[i]) * neuron.getDerivative());
        }

        for (int layer = layers.length - 2; layer > 0; layer--) {
            Layer currentLayer = layers[layer];
            for (int neuron = 0; neuron < layers[layer].getNeurons().length; neuron++) {
                Neuron currentNeuron = currentLayer.getNeurons()[neuron];
                double sum = 0;
                for (int nextNeuron = 0; nextNeuron < layers[layer + 1].getWeights().length; nextNeuron++)
                    sum += layers[layer + 1].getWeights()[nextNeuron][neuron] * layers[layer + 1].getNeurons()[nextNeuron].getError();
                currentNeuron.setError(sum * currentNeuron.getDerivative());
            }
        }
    }

    private void updateWeights(double learningRate) {
        for (int layer = 1; layer < layers.length; layer++) {
            Layer currentLayer = layers[layer];
            for (int neuron = 0; neuron < layers[layer].getNeurons().length; neuron++) {
                Neuron currentNeuron = currentLayer.getNeurons()[neuron];
                double delta = -learningRate * currentNeuron.getError();
                currentNeuron.setBias(currentNeuron.getBias() + delta);
                for (int prevNeuron = 0; prevNeuron < currentLayer.getWeights()[neuron].length; prevNeuron++)
                    currentLayer.getWeights()[neuron][prevNeuron] += delta * layers[layer - 1].getNeurons()[prevNeuron].getValue();
            }
        }
    }

    public String save() {
        return save("saved_networks/");
    }

    public String save(String path) {
        try {
            File file = new File(path, NETWORK_ID + ".network"), parent = new File(file.getParent());
            if (file.exists()) {
                System.out.println("Error while saving network[" + NETWORK_ID + "]: file already exists.");
                return null;
            }
            if (!parent.exists() && !parent.mkdirs()) return null;
            if (file.createNewFile()) {
                PrintWriter writer = new PrintWriter(file, StandardCharsets.UTF_8);
                String network = toString();
                writer.println(network);
                writer.close();
                System.out.println("Network[" + NETWORK_ID + "] was saved into file[" + file.getAbsolutePath() + "].");
                return file.getAbsolutePath();
            }
            System.out.println("Network[" + NETWORK_ID + "] was not saved.");
        } catch (Exception e) {
            e.printStackTrace();
            System.out.println("Error occurred while saving network[" + NETWORK_ID + "].");
        }
        return null;
    }

    public static Network load(String path) {
        return load(new File(path));
    }

    public static Network load(String path, String networkId) {
        return load(new File(path, networkId));
    }

    public static Network load(File file) {
        Network network = null;
        try {
            if (!file.exists()) {
                System.out.println("Error while loading network: file does not exist.");
                return null;
            }
            Scanner scanner = new Scanner(file);
            StringBuilder lines = new StringBuilder();
            while (scanner.hasNextLine()) lines.append(scanner.nextLine());
            return new Gson().fromJson(lines.toString(), NetworkData.class).load();
        } catch (Exception e) {
            e.printStackTrace();
            System.out.println("Error occurred while loading network.");
        }
        return network;
    }

    private void log(String text) {
        System.out.println(getCurrentDate() + " - " + text);
    }

    private String getCurrentDate() {
        return DateTimeFormatter.ofPattern("yyyy/MM/dd HH:mm:ss").format(LocalDateTime.now());
    }

    @Override
    public String toString() {
        return gson.toJson(new NetworkData(this));
    }
}
