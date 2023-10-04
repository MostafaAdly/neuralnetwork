package me.mostafa.network.v3;

import java.util.Random;

public class MathTools {

    private static Random random = new Random();

    public static double[] createRandomArray(int size, double lower_bound, double upper_bound) {
        if (size < 1) return null;

        double[] ar = new double[size];
        for (int i = 0; i < size; i++)
            ar[i] = randomValue(lower_bound, upper_bound);
        return ar;
    }

    public static double[][] createRandomArray(int sizeX, int sizeY, double lower_bound, double upper_bound) {
        if (sizeX < 1 || sizeY < 1) return null;
        double[][] ar = new double[sizeX][sizeY];
        for (int i = 0; i < sizeX; i++) {
            ar[i] = createRandomArray(sizeY, lower_bound, upper_bound);
        }
        return ar;
    }

    public static double randomValue(double lower_bound, double upper_bound) {
        return Math.random() * (upper_bound - lower_bound) + lower_bound;
    }

    public static Integer[] randomValues(int lower_bound, int upper_bound, int amount) {
        lower_bound--;

        if (amount > (upper_bound - lower_bound)) return null;

        Integer[] values = new Integer[amount];
        for (int i = 0; i < amount; i++) {
            int n = (int) (Math.random() * (upper_bound - lower_bound + 1) + lower_bound);
            while (containsValue(values, n)) {
                n = (int) (Math.random() * (upper_bound - lower_bound + 1) + lower_bound);
            }
            values[i] = n;
        }
        return values;
    }

    public static <T extends Comparable<T>> boolean containsValue(T[] ar, T value) {
        for (int i = 0; i < ar.length; i++)
            if (ar[i] != null && value.compareTo(ar[i]) == 0)
                return true;
        return false;
    }

    public static int indexOfHighestValue(double[] values) {
        int index = 0;
        for (int i = 1; i < values.length; i++)
            if (values[i] > values[index])
                index = i;
        return index;
    }

    public static double activate(double value, String function) {
        if (function == null) return value;
        switch (function.toLowerCase()) {
            case "sigmoid":
                return (1d / (1 + Math.exp(-value)));
            case "relu":
                return Math.max(0, value);
            default:
                break;
        }
        return value;
    }

    public static String formatMSE(double mse) {
        String str = String.valueOf(mse);
        return str.contains("E-") ? (str.substring(0, 5) + " *10^-" + str.split("E-")[1]) : str;
    }
}
