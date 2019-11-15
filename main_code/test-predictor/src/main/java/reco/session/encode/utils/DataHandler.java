package reco.session.encode.utils;

import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

public class DataHandler {

    private boolean shuffle;
    private int batchSize;

    public DataHandler(int batchSize, boolean shuffle) {
        this.batchSize = batchSize;
        this.shuffle = shuffle;
    }

    /**
     * Creates a DataSetIterator for the training purposes
     * @param trainVectors
     * @param batchSize
     * @return DataSetIterator
     */
    public DataSetIterator generateDataSetIterator(List<double[]> trainVectors) {
        Collection<DataSet> list = new ArrayList<>();

        for (double[] v : trainVectors){
            INDArray in = Nd4j.create(v);
            INDArray out = Nd4j.create(v);
            list.add(new DataSet(in, out));
        }

        if (shuffle) {
            Collections.shuffle(trainVectors);
        }

        return new ListDataSetIterator(list, batchSize);
    }
    
    public DataSetIterator generateDataSetIterator(Map<String, List<double[]>> trainVectors, int itemCount) {
        Collection<DataSet> list = new ArrayList<>();

        for (String sessionId : trainVectors.keySet()) {
            List<double[]> sessionSequence = trainVectors.get(sessionId);
            // INDArray constructedArray = Nd4j.create(new int[]{numExamples, numFeatures, tsLength});
            INDArray in = Nd4j.zeros(sessionSequence.size(), itemCount);
            INDArray out = Nd4j.zeros(sessionSequence.size(), itemCount);

            for (int inIndex = 0; inIndex < sessionSequence.size(); inIndex++) {
                double[] inputVector = sessionSequence.get(inIndex);
                double[] outputVector = sessionSequence.get(inIndex);
                
                for (int i = 0; i < inputVector.length; i++) {
                    if (inputVector[i] == 1.0) {
                        in.putScalar(inIndex, i, 1.0);
                        break;
                    }
                }
                
                for (int i = 0; i < outputVector.length; i++) {
                    if (outputVector[i] == 1.0) {
                        out.putScalar(inIndex, i, 1.0);
                        break;
                    }
                }
            }
            
            list.add(new DataSet(in, out));

        }

        return new ListDataSetIterator(list, batchSize);
    }
    

    public static double[] extractVector(INDArray in) {
        int numRows = in.rows();
        int numColumns = in.columns();

        double[] extractedVector = new double[numRows * numColumns];

        for (int i = 0; i < numRows; i++) {
            for( int j = 0; j < numColumns; j++ ){
                double value = in.getDouble(i, j);
                extractedVector[(numColumns * i) + j] = value;
            }
        }

        return extractedVector;
    }
    
    /**
     * Sorts map based on the values
     * @param map
     * @return
     */
    public static <K, V extends Comparable<V>> TreeMap<K, V> sortByValues(final Map<K, V> map) {
        Comparator<K> valueComparator = new Comparator<K>() {
            public int compare(K o1, K o2) {
                int compare = map.get(o2).compareTo(map.get(o1));
                if (compare == 0) 
                    return -1;
                else 
                    return compare;
            }
        };

        TreeMap<K, V> sortedByValues = new TreeMap<K, V>(valueComparator);
        sortedByValues.putAll(map);
        return sortedByValues;
    }
}
