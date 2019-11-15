package reco.session.encode.impl.baseline;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.atomic.AtomicInteger;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import reco.session.encode.utils.DataHandler;
import reco.session.encode.utils.DataLoader;
import reco.session.encode.utils.PropertyHolder;

/**
 * 
 * Bayes baseline approach as described in the paper
 * 
 */
public class PredictionBayesRunner implements Runnable {

    private static final Logger logger = LoggerFactory.getLogger(PredictionBayesRunner.class);

    // default values
    private int cutOff = 10;
    // constructor values
    private String predictPath;
    private String delimiter;
    private String modelName;


    private String testPath;


    public PredictionBayesRunner() {
        this.delimiter = PropertyHolder.getInstance().getDelimiter();
        this.testPath = PropertyHolder.getInstance().getTestFile();
        this.predictPath = PropertyHolder.getInstance().getPredictPath();
        this.modelName = PropertyHolder.getInstance().getModelName();
        this.cutOff = PropertyHolder.getInstance().getCutOff();
    }

    @Override
    public void run() {
        logger.info("Initializing training data...");
        DataLoader loader = new DataLoader();
        loader.init();

        Map<String, List<String>> testSessions = loader.mapTestSessionSequence(testPath);
        logger.info("Loaded " + testSessions.size() + " sessions from test.");

        runPrediction(modelName, testSessions, loader);
    }

    protected void runPrediction(String currentModelName, Map<String, List<String>> testSessions, DataLoader loader) {
        logger.info("Predicting with mean vector for model: " + currentModelName);
        List<String> predictionOutput = Collections.synchronizedList(new ArrayList<>());

        Map<String, List<String>> sessionItemTrainMap = loader.mapTrainSessionSequence();


        int cores = Runtime.getRuntime().availableProcessors();
        ForkJoinPool mainPool = new ForkJoinPool(cores/4);
        ForkJoinPool secondaryPool = new ForkJoinPool(cores/2);

        Map<String, Set<String>> itemSessionMap = constructItemSessionMap(sessionItemTrainMap, mainPool);
        Map<String, Double> itemProbabilities = getItemProbbabilities(itemSessionMap, sessionItemTrainMap.size(), mainPool);

        AtomicInteger counter = new AtomicInteger(0);

        try {
            mainPool.submit(() ->
            testSessions.keySet().stream().parallel().forEach( (sessionId) -> {
                // test one session and its subsequences
                List<String> sessionSequence = testSessions.get(sessionId);

                List<String> currentTestSequence = new ArrayList<>();
                double pxHistory = 1.0;
                // go from item to item
                // leave last one for testing
                for (int sequenceIndex = 0; sequenceIndex < sessionSequence.size() - 1; sequenceIndex++) {
                    String inputItem = sessionSequence.get(sequenceIndex);
                    currentTestSequence.add(inputItem);
                    // calculate current denominator
                    pxHistory *= itemProbabilities.get(inputItem);

                    final double denominator = pxHistory;
                    Map<String, Double> candidateProbabilities = new ConcurrentHashMap<>();
                    try {
                        secondaryPool.submit(() ->
                        itemProbabilities.keySet().stream().parallel().forEach( (xi) -> {
                            // only for items not contained in history
                            if (! currentTestSequence.contains(xi)) {
                                double pxi = itemProbabilities.get(xi);

                                Set<String> candidateSessions = itemSessionMap.get(xi);

                                double numerator = 1.0;

                                for (String historyItem : currentTestSequence) {
                                    // Calculate p(history_item | xi) 
                                    Set<String> historySessions = new HashSet<>(itemSessionMap.get(historyItem));
                                    historySessions.retainAll(candidateSessions);
                                    // P(A ∩ B)
                                    double pHistGivenXi = historySessions.size() / ((double) sessionItemTrainMap.size());
//                                    if (pHistGivenXi == 0.0) {
//                                        pHistGivenXi = 0.00001;
//                                    }
                                    // p(A|B) = P(A ∩ B) / P(B)
                                    pHistGivenXi = pHistGivenXi / pxi;
                                    numerator *= pHistGivenXi;
                                }
                                numerator *= pxi;

                                double candidateProbability = numerator / denominator;
                                candidateProbabilities.put(xi, candidateProbability);
                            }
                        })).get();
                    } catch (InterruptedException | ExecutionException e) {
                        logger.error(e.getMessage(), e);
                    }
                    List<String> recItems = getRecs(candidateProbabilities);

                    String predictedItems = String.join(",", recItems);


                    List<String> remainingSequence = sessionSequence.subList(sequenceIndex + 1, sessionSequence.size());

                    String joinedInputItems = String.join(",", currentTestSequence);
                    int inputLength = currentTestSequence.size();
                    String joinedRemainingItems = String.join(",", remainingSequence);
                    int remainingLength = remainingSequence.size();

                    String position = "MID";
                    if (inputLength == 1) {
                        position = "FIRST";
                    }
                    if (remainingLength == 1) {
                        position = "LAST";
                    }

                    StringBuilder sb = new StringBuilder();
                    sb.append(delimiter);
                    sb.append(sessionId);
                    sb.append(delimiter);
                    sb.append(joinedInputItems);
                    sb.append(delimiter);
                    sb.append(currentTestSequence.size());
                    sb.append(delimiter);
                    sb.append(position);
                    sb.append(delimiter);
                    sb.append(joinedRemainingItems);
                    sb.append(delimiter);
                    sb.append(remainingSequence.size());
                    sb.append(delimiter);
                    sb.append(predictedItems);
                    sb.append("\n");

                    predictionOutput.add(sb.toString());

                    int currentCounter = counter.incrementAndGet();
                    if (currentCounter % 1000 == 0) {
                        logger.info("Made predictions for " + currentCounter + " entries...");
                    }
                }
            })).get();
        } catch (InterruptedException | ExecutionException e) {
            logger.error(e.getMessage(), e);
        }

        storePredictions(currentModelName, predictionOutput);
    }

    protected void storePredictions(String currentModelName, List<String> predictionOutput) {
        String storePath = predictPath + "test_d14_bayes_" + currentModelName + ".csv";

        DataLoader.deleteExisting(storePath);
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(storePath))) {
            StringBuilder headerBuilder = new StringBuilder();
            headerBuilder.append(delimiter);
            headerBuilder.append("session_id");
            headerBuilder.append(delimiter);
            headerBuilder.append("input_items");
            headerBuilder.append(delimiter);
            headerBuilder.append("input_count");
            headerBuilder.append(delimiter);
            headerBuilder.append("position");
            headerBuilder.append(delimiter);
            headerBuilder.append("remaining_items");
            headerBuilder.append(delimiter);
            headerBuilder.append("remaining_count");
            headerBuilder.append(delimiter);
            headerBuilder.append("predictions");
            headerBuilder.append("\n");

            bw.write(headerBuilder.toString());

            for (String prediction : predictionOutput) {
                bw.write(prediction);
            }
            logger.info("Stored test predictions to: " + storePath);
        } catch (IOException e) {
            logger.error("Failed to store predictions for test data.", e);
        }
    }

    protected List<String> getRecs(Map<String, Double> itemProbabilities) {
        TreeMap<String, Double> sortedItemProbailities = DataHandler.sortByValues(itemProbabilities);

        List<String> items = new ArrayList<>(sortedItemProbailities.keySet());

        if (items.size() >= cutOff) {
            return items.subList(0,  cutOff);
        } else {
            return items;
        }
    }


    protected Map<String, Set<String>> constructItemSessionMap(
            Map<String, List<String>> trainHistory,
            ForkJoinPool mainPool) {
        Map<String, Set<String>> itemSessionsMapping = new ConcurrentHashMap<>();
        try {
            logger.info("Constructing item-sessions map...");
            mainPool.submit(() ->
            trainHistory.keySet().stream().parallel().forEach( (sessionId) -> {
                List<String> sessionItems = trainHistory.get(sessionId);

                sessionItems.forEach( (item) -> {
                    itemSessionsMapping.compute(item, (key, value) -> {
                        Set<String> sessions = value;
                        if (value == null) {
                            sessions = new HashSet<>();
                        }
                        sessions.add(sessionId);
                        return sessions;
                    });
                });
            })).get();
        } catch (InterruptedException | ExecutionException e) {
            logger.error(e.getMessage(), e);
        }

        return itemSessionsMapping;
    }

    protected Map<String, Double> getItemProbbabilities(
            Map<String, Set<String>> itemSessionMap,
            Integer numOfTrainSessions,
            ForkJoinPool mainPool) {
        Map<String, Double> itemProbabilities = new ConcurrentHashMap<>();
        try {
            logger.info("Constructing item-probability map...");
            mainPool.submit(() ->
            itemSessionMap.keySet().stream().parallel().forEach( (itemId) -> {
                Integer sessionCount = itemSessionMap.get(itemId).size();
                double probability = sessionCount / ((double) numOfTrainSessions);
                itemProbabilities.put(itemId, probability);
            })).get();
        } catch (InterruptedException | ExecutionException e) {
            logger.error(e.getMessage(), e);
        }

        return itemProbabilities;
    }

}
