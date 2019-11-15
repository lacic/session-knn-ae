package reco.session.encode.impl.knn;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.atomic.AtomicInteger;

import org.deeplearning4j.parallelism.ConcurrentHashSet;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import reco.session.encode.ConfConstants;
import reco.session.encode.utils.DataHandler;
import reco.session.encode.utils.DataLoader;
import reco.session.encode.utils.PropertyHolder;


public class PredictionHiddenVecRunner extends AbstractKNNRunner implements Runnable {

    private static final Logger logger = LoggerFactory.getLogger(PredictionHiddenVecRunner.class);

    // default values
    private int cutOff = 10;
    // constructor values
    private String inferPath;
    private String predictPath;
    private String delimiter;

    private String modelType;


    public PredictionHiddenVecRunner() {
        this.inferPath = PropertyHolder.getInstance().getInferPath();
        this.predictPath = PropertyHolder.getInstance().getPredictPath();
        this.delimiter = PropertyHolder.getInstance().getDelimiter();

        this.modelType = PropertyHolder.getInstance().getModelType();

        this.cutOff = PropertyHolder.getInstance().getCutOff();
    }

    @Override
    public void run() {
        String[] filePrefixes = modelType.split(",");

        for (String filePrefix : filePrefixes) {
            if (PropertyHolder.getInstance().hasAttention()) {
                runPrediction("att_" + filePrefix);
            } else {
                runPrediction(filePrefix);
            }
        }
    }

    /**
     * Using the provided train file fills trainVectorsToInit and trainHistoryToInit
     * @param trainFile file to read data from
     * @param trainVectorsToInit vectors representing a session
     * @param trainHistoryToInit items for every session
     */
    protected void fillTrainData(
            String trainFile,
            Map<String, INDArray> trainVectorsToInit,
            Map<String, List<String>> trainHistoryToInit) {
        // init train
        try (BufferedReader bReader = new BufferedReader(new FileReader(trainFile))){
            String line;
            while ((line = bReader.readLine()) != null) {
                String[] dataSplit = line.split(delimiter);

                String sessionId = dataSplit[0];
                String sessionVector = dataSplit[1];
                String sessionItems = dataSplit[2];

                double[] vectorValues = Arrays.stream(sessionVector.split(",")).mapToDouble(Double::parseDouble).toArray();
                INDArray vectorArray = Nd4j.create(vectorValues);
                trainVectorsToInit.put(sessionId, vectorArray);

                List<String> sessionHistory = Arrays.asList(sessionItems.split(","));
                trainHistoryToInit.put(sessionId, sessionHistory);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Runs the prediction
     * @param inferSessionFilePrefix should consist of the information which autoencoder is used and with or without attention (e.g., "att_dae", "vae", etc.)
     */
    protected void runPrediction(String inferSessionFilePrefix) {
        String inferSessionTrainFile = inferPath + inferSessionFilePrefix + "_encoder_train.csv";
        String inferSessionTestFile = inferPath + inferSessionFilePrefix + "_encoder_test.csv";

        if (new File(inferSessionTrainFile).exists() && new File(inferSessionTestFile).exists()) {
            Map<String, INDArray> trainVectors = new HashMap<>();
            Map<String, List<String>> trainHistory = new HashMap<>();
            // init session vectors and session items that were interacted with
            fillTrainData(inferSessionTrainFile, trainVectors, trainHistory);

            generateTopKPredictions(inferSessionFilePrefix, inferSessionTestFile, trainVectors, trainHistory);
        } else {
            logger.error("Inferred train and test file do not exist! Skipping prediction for model:" + inferSessionFilePrefix);
        }
    }

    protected void storePredictions(String currentModelName, List<String> predictionOutput) {
        storePredictions(currentModelName, predictionOutput, -1);
    }

    /**
     * Stores predictions for evaluation
     *
     * @param currentModelName
     * @param predictionOutput
     * @param topKSessions
     */
    protected void storePredictions(String currentModelName, List<String> predictionOutput, int topKSessions) {
        String storePath = predictPath + currentModelName + "_test_14d" + ".csv";
        if (topKSessions > 0) {
            storePath = predictPath + currentModelName + "_test_14d_topks_" + topKSessions + ".csv";

        }

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

    protected void generateTopKPredictions(
            String currentModelName,
            String testFile,
            Map<String, INDArray> trainVectors,
            Map<String, List<String>> trainHistory) {
        PropertyHolder propHolder = PropertyHolder.getInstance();
        int topKSessionsMin = propHolder.getTopKSessionsMin();
        int topKSessionsMax = propHolder.getTopKSessionsMax();
        int topKStep = propHolder.getTopKSessionsStep();

        DataLoader loader = new DataLoader();
        loader.init();
        List<String> testInput = loadInputLines(testFile);

        Map<Integer, List<String>> predictionOutput = new ConcurrentHashMap<>();
        for (int topKSession = topKSessionsMin; topKSession <= topKSessionsMax; topKSession += topKStep) {
            // placeholder for outputs
            predictionOutput.put(topKSession, Collections.synchronizedList(new ArrayList<>()));
        }

        Map<String, Set<String>> itemSessionsMapping = constructItemSessionMap(trainHistory);

        Set<String> allSessions = new HashSet<>();
        for (Set<String> itemSessions : itemSessionsMapping.values()) {
            allSessions.addAll(itemSessions);
        }

        logger.info("Recommending with top-k sessions from interval: [" + topKSessionsMin + "," + topKSessionsMax + "] "
                + "with step size: " + topKStep);
        logger.info("Reminder ratio: [" + propHolder.getReminderRatio() + "]. Reminder location: [" + propHolder.getReminderLocation() + "]");
        AtomicInteger counter = new AtomicInteger(0);

        int cores = Runtime.getRuntime().availableProcessors();
        // init forkJoinPools
        ForkJoinPool mainPool = new ForkJoinPool(2);
        ForkJoinPool secondaryPool = new ForkJoinPool(cores/3);

        int minSessionFound = 2;
        AtomicInteger tooSmallSessionCounter = new AtomicInteger(0);

        Set<String> smallSessions = new ConcurrentHashSet<>();
        Set<String> testedSessions = new ConcurrentHashSet<>();
        try {
            mainPool.submit(() ->
                    testInput.stream().parallel().forEach( (line) -> {
                        // read a test case
                        String[] dataSplit = line.split(delimiter);

                        String sessionId = dataSplit[0];
                        String sessionVector = dataSplit[1];
                        String inputItems = dataSplit[2];
                        String remainingItems = dataSplit[3];
                        // to report session count
                        testedSessions.add(sessionId);

                        List<String> currSessionItems = Arrays.asList(inputItems.split(","));

                        int inputLength = currSessionItems.size();

                        int remainingLength = remainingItems.split(",").length;
                        String position = "MID";
                        if (inputLength == 1) {
                            position = "FIRST";
                        }
                        if (remainingLength == 1) {
                            position = "LAST";
                        }

                        double[] vectorValues = Arrays.stream(sessionVector.split(",")).mapToDouble(Double::parseDouble).toArray();
                        INDArray vectorArray = Nd4j.create(vectorValues);

                        String currInputItem = currSessionItems.get(inputLength - 1);
                        Set<String> candidateSessions = new HashSet<>(itemSessionsMapping.get(currInputItem));

                        //Set<String> candidateSessions = new HashSet<>();
                        //for (String sessionItem : currSessionItems) {
                        //    candidateSessions.addAll(itemSessionsMapping.get(sessionItem));
                        //}

                        /*
                        if (candidateSessions.size() < minSessionFound) {
                            // check all
                            tooSmallSessionCounter.incrementAndGet();
                            smallSessions.add(sessionId);
                            candidateSessions.addAll(loader.getRecentSessions().subList(0, topKSessionsMax - candidateSessions.size()));
                        }
                        */

                        Map<String, Double> trainSessionSimilarityMap = new ConcurrentHashMap<>();

                        try {
                            // first calculate session similarity
                            secondaryPool.submit(() ->
                                    candidateSessions.stream().parallel().forEach( (trainSessionId) -> {
                                        try {
                                            INDArray trainVector = trainVectors.get(trainSessionId);
                                            double similarity = Transforms.cosineSim(vectorArray, trainVector);
                                            trainSessionSimilarityMap.put(trainSessionId, similarity);
                                        } catch (Exception en) {
                                            logger.error("Train sessionId: " + trainSessionId);
                                            logger.error("Train vectors is null: " + (trainVectors == null));
                                            logger.error("Train Vector: " + trainVectors.get(trainSessionId));
                                            logger.error(en.getMessage(), en);
                                            throw en;
                                        }

                                    })).get();
                        } catch (InterruptedException | ExecutionException e) {
                            logger.error(e.getMessage(), e);
                        }

                        // get top k similar sessions
                        TreeMap<String, Double> sortedSessionMap = DataHandler.sortByValues(trainSessionSimilarityMap);

                        for (int topKSession = topKSessionsMin; topKSession <= topKSessionsMax; topKSession += topKStep) {
                            int currentTopK = topKSession;
                            if (currentTopK > sortedSessionMap.size()) {
                                currentTopK = sortedSessionMap.size();
                            }
                            List<String> topKSessionsIds = new ArrayList<>(sortedSessionMap.keySet()).subList(0, currentTopK);

                            Map<String, Double> itemSimilarityItemMap = new ConcurrentHashMap<>();
                            // calculate relevant items
                            topKSessionsIds.stream().forEach( (trainSessionId) -> {
                                double similarity = trainSessionSimilarityMap.get(trainSessionId);

                                List<String> sessionItems = trainHistory.get(trainSessionId);

                                try {
                                    secondaryPool.submit(() ->
                                            sessionItems.stream().parallel().forEach( (item) -> {
                                                itemSimilarityItemMap.putIfAbsent(item, similarity);
                                                itemSimilarityItemMap.computeIfPresent(item, (key,value) -> {
                                                    Double prevSimilarity = itemSimilarityItemMap.get(key);
                                                    double similaritySum = prevSimilarity + similarity;
                                                    return similaritySum;
                                                });

                                            })).get();
                                } catch (InterruptedException | ExecutionException e) {
                                    logger.error(e.getMessage(), e);
                                }
                            });


                            int remindCount = (int) Math.floor(inputLength * propHolder.getReminderRatio());
                            int novelCount = inputLength - remindCount;

                            // extract items to recommend
                            Map<String, Double> sortedItemMap = DataHandler.sortByValues(itemSimilarityItemMap);

                            List<String> reminderItems = new ArrayList<>();
                            List<String> novelItems = new ArrayList<>();

                            for (String item : sortedItemMap.keySet()) {
                                if (reminderItems.size() < remindCount) {
                                    if (currSessionItems.contains(item)) {
                                        reminderItems.add(item);
                                    }
                                } else if (novelItems.size() < novelCount) {
                                    if (! currSessionItems.contains(item)) {
                                        novelItems.add(item);
                                    }
                                } else {
                                    break;
                                }
                            }


                            List<String> recommendations = new ArrayList<>();

                            // configure for GPU device
                            switch (propHolder.getReminderLocation()) {
                                case ConfConstants.REMINDER_LOC_BOT:
                                    recommendations.addAll(novelItems);
                                    recommendations.addAll(reminderItems);
                                    break;
                                case ConfConstants.REMINDER_LOC_TOP:
                                    recommendations.addAll(reminderItems);
                                    recommendations.addAll(novelItems);
                                    break;
                                case ConfConstants.REMINDER_LOC_MID:
                                    recommendations.addAll(novelItems.subList(novelItems.size() / 2, novelItems.size()));
                                    recommendations.addAll(reminderItems);
                                    recommendations.addAll(novelItems.subList(0, novelItems.size() / 2));
                                    break;
                                case ConfConstants.REMINDER_LOC_ALT:
                                    // REMIND_IN AN ALTERNATING FASHION
                                    while (reminderItems.size() > 0 || novelItems.size() > 0) {
                                        if (novelItems.size() > 0) {
                                            recommendations.add(novelItems.remove(0));
                                        }
                                        if (reminderItems.size() > 0) {
                                            recommendations.add(reminderItems.remove(0));
                                        }
                                    }
                                    break;
                                default:
                                    // just recommend items based on score, disregarding reminders
                                    break;
                            }

                            for (String item : sortedItemMap.keySet()) {
                                if (recommendations.size() >= cutOff) {
                                    break;
                                }
                                // if not enough fill with rest
                                if (!recommendations.contains(item)) {
                                    recommendations.add(item);
                                }
                            }


                            String predictions = String.join(",", recommendations);

                            StringBuilder sb = new StringBuilder();
                            sb.append(delimiter);
                            sb.append(sessionId);
                            sb.append(delimiter);
                            sb.append(inputItems);
                            sb.append(delimiter);
                            sb.append(inputLength);
                            sb.append(delimiter);
                            sb.append(position);
                            sb.append(delimiter);
                            sb.append(remainingItems);
                            sb.append(delimiter);
                            sb.append(remainingLength);
                            sb.append(delimiter);
                            sb.append(predictions);
                            sb.append("\n");

                            predictionOutput.compute(topKSession, (key, value) -> {
                                List<String> currentOutput = predictionOutput.get(key);
                                currentOutput.add(sb.toString());
                                return currentOutput;
                            });
                        }

                        int currentCounter = counter.incrementAndGet();
                        if (currentCounter % 1000 == 0) {
                            logger.info("Made predictions for " + currentCounter + " entries...");
                        }
                    })).get();
        } catch (InterruptedException | ExecutionException e) {
            logger.error(e.getMessage(), e);
        }

        logger.info(smallSessions.size() + " sessions had a neighborhood which is smaller than " + minSessionFound + ". With " + tooSmallSessionCounter.get() + " occurances.");



        logger.info("Made predictions for " + testedSessions.size() + " sessions.");


        for (Integer topK : predictionOutput.keySet()) {
            String remindModelName =
                    "r_" + propHolder.getReminderLocation() +
                            "_" + propHolder.getReminderRatio() +
                            "_" + currentModelName;
            storePredictions(remindModelName, predictionOutput.get(topK), topK);
        }
    }


}
