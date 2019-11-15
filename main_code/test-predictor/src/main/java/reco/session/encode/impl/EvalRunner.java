package reco.session.encode.impl;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
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

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import reco.session.encode.ConfConstants;
import reco.session.encode.evaluator.util.EvaluationScorer;
import reco.session.encode.evaluator.util.NonAccuracyHelper;
import reco.session.encode.evaluator.util.data.DataStrategy;
import reco.session.encode.evaluator.util.data.GenericStrategy;
import reco.session.encode.evaluator.util.data.dao.SessionInteraction;
import reco.session.encode.utils.DataLoader;
import reco.session.encode.utils.PropertyHolder;

public class EvalRunner implements Runnable {

    public static String EVAL_STRATEGY_ALL = "ALL";
    public static String EVAL_STRATEGY_NEXT = "NEXT";

    private static final Logger logger = LoggerFactory.getLogger(EvalRunner.class);

    // default values
    private int cutOff = 10;
    // constructor values
    private String evalPath;
    private String predictPath;
    private String delimiter;
    private String modelName;


    public EvalRunner() {
        this.predictPath = PropertyHolder.getInstance().getPredictPath();
        this.evalPath = PropertyHolder.getInstance().getEvalPath();
        this.cutOff = PropertyHolder.getInstance().getCutOff();
        this.delimiter = PropertyHolder.getInstance().getDelimiter();
        this.modelName = PropertyHolder.getInstance().getModelName();
    }

    @Override
    public void run() {
        initNonAccStatistics();

        File folder = new File(predictPath);
        File[] listOfFiles = folder.listFiles();

        List<File> algoPredictionFiles = new ArrayList<>();

        for (int i = 0; i < listOfFiles.length; i++) {
            File file = listOfFiles[i];
            if (file.isFile()) {
                String fileName = file.getName();
                if (fileName.contains(modelName)) {
                    algoPredictionFiles.add(file);
                }
            }
        }


        algoPredictionFiles.stream().forEach( (algoFile) -> {
            try {
                evaluate(algoFile);
                logger.info("Evaluated " + algoPredictionFiles.indexOf(algoFile) + "/" + algoPredictionFiles.size());
            } catch (IOException e) {
                e.printStackTrace();
            }
        });
    }

    protected void evaluate(File predictionFile) 
                    throws FileNotFoundException, IOException {
        String algorithm = predictionFile.getName();
        
        File f1 = new File(evalPath + EVAL_STRATEGY_NEXT.toLowerCase() + "/" + algorithm);
        File f2 = new File(evalPath + EVAL_STRATEGY_ALL.toLowerCase() + "/" + algorithm);
        if (f1.exists() && f2.exists()) {
            logger.info("Skipping " + algorithm);
            return;
        }
        
        Map<String, EvaluationScorer> evaluationStrategies = new HashMap<>();
        evaluationStrategies.put(EVAL_STRATEGY_ALL, new EvaluationScorer(algorithm, cutOff));
        evaluationStrategies.put(EVAL_STRATEGY_NEXT, new EvaluationScorer(algorithm, cutOff));

        
        logger.info("Loading [" + algorithm + "] results ");

        List<String> lines = new ArrayList<>();
        try (BufferedReader bReader = new BufferedReader(new FileReader(predictionFile))){
            // skip header
            String line = bReader.readLine();

            while ((line = bReader.readLine()) != null) {
                lines.add(line);
            }
        } catch (IOException e) {
            logger.error(e.getMessage(), e);
        }

        int cores = Runtime.getRuntime().availableProcessors();
        try {
            ForkJoinPool fjp = new ForkJoinPool(cores);
            fjp.submit(() ->
            lines.stream().parallel().forEach( (line) -> {
                String[] dataValues = line.split("\t");

                String sessionId = dataValues[1];

                List<String> inputItems = Arrays.asList(dataValues[2].split("\\s*,\\s*"));
                Integer inputCount = Integer.valueOf(dataValues[3]);

                String position = dataValues[4];

                List<String> remainingItems = Arrays.asList(dataValues[5].split("\\s*,\\s*"));
                Integer remainingCount = Integer.valueOf(dataValues[6]);

                List<String> predictions = new ArrayList<>();
                if (dataValues.length >= 8) {
                    predictions = Arrays.asList(dataValues[7].split("\\s*,\\s*"));
                } else {
                    logger.warn("Session [" + sessionId + "] with input count [" + inputCount + "] and remaining count [" + remainingCount + "]had no predictions.");
                }

                evaluationStrategies.get(EVAL_STRATEGY_NEXT).addResult(sessionId, algorithm, remainingItems.subList(0, 1), predictions, inputItems);
                evaluationStrategies.get(EVAL_STRATEGY_ALL).addResult(sessionId, algorithm, remainingItems, predictions, inputItems);
            })).get();
        } catch (Exception e) {
            e.printStackTrace();
        }
        
        evaluationStrategies.keySet().forEach( (strategy) -> {
            storeEval(
                    evaluationStrategies.get(strategy).outputResults(algorithm, delimiter, cutOff),
                    algorithm,
                    strategy
                    );
        });
    }

    private void storeEval(String output, String algoName, String strategy) {
        String storePath = evalPath + strategy.toLowerCase() + "/" + algoName;

        DataLoader.deleteExisting(storePath);
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(storePath))) {
            bw.write(output);
            logger.info("Stored evaluation results to: " + storePath);
        } catch (IOException e) {
            logger.error("Failed to store evaluation results.", e);
        }

    }

    protected void initNonAccStatistics() {
        PropertyHolder prop = PropertyHolder.getInstance();
        DataStrategy loader = new GenericStrategy(
                prop.getDelimiter(), 
                prop.getUserIndex(), 
                prop.getSessionIndex(), 
                prop.getItemIndex(), 
                prop.getTimestampIndex()
                );

        List<SessionInteraction> interactions = new ArrayList<>();

        try (BufferedReader bReader = new BufferedReader(new FileReader(prop.getFilteredFile()))){
            // skip header
            String line = bReader.readLine();

            while ((line = bReader.readLine()) != null) {
                SessionInteraction interaction = loader.extractInteraction(line);
                interactions.add(interaction);
            }
        } catch (IOException e) {
            logger.error(e.getMessage(), e);
        }
        logger.info("Loaded " + prop.getFilteredFile());

        // init first in-memory
        NonAccuracyHelper.getInstance().init(interactions, prop.getModelPath(), prop.getDelimiter());
    }


}
