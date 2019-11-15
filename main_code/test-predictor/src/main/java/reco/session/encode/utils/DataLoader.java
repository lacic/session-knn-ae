package reco.session.encode.utils;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.TreeMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.util.concurrent.AtomicDouble;

import reco.session.encode.ConfConstants;

/**
 * Utility class to load data to train or test on
 * 
 */
public class DataLoader {

    private String dataPath;
    private String delimiter;
    private Integer sessionIndex;
    private Integer itemIndex;
    private Integer tsIndex;

    private AtomicInteger itemIndexCounter;
    private Map<String, Integer> itemIndicies;

    private AtomicDouble popItemIndexCounter;
    private Map<String, Double> popItemIndicies;

    private Map<String, Map<Long, String>> sessionData;
    
    private List<String> recentSessions;


    private static final Logger logger = LoggerFactory.getLogger(DataLoader.class);


    public DataLoader(String path, String delimiter, Integer sessionIndex, Integer itemIndex, Integer timestampIndex) {
        this.dataPath = path;
        this.delimiter = delimiter;
        this.sessionIndex = sessionIndex;
        this.itemIndex = itemIndex;
        this.tsIndex = timestampIndex;
    }

    /**
     * Initializes loader with the property repository
     */
    public DataLoader() {
        PropertyHolder propHolder = PropertyHolder.getInstance();
        this.dataPath = propHolder.getTrainFile();
        this.delimiter = propHolder.getDelimiter();
        this.sessionIndex = propHolder.getSessionIndex();
        this.itemIndex = propHolder.getItemIndex();
        this.tsIndex = propHolder.getTimestampIndex();

    }

    /**
     * Initializes the data loader for discrete input (e.g., 1 if a user has interacted with an item and 0 otherwise)
     * @return
     */
    public DataLoader init(){
        init(true);
        return this;
    }
    
    /**
     * Initializes internal data structure
     * @return self
     */
    public DataLoader init(boolean useDiscrete){
        logger.info("Loading train data from: " + dataPath);
        this.sessionData = new HashMap<>();
        this.itemIndicies = new HashMap<>();
        this.itemIndexCounter = new AtomicInteger(0);
        
        
        Map<String, Long> sessionTimes = new TreeMap<>(Collections.reverseOrder());
        
        try (BufferedReader bReader = new BufferedReader(new FileReader(dataPath))){
            // skip header
            String line = bReader.readLine();

            while ((line = bReader.readLine()) != null) {
                String[] dataSplit = line.split(delimiter);

                String sessionId = dataSplit[sessionIndex];
                String itemId = dataSplit[itemIndex];
                Long timestamp = Long.parseLong(dataSplit[tsIndex]);
                
                sessionTimes.putIfAbsent(sessionId, timestamp);
                if (timestamp > sessionTimes.get(sessionId)) {
                    sessionTimes.put(sessionId, timestamp);
                }

                sessionData.putIfAbsent(sessionId, new TreeMap<>());

                Map<Long, String> sessionInteractions = sessionData.get(sessionId);
                sessionInteractions.put(timestamp, itemId);
                sessionData.put(sessionId, sessionInteractions);

                if (!itemIndicies.containsKey(itemId)) {
                    itemIndicies.put(itemId, itemIndexCounter.getAndIncrement());
                }
            }

            
            TreeMap<String, Long> sortedSessions = DataHandler.sortByValues(sessionTimes);
            int maxSessions = 1000;
            if (maxSessions > sortedSessions.size()) {
                maxSessions = sortedSessions.size();
            }
            this.recentSessions = new ArrayList<>(sortedSessions.keySet()).subList(0, maxSessions);
        } catch (IOException e) {
            e.printStackTrace();
        }

//        double halfDown = - Math.floor((itemIndexCounter.get() / 2.0));
        this.popItemIndexCounter = new AtomicDouble(1.0);

        this.popItemIndicies = new HashMap<>();
        itemIndicies.keySet().forEach( (itemId) -> {
            popItemIndicies.put(itemId, popItemIndexCounter.getAndAdd(1.0));

        });

        return this;
    }

    public List<String> getRecentSessions() {
        return recentSessions;
    }

    public Double getMinContItemIndex() {
        return 1.0;
    }

    public Double getmaxContItemIndex() {
        return popItemIndexCounter.get();
    }

    public String getItemForIndex(Integer index) {
        return itemIndicies.keySet()
                .stream()
                .filter(key -> index.equals(itemIndicies.get(key)))
                .findFirst().get();
    }

    public String getContItemForIndex(Double index) {
        if (index  < 1.0 || index >= popItemIndexCounter.get()) {
            return null;
        }
        
        String item = null;
        try {
            item = popItemIndicies.keySet()
                    .stream()
                    .filter(key -> index.equals(popItemIndicies.get(key)))
                    .findFirst().get();
        } catch (NoSuchElementException ex) {
            System.out.println(index);
            ex.printStackTrace();
        }
        return item;
    }


    /**
     * 
     * @return
     */
    public List<double[]> fetchTrainSessionVectors(String samplingStrategy, boolean useDiscrete) {
        Map<String, List<String>> trainSessionSequence = mapTrainSessionSequence();

        List<double[]> sessionVectors = new ArrayList<>();

        trainSessionSequence.keySet().stream().forEach( (sessionId) -> {
            List<String> itemSequence = trainSessionSequence.get(sessionId);

            // vector values initially are set to 0.0
            double[] originalVector = new double[itemIndexCounter.get()];

            IntStream.range(0, itemSequence.size()).forEach( (sequenceIndex) -> {
                String item = itemSequence.get(sequenceIndex);
                Integer index = itemIndicies.get(item);
                if (useDiscrete) {
                    originalVector[index] = 1.0;
                } else {
                    originalVector[index] = sequenceIndex + 1.0;
                }
                // should we subsample the session sequence to generate more examples
                if (samplingStrategy.equals(ConfConstants.DATA_SAMPL_SUBSAMPLE)) {
                    double[] subVector = new double[itemIndexCounter.get()];

                    IntStream.range(0, sequenceIndex).forEach( (subSequenceIndex) -> {
                        String subItem = itemSequence.get(subSequenceIndex);
                        Integer subIndex = itemIndicies.get(subItem);
                        if (useDiscrete) {
                            subVector[subIndex] = 1.0;
                        } else {
                            subVector[subIndex] = subSequenceIndex + 1.0;
                        }
                    });

                    sessionVectors.add(subVector);
                }
            });


            sessionVectors.add(originalVector);
        });

        return sessionVectors;
    }
    
    /**
     * Fetches session vectors for all sessions in the train set
     * @return session to vector mapping
     */
    public Map<String, List<double[]>> fetchTrainSessionVectors() {
        Map<String, List<String>> trainSessionSequence = mapTrainSessionSequence();

        Map<String, List<double[]>> sessionVectors = new HashMap<>();

        trainSessionSequence.keySet().stream().forEach( (sessionId) -> {
            List<String> itemSequence = trainSessionSequence.get(sessionId);
            
            List<double[]> sessionInteractions = new ArrayList<>();
            IntStream.range(0, itemSequence.size()).forEach( (sequenceIndex) -> {
             // vector values initially are set to 0.0
                double[] originalVector = new double[itemIndexCounter.get()];
                
                String item = itemSequence.get(sequenceIndex);
                Integer index = itemIndicies.get(item);
                originalVector[index] = 1.0;
                sessionInteractions.add(originalVector);
            });
            sessionVectors.put(sessionId, sessionInteractions);
        });

        return sessionVectors;
    }

    /**
     * Constructs input vectors which have a fixed size and at position i contain the index of item i
     * @param samplingStrategy original data or subsample a session
     * @param maxSize vector size
     * @return
     */
    public List<double[]> fetchTrainContSessionVectors(String samplingStrategy, int maxSize) {
        Map<String, List<String>> trainSessionSequence = mapTrainSessionSequence();

        List<double[]> sessionVectors = new ArrayList<>();

        trainSessionSequence.keySet().stream().forEach( (sessionId) -> {
            List<String> itemSequence = trainSessionSequence.get(sessionId);

            // vector values initially are set to 0.0
            double[] originalVector = new double[maxSize];
            //            Arrays.fill(originalVector, -1.0);

            IntStream.range(0, itemSequence.size()).forEach( (sequenceIndex) -> {
                if (sequenceIndex >= maxSize) {
                    return;
                }

                String item = itemSequence.get(sequenceIndex);
                Double index = popItemIndicies.get(item);
                originalVector[sequenceIndex] = index;
                // should we subsample the session sequence to generate more examples
                if (samplingStrategy.equals(ConfConstants.DATA_SAMPL_SUBSAMPLE)) {
                    double[] subVector = new double[maxSize];
                    //                    Arrays.fill(subVector, -1.0);

                    IntStream.range(0, sequenceIndex).forEach( (subSequenceIndex) -> {
                        String subItem = itemSequence.get(subSequenceIndex);
                        Double subIndex = popItemIndicies.get(subItem);
                        subVector[sequenceIndex] = subIndex;
                    });

                    sessionVectors.add(subVector);
                }
            });


            sessionVectors.add(originalVector);
        });

        return sessionVectors;
    }


    /**
     * Creates a mapping for item sequences for a given session in the train set
     * @return map containing the item sequence for a given session id in the train set
     */
    public Map<String, List<String>> mapTrainSessionSequence() {
        Map<String, List<String>> trainSessions = new HashMap<>();

        sessionData.keySet().stream().forEach( (sessionId) -> {
            Map<Long, String> sessionInteractions = sessionData.get(sessionId);
            List<String> itemSequence = new ArrayList<>(sessionInteractions.values());
            trainSessions.put(sessionId, itemSequence);
        });

        return trainSessions;
    }


    /**
     * Size of the session vector
     * @return
     */
    public Integer getSessionVectorSize() {
        return itemIndexCounter.get();
    }

    /**
     * Fetches session vectors for the test dataset
     * @return list of session vectors
     */
    public List<double[]> fetchTestSessionVectors(String testPath) {
        Map<String, List<String>> sessionSequenceMap = mapTestSessionSequence(testPath);

        List<double[]> testSessionVectors = new ArrayList<>();

        sessionSequenceMap.keySet().stream().forEach( (testSessionId) -> {
            List<String> itemSequence = sessionSequenceMap.get(testSessionId);

            // vector values initially are set to 0.0
            double[] originalVector = new double[itemIndexCounter.get()];

            IntStream.range(0, itemSequence.size()).forEach( (sequenceIndex) -> {
                String item = itemSequence.get(sequenceIndex);
                Integer index = itemIndicies.get(item);
                originalVector[index] = 1.0;
            });


            testSessionVectors.add(originalVector);
        });

        return testSessionVectors;
    }

    public Map<String, List<String>> mapTestSessionSequence(String testPath) {
        return mapTestSessionSequence(testPath, true);
    }


    /**
     * Creates a mapping for item sequences for a given session in the test set
     * @param testPath path containing the test data
     * @param useDiscrete should discrete input values be used
     * @return map containing the item sequence for a given session id
     */
    public Map<String, List<String>> mapTestSessionSequence(String testPath, boolean useDiscrete) {
        logger.info("Loading test data from: " + testPath);
        Map<String, Map<Long, String>> testSessionData = new HashMap<>();

        try (BufferedReader bReader = new BufferedReader(new FileReader(testPath))){
            // skip header
            String line = bReader.readLine();

            while ((line = bReader.readLine()) != null) {
                String[] dataSplit = line.split(delimiter);

                String sessionId = dataSplit[sessionIndex];
                String itemId = dataSplit[itemIndex];
                Long timestamp = Long.parseLong(dataSplit[tsIndex]);

                boolean itemInDiscreteSet = useDiscrete && itemIndicies.containsKey(itemId);
                boolean iteminContinousSet = !useDiscrete && popItemIndicies.containsKey(itemId);
                if ( itemInDiscreteSet 
                        || iteminContinousSet ){
                    testSessionData.putIfAbsent(sessionId, new TreeMap<>());

                    Map<Long, String> sessionInteractions = testSessionData.get(sessionId);
                    sessionInteractions.put(timestamp, itemId);
                    testSessionData.put(sessionId, sessionInteractions);
                }
            }

        } catch (IOException e) {
            logger.error(e.getMessage(), e);
        }

        Map<String, List<String>> sessionMapping = new HashMap<>();

        testSessionData.keySet().stream().forEach( (testSessionId) -> {
            Map<Long, String> sessionInteractions = testSessionData.get(testSessionId);
            List<String> itemSequence = new ArrayList<>(sessionInteractions.values());
            sessionMapping.put(testSessionId, itemSequence);
        });

        return sessionMapping;
    }

    /**
     * @return map containing index for a given item
     */
    public Map<String, Integer> getItemIndicies(){
        return itemIndicies;
    }

    /**
     * @return map containing a continous index for a given item
     */
    public Map<String, Double> getContItemIndicies(){
        return popItemIndicies;
    }


    /**
     * Deletes file if exists
     * @param path
     * @return
     */
    public static boolean deleteExisting(String path) {
        boolean deleted = false;
        File fileToDelete = new File(path);
        if (fileToDelete.exists()) {
            deleted = fileToDelete.delete();
        }
        return deleted;
    }



}
