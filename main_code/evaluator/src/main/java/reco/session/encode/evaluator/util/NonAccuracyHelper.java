package reco.session.encode.evaluator.util;


import java.io.*;
import java.math.RoundingMode;
import java.text.DecimalFormat;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

import com.google.common.collect.Sets;
import org.apache.commons.math3.stat.descriptive.SynchronizedDescriptiveStatistics;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import reco.session.encode.evaluator.util.data.dao.SessionInteraction;


/**
 * Abstract class for calculating non-accuracy measures.<br/><br/>
 * Look also at the paper: <br/>
 * <i>Zhang et al., Auralist: Introducing Serendipity into Music Recommendation</i>
 *
 */
public class NonAccuracyHelper {

    private static final Logger logger = LoggerFactory.getLogger(NonAccuracyHelper.class);

    private static class Holder {
        static final NonAccuracyHelper INSTANCE = new NonAccuracyHelper();
    }

    protected NonAccuracyHelper() {	}

    /**
     * Get's instance using the initialization-on-demand holder idiom
     * @return
     */
    public static NonAccuracyHelper getInstance() {
        return Holder.INSTANCE;
    }

    private Map<String, List<String>> interactionMap;
    private Map<String,Integer> popularityMap;
    private AtomicInteger biggestPopularity;

    private Map<String, Map<String,Double>> disimilarityMap;



    public Map<String, List<String>> getCachedInteractionMap() {
        return interactionMap;
    }

    public Map<String, Integer> getCachedPopularityMap() {
        return popularityMap;
    }

    public Integer getBiggestPopularity() {
        if (biggestPopularity.get() > 0) {
            // return cached value if set
            return biggestPopularity.get();
        }

        throw new IllegalStateException("You first need to init the popularity map!");
    }

    private final static String DISSIMILARITY_FILE_NAME = "item_dissimilarity.csv";
    /**
     * Sets and returns the popularity of the provided items
     *
     * @param interactions interaction data 
     * @return map containing the popularity for each item id
     */
    public void init(List<SessionInteraction> interactions, String interimPath, String delimiter) {
        biggestPopularity = new AtomicInteger(-1);
        popularityMap = new ConcurrentHashMap<>();
        interactionMap = new ConcurrentHashMap<>();
        disimilarityMap = new ConcurrentHashMap<>();

        File dissimFile = new File(interimPath + DISSIMILARITY_FILE_NAME);
        if (dissimFile.exists()) {
            System.out.println("Loading dissimilarities from: " + dissimFile.getAbsolutePath());
            disimilarityMap = loadDissimilarityMap(delimiter, dissimFile);
        }

        logger.info("Starting to calculate item popularity from " + interactions.size() + " interactions ...");
        ForkJoinPool forkJoinPool = new ForkJoinPool(2);
        try {
            forkJoinPool.submit(() ->
                    interactions.stream().parallel().forEach( (interaction) -> {
                        String itemId = interaction.getItemId();
                        // check if file does not exists, then need to init new
                        if (!dissimFile.exists()) {
                            disimilarityMap.putIfAbsent(itemId, new ConcurrentHashMap<>());
                        }

                        // calculate popularity
//                if (interaction.getType().equals(interactionType)) {
                        popularityMap.putIfAbsent(itemId, 1);
                        popularityMap.computeIfPresent(itemId, (key, value) -> {
                            int newCount = popularityMap.get(key) + 1;
                            popularityMap.put(key, newCount);
                            // check the biggest popularity
                            if (biggestPopularity.get() < newCount) {
                                biggestPopularity.set(newCount);
                            }

                            return newCount;
                        });
//                }
                        // set item interactions for calculating item dissimilarity
                        interactionMap.putIfAbsent(itemId, new ArrayList<>());
                        interactionMap.computeIfPresent(itemId, (key, value) -> {
                            value.add(interaction.getUserId());
                            interactionMap.put(key, value);

                            return value;
                        });
                    })).get();
        } catch (InterruptedException | ExecutionException e) {
            biggestPopularity = new AtomicInteger(-1);
            popularityMap = new ConcurrentHashMap<>();
            interactionMap = new ConcurrentHashMap<>();

            logger.error("Could not calculate dissimilarities!", e);
        }

        // check if file does not exists, then need to init new
        if (!dissimFile.exists()) {
            calcInteractionDissimilarity();
            storeDissimilarities(dissimFile, delimiter);
        }
        System.out.println("Successfuly initialized helper for non-accuracy metrics with " + interactions.size() + " interactions.");

    }

    private Map<String, Map<String, Double>> loadDissimilarityMap(String delimiter, File dissimFile) {
        Map<String, Map<String, Double>> loadedDissimilarityMap = new ConcurrentHashMap<>();

        try (BufferedReader bReader = new BufferedReader(new FileReader(dissimFile))){
            // skip header
            String line = bReader.readLine();

            while ((line = bReader.readLine()) != null) {
                String[] dataSplit = line.split(delimiter);

                String itemA = dataSplit[0];
                String itemB = dataSplit[1];
                Double dissimilarity = Double.parseDouble(dataSplit[2]);

                loadedDissimilarityMap.putIfAbsent(itemA, new ConcurrentHashMap<>());
                loadedDissimilarityMap.compute(itemA, (key, value) -> {
                    Map<String, Double> mapping = loadedDissimilarityMap.get(key);
                    mapping.put(itemB, dissimilarity);
                    return mapping;
                });
            }

        } catch (IOException e) {
            e.printStackTrace();
        }
        return loadedDissimilarityMap;
    }

    public void calcInteractionDissimilarity(){
        List<String> items = new ArrayList<>(interactionMap.keySet());


        AtomicInteger counter = new AtomicInteger(0);

        int cores = (int) (Runtime.getRuntime().availableProcessors() * 0.8);
        ForkJoinPool forkJoinPool = new ForkJoinPool(cores);

        logger.info("Starting to calculate item dissimilarity with " + cores + " cores for " + items.size() + " items ...");

        // for every item
        IntStream.range(0, items.size()).forEach( (i) -> {
            // multi-thread for calculating similarities with others
            String itemA = items.get(i);
            try {
                forkJoinPool.submit(() ->
                        IntStream.range(i + 1, items.size()).parallel().forEach( (j) -> {
                            String itemB = items.get(j);

                            Double dissimilarity = calcDissimilarity(itemA, itemB);

                            double differenceCheck = Math.abs(1.0 - dissimilarity);
                            // store only if dissimilarity is not 1.0 (i.e., have nothing in common)
                            if (differenceCheck > 0.000001) {
                                disimilarityMap.compute(itemB, (key, value) -> {
                                    Map<String, Double> mapping = disimilarityMap.get(key);
                                    mapping.put(itemA, dissimilarity);
                                    disimilarityMap.put(key, mapping);
                                    return value;
                                });
                            }

                        })).get();
            } catch (InterruptedException | ExecutionException e) {
                e.printStackTrace();
            }
            int currCount = counter.incrementAndGet();
            if (currCount % 1000 == 0) {
                logger.info((items.size() - currCount) + " items remaining to calculate dissimilarity.");
            }
        });


        System.out.println("Successfuly computed dissimilarity map");
    }

    public void storeDissimilarities(File dissimFile, String delimiter) {
        DecimalFormat df = new DecimalFormat("#.#####");
        df.setRoundingMode(RoundingMode.CEILING);

        try (BufferedWriter bw = new BufferedWriter(new FileWriter(dissimFile))) {

            StringBuilder sb = new StringBuilder();
            sb.append("itemA");
            sb.append(delimiter);
            sb.append("itemB");
            sb.append(delimiter);
            sb.append("dissimilarity");
            sb.append("\n");

            bw.write(sb.toString());


            for (String itemA : disimilarityMap.keySet()) {
                Map<String, Double> itemADissimilarities = disimilarityMap.get(itemA);
                for (String itemB : itemADissimilarities.keySet()) {
                    Double dissimilarity = itemADissimilarities.get(itemB);

                    bw.write(itemA);
                    bw.write(delimiter);
                    bw.write(itemB);
                    bw.write(delimiter);
                    bw.write(df.format(dissimilarity));
                    bw.write("\n");
                }
            }

            System.out.println("Stored dissimilarities to: " + dissimFile.getAbsolutePath());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public Double fetchDissimilarity(String itemA, String itemB) {
        if (itemA.equals(itemB)) {
            return 0.0;
        }

        Double dissimilarity = Double.NaN;
        // first try for A
        Map<String, Double> itemAdissimilarities = disimilarityMap.get(itemA);

        if (itemAdissimilarities != null && itemAdissimilarities.containsKey(itemB)){
            // pick A -> B mapping
            dissimilarity = itemAdissimilarities.get(itemB);
        } else {
            // try for B
            Map<String, Double> itemBdissimilarities = disimilarityMap.get(itemB);
            if (itemBdissimilarities != null && itemBdissimilarities.containsKey(itemA)){
                // pick B -> A mapping
                dissimilarity = itemBdissimilarities.get(itemA);
            } else {
                // dissimilarity mapping does not exist, items don't have anything in common
                dissimilarity = 1.0;
                //throw new IllegalStateException("Missing dissimalirity mapping for items [" + itemA + "] and [" + itemB + "].");
            }

        }

        return dissimilarity;
    }

    /**
     * Calculates the dissimilarity between two items
     * @param itemA id of the first item to compare
     * @param itemB id of the second item to compare
     * @return dissimilarity between two items
     */
    public Double calcDissimilarity(String itemA, String itemB) {
        List<String> usersOfItemA = interactionMap.get(itemA);
        List<String> usersOfItemB = interactionMap.get(itemB);

        Double dissimilarity = calcDissimilarity(usersOfItemA, usersOfItemB);
        return dissimilarity;
    }

    /**
     * Calculates the dissimilarity between two items
     * @param usersOfItemA users that interacted with the first item to compare
     * @param usersOfItemB users that interacted with the second item to compare
     * @return dissimilarity between two items
     */
    public static Double calcDissimilarity(List<String> usersOfItemA, List<String> usersOfItemB) {
        return 1.0 - calcSimilarity(usersOfItemA, usersOfItemB);
    }

    /**
     * Calculates the cosine similarity between two items as defined in the paper of: <br/>
     * Zhang et al., Auralist: Introducing Serendipity into Music Recommendation 
     * @param usersOfItemA users that interacted with the first item to compare
     * @param usersOfItemB users that interacted with the second item to compare
     * @return similarity between two items
     */
    public static Double calcSimilarity(List<String> usersOfItemA, List<String> usersOfItemB) {
        Integer intersectingUsersCount = null;

        //  performs slightly better when set1 is the smaller of the two sets
        if (usersOfItemA.size() > usersOfItemB.size()) {
            intersectingUsersCount =
                    Sets.intersection(Sets.newHashSet(usersOfItemB), Sets.newHashSet(usersOfItemA))
                            .size();
        } else {
            intersectingUsersCount =
                    Sets.intersection(Sets.newHashSet(usersOfItemA), Sets.newHashSet(usersOfItemB))
                            .size();
        }

        //Long intersectingUsersCount = usersOfItemA.stream().parallel().filter( (a) -> {
        //    return usersOfItemB.contains(a);
        //}).count();

        Double denominator = Math.sqrt( usersOfItemA.size() ) * Math.sqrt( usersOfItemB.size() );

        double similarity = intersectingUsersCount / denominator;

        return similarity;
    }

}