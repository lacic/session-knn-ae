package reco.session.encode.impl.knn;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * To be extended for approaches which need to recommend in a KNN manner
 *
 */
public abstract class AbstractKNNRunner {

    private static final Logger logger = LoggerFactory.getLogger(AbstractKNNRunner.class);

    /**
     * Loads input from file in memory
     *
     * @param file file name with path to load
     * @return
     */
    protected List<String> loadInputLines(String file) {
        List<String> testInput = new ArrayList<>();
        try (BufferedReader bReader = new BufferedReader(new FileReader(file))){
            String line;
            while ((line = bReader.readLine()) != null) {
                testInput.add(line);
            }
        } catch (IOException e) {
            logger.error(e.getMessage(), e);
        }


        return testInput;
    }

    /**
     * Given session data from the train set constructs a reverse mapping to get sessions that interacted with an item
     *
     * @param trainHistory session history
     * @return item session mapping
     */
    protected Map<String, Set<String>> constructItemSessionMap(Map<String, List<String>> trainHistory) {
        Map<String, Set<String>> itemSessionsMapping = new ConcurrentHashMap<>();
        try {
            ForkJoinPool mainPool = new ForkJoinPool(2);
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

}
