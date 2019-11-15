package reco.session.encode.evaluator.util.data;

import reco.session.encode.evaluator.util.data.dao.SessionInteraction;

/**
 * Strategy how to extract interaction data from one row of tsv/csv file. 
 * 
 */
public interface DataStrategy {
    
    public static final String DEFAULT_SPLIT_DELIMITER = "\t";

    /**
     * Checks if the file format is valid for the given strategy
     * @param line header of the file containing the dataset
     * @return true if the file structure is valid
     */
    public boolean checkHeader(String line);
    
    /**
     * Extracts a session interaction out of the provided row
     * @param line contains interaction data to extract
     * @return object containing the session interaction
     */
    public SessionInteraction extractInteraction(String line);
    
}