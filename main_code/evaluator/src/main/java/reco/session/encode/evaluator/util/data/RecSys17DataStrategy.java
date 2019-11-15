package reco.session.encode.evaluator.util.data;

import reco.session.encode.evaluator.util.data.dao.SessionInteraction;

/**
 * Strategy how to extract interaction data from one row of tsv/csv file of the RecSys17 dataset.
 * 
 */
public class RecSys17DataStrategy implements DataStrategy {
    
    private String delimiter;
    
    /**
     * Default character to split a row is a tab
     */
    public RecSys17DataStrategy() {
        delimiter = DataStrategy.DEFAULT_SPLIT_DELIMITER;
    }
    
    /**
     * Init strategy with another delimiter
     * @param delimiter
     */
    public RecSys17DataStrategy(String delimiter) {
        this.delimiter = delimiter;
    }
    
    @Override
    public boolean checkHeader(String line) {
        String[] headerValues = line.split(delimiter);

        if (headerValues.length < 6) { 
            System.out.println("Not enough columns. It was " + headerValues.length + ", but should be at least 6 (including first empty column for the index)."); 
            return false; 
        }
        if (! headerValues[1].equals("user_id")) { System.out.println("Expecting user_id on the column with index 1.");return false; }
        if (! headerValues[2].equals("item_id")) { System.out.println("Expecting item_id on the column with index 2."); return false; }
        if (! headerValues[3].equals("interaction_type")) { System.out.println("Expecting interaction_type on the column with index 3."); return false; }
        if (! headerValues[5].equals("session_id")) { System.out.println("Expecting session_id on the column with index 5."); return false; }
        
        return true;
    }
    
    @Override
    public SessionInteraction extractInteraction(String line) {
        String[] dataValues = line.split(delimiter);

        SessionInteraction interaction = new SessionInteraction();
        interaction.setUserId(dataValues[1]);
        interaction.setItemId(dataValues[2]);
        interaction.setType(dataValues[3]);
        interaction.setSessionId(dataValues[5]);
        
        return interaction;
    }
    
}