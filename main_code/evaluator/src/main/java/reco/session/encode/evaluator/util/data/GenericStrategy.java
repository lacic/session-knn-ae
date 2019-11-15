package reco.session.encode.evaluator.util.data;

import reco.session.encode.evaluator.util.data.dao.SessionInteraction;

/**
 * Strategy how to extract interaction data from one row of tsv/csv file of the Studo dataset. 
 * 
 */
public class GenericStrategy implements DataStrategy {
    
    private String delimiter;
    private Integer userIndex;
    private Integer sessionIndex; 
    private Integer itemIndex;
    private Integer timeIndex;
    
    public GenericStrategy(String delimiter, Integer userIndex, Integer sessionIndex, Integer itemIndex, Integer timeIndex) {
        this.delimiter = delimiter;
        this.userIndex = userIndex;
        this.sessionIndex = sessionIndex;
        this.itemIndex = itemIndex;
        this.timeIndex = timeIndex;
    }
    
    
    @Override
    public boolean checkHeader(String line) {
        // don' check
        return true;
    }
    
    @Override
    public SessionInteraction extractInteraction(String line) {
        String[] dataValues = line.split(delimiter);

        SessionInteraction interaction = new SessionInteraction();
        interaction.setUserId(dataValues[userIndex]);
        interaction.setItemId(dataValues[itemIndex]);
        interaction.setSessionId(dataValues[sessionIndex]);
        interaction.setType(dataValues[timeIndex]);
        
        return interaction;
    }
    
}