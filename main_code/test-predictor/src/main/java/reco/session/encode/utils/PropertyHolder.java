package reco.session.encode.utils;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Properties;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import reco.session.encode.ConfConstants;

/**
 * Container of properties which define how to run the application
 */
public class PropertyHolder {
    
    private static final Logger logger = LoggerFactory.getLogger(PropertyHolder.class);

    private static class Holder {
       private static final PropertyHolder INSTANCE = new PropertyHolder();
    }

    public static PropertyHolder getInstance() {
        return Holder.INSTANCE;
    }

    private String trainFile;
    private String processedTrainFile;
    private String testFile;

    private String inferPath;
    private String predictPath;
    private String evalPath;

    
    private String delimiter;
    
    private Integer userIndex;
    private Integer itemIndex;
    private Integer sessionIndex;
    private Integer timestampIndex;
    
    private String samplingStrategy;
    private String gpuMode;
    
    private String modelPath;
    private String modelName;
    private String modelType;
    private Boolean attention;

    private Integer hiddenLayerSize;
    private Integer batchSize;
    private Integer iterations;
    private Integer listenerFrequency;
    private Integer cutOff;
    private Boolean storeEveryEpoch;
    
    private Integer topKSessionsMin;
    private Integer topKSessionsMax;
    private Integer topKSessionsStep;

    private Integer epochsMin;
    private Integer epochsMax;
    private Integer epochsStep;

    private Double reminderRatio;
    private String reminderLocation;
    private String generateStrategy;
    
    private String filteredFile;

    private PropertyHolder(){ }        
    
    public void initProperties(String confFile) {
        InputStream input = null;

        try {
            input = new FileInputStream(confFile);

            Properties prop = new Properties();
            // load a properties file
            prop.load(input);
            setProperties(prop);
        } catch (IOException ex) {
            logger.error("Could not read configuration file: " + ex.getMessage(), ex);
        } finally {
            if (input != null) {
                try {
                    input.close();
                } catch (IOException e) {
                    logger.error(e.getMessage(), e);
                }
            }
        }
    };
    
    /**
     * Initializes properties from file
     * @param prop 
     */
    private void setProperties(Properties prop) {
        trainFile = prop.getProperty("file.train.original");
        processedTrainFile = prop.getProperty("file.train.processed");
        testFile = prop.getProperty("file.test");

        inferPath = prop.getProperty("path.infer");
        predictPath = prop.getProperty("path.predict");
        evalPath = prop.getProperty("path.eval");
        
        filteredFile = prop.getProperty("file.filtered");

        delimiter = prop.getProperty("delimiter");

        userIndex = 
            getIntOrDefault(prop, "index.user", 1);
        itemIndex = 
                getIntOrDefault(prop, "index.item", 2);
        sessionIndex = 
                getIntOrDefault(prop, "index.session", 3);
        timestampIndex = 
                getIntOrDefault(prop, "index.timestamp", 4);
        
        samplingStrategy = prop.getProperty("strategy.sampling");
        gpuMode = prop.getProperty("gpuMode");
        
        modelPath = prop.getProperty("model.path");
        modelName = prop.getProperty("model.name");
        modelType = prop.getProperty("model.type");
        attention = Boolean.parseBoolean(prop.getProperty("model.attention", "false"));

        hiddenLayerSize = 
                getIntOrDefault(prop, "hiddenLayer", 100);
        batchSize = 
                getIntOrDefault(prop, "batchSize", 100);
        iterations = 
                getIntOrDefault(prop, "iterations", 1);
        listenerFrequency = 
                getIntOrDefault(prop, "listenerFrequency", 1000);
        cutOff = 
                getIntOrDefault(prop, "cutOff", 20);
        
        epochsMax = 
                getIntOrDefault(prop, "epoch.max", 10);
        epochsMin = 
                getIntOrDefault(prop, "epoch.min", 1);
        epochsStep = 
                getIntOrDefault(prop, "epoch.step", 1);
        
        topKSessionsMax = 
                getIntOrDefault(prop, "topKSessions.max", 60);
        topKSessionsMin = 
                getIntOrDefault(prop, "topKSessions.min", 40);
        topKSessionsStep = 
                getIntOrDefault(prop, "topKSessions.step", 5);
        
        reminderRatio = 
                getDoubleOrDefault(prop, "remind.ratio", 0.5);
        reminderLocation = prop.getProperty("remind.location");
        generateStrategy = prop.getProperty("strategy.generate");

        storeEveryEpoch = Boolean.parseBoolean(prop.getProperty("storeEveryEpoch"));
        
    }
    
    private int getIntOrDefault(Properties prop, String propertyName, Integer defaultValue) {
        String value = prop.getProperty(propertyName, defaultValue.toString());
        try {
            return Integer.parseInt(value);
        } catch (NumberFormatException e) {
            logger.error(e.getMessage(), e);
            return defaultValue;
        }
    }
    
    private double getDoubleOrDefault(Properties prop, String propertyName, Double defaultValue) {
        String value = prop.getProperty(propertyName, defaultValue.toString());
        try {
            return Double.parseDouble(value);
        } catch (NumberFormatException e) {
            logger.error(e.getMessage(), e);
            return defaultValue;
        }
    }
    
    public String getTrainFile() {
        return trainFile;
    }

    public String getProcessedTrainFile() {
        return processedTrainFile;
    }

    public String getTestFile() {
        return testFile;
    }

    public String getInferPath() {
        return inferPath;
    }

    public String getPredictPath() {
        return predictPath;
    }

    public String getDelimiter() {
        return delimiter;
    }

    public Integer getItemIndex() {
        return itemIndex;
    }

    public Integer getSessionIndex() {
        return sessionIndex;
    }

    public Integer getTimestampIndex() {
        return timestampIndex;
    }

    public String getSamplingStrategy() {
        return samplingStrategy;
    }

    public String getGpuMode() {
        return gpuMode;
    }

    public String getModelPath() {
        return modelPath;
    }

    public String getModelName() {
        return modelName;
    }

    public Integer getHiddenLayerSize() {
        return hiddenLayerSize;
    }

    public Integer getBatchSize() {
        return batchSize;
    }

    public Integer getIterations() {
        return iterations;
    }

    public Integer getListenerFrequency() {
        return listenerFrequency;
    }

    public Integer getCutOff() {
        return cutOff;
    }

    public Boolean getStoreEveryEpoch() {
        return storeEveryEpoch;
    }

    public Integer getTopKSessionsMin() {
        return topKSessionsMin;
    }

    public Integer getTopKSessionsMax() {
        return topKSessionsMax;
    }

    public Integer getTopKSessionsStep() {
        return topKSessionsStep;
    }

    public Integer getEpochsMin() {
        return epochsMin;
    }

    public Integer getEpochsMax() {
        return epochsMax;
    }

    public Integer getEpochsStep() {
        return epochsStep;
    }

    public Double getReminderRatio() {
        return reminderRatio;
    }

    public String getReminderLocation() {
        if (reminderLocation != null) {
            return reminderLocation;
        } else {
            return ConfConstants.NONE;
        }
    }

    public String getEvalPath() {
        return evalPath;
    }

    public String getFilteredFile() {
        return filteredFile;
    }

    public Integer getUserIndex() {
        return userIndex;
    }

    public String getGenerateStrategy() {
        if (generateStrategy != null) {
            return generateStrategy;
        } else {
            return ConfConstants.NONE;
        }
    }

    public String getModelType() {
        return modelType;
    }

    public Boolean hasAttention() {
        return attention;
    }
    
    
    
}
