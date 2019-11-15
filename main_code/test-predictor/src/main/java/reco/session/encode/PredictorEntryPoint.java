package reco.session.encode;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import reco.session.encode.impl.AllRunner;
import reco.session.encode.impl.EvalRunner;
import reco.session.encode.impl.baseline.PredictionBayesRunner;
import reco.session.encode.impl.knn.PredictionHiddenVecRunner;
import reco.session.encode.utils.PropertyHolder;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

/**
 * Starting point of the UMUAI "Using Autoencoders for Session-based Job Recommendations" code
 *
 */
public class PredictorEntryPoint {

    private static final Logger logger = LoggerFactory.getLogger(PredictorEntryPoint.class);

    public static void main(String[] args) throws Exception {
        if (args.length != 2) {
            logger.error("Missing arguments! \nNeed to provide (1) path to config file and (2) the run-mod [predict-bayes|predict|eval].");
            return;
        }

        Map<String, Runnable> modeStrategies = createRunModes(args[0]);

        String[] runModes = args[1].split(",");
        
        for (String runMode : runModes) {
            Runnable modeRunner = modeStrategies.getOrDefault(runMode, new Runnable() {
                
                @Override
                public void run() {
                    logger.info("Invalid run-mode provided as argument. Please choose between [preprocess|train|infer|predict|all].");
                }
            });
            
            modeRunner.run();
        }
    }

    /**
     * Default running modes: [predict|eval]
     * 
     * Bayes baseline mode: [predict-bayes]
     **
     * @param confFile configuration file
     * @return map containing modes runners
     */
    protected static Map<String, Runnable> createRunModes(String confFile) {
        PropertyHolder.getInstance().initProperties(confFile);

        Map<String, Runnable> modeStrategies = new HashMap<>();

        // init mode strategies for the paper
        PredictionHiddenVecRunner predictRunner = new PredictionHiddenVecRunner();
        modeStrategies.put(ConfConstants.MODE_PREDICT, predictRunner);

        EvalRunner evalRunner = new EvalRunner();
        modeStrategies.put(ConfConstants.MODE_EVAL, evalRunner);
        
        // extra modes
        
        PredictionBayesRunner predBayesRunner = new PredictionBayesRunner();
        modeStrategies.put(ConfConstants.MODE_PREDICT_BAYES, predBayesRunner);
        
        return modeStrategies;
    }

}
