/**
 * Copyright (C) 2017
 * "Know-Center GmbH Research Center for Data-Driven Business & Big Data Analytics",
 * Graz, Austria, office@know-center.at. All rights reserved.
 *
 * This work is part of a proof of concept. It is provided "as is", without warranty of any kind,
 * expressed or implied. In no event shall the authors, holders of copyright or exploitation rights, be liable
 * for any claim, damages, or other liability, arising in connection with this work and its use.
 *
 * For further information about intellectual property rights, warranty and liability,
 * please consult the project cooperation contract.
 */
package reco.session.encode.evaluator.util.metrics.accuracy;
import java.util.List;

/**
 * Calculator for the MRR measure
 * @author Emanuel Lacic
 *
 */
public class MRR {

    /**
     * Compute the MRR of a list of ranked items.
     *
     * @return the MRR for the given data
     */
    public static double calculateMRR(List<String> realData, List<String> predictionData) {
        double sum = 0.0;
        if (predictionData.size() != 0 && realData.size() != 0) {
            for (String val : realData) {
                int index = 0;
                if ((index = predictionData.indexOf(val)) != -1) {
                    sum += (1.0 / (index + 1));
                } else {
                    sum += 0.0;
                }
            }
        }
        if (realData.size() == 0) {
            return 0.0;
        }
        return sum / realData.size();
    }

}
