package reco.session.encode.evaluator.util;

import java.math.RoundingMode;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

import org.apache.commons.math3.stat.descriptive.SynchronizedDescriptiveStatistics;

import reco.session.encode.evaluator.util.metrics.accuracy.AccuracyCalc;
import reco.session.encode.evaluator.util.metrics.other.DiversityCalc;
import reco.session.encode.evaluator.util.metrics.other.NoveltyCalc;
import reco.session.encode.evaluator.util.metrics.other.SerendipityCalc;


/**
 * Provides recommendation scoring functionality to use when evaluating algorithms
 *
 */
public class EvaluationScorer {

    /**
     * Keeps track of recommendation statistics
     */
    private volatile Map<String, Map<String, SynchronizedDescriptiveStatistics>> algoStatistics;
    private Integer maxResult;;

    public EvaluationScorer(String algorithm, Integer maxResult) {
        List<String> algorithms = new ArrayList<String>();
        algorithms.add(algorithm);
        init(algorithms, maxResult);
    }
    
    public EvaluationScorer(List<String> algorithms, Integer maxResult) {	
        init(algorithms, maxResult);
    }
    
    protected void init(List<String> algorithms, Integer maxResult) {
        this.algoStatistics = new ConcurrentHashMap<>();
        this.maxResult = maxResult;
        // init statistics
        for (String algo : algorithms) {
            Map<String, SynchronizedDescriptiveStatistics> metricStatistics = new ConcurrentHashMap<String, SynchronizedDescriptiveStatistics>();
            for (int i = 1; i <= maxResult; i++){
                metricStatistics.put(NDCG + "@" + i, new SynchronizedDescriptiveStatistics());
                metricStatistics.put(DIVERSITY + "@" + i, new SynchronizedDescriptiveStatistics());
                metricStatistics.put(SERENDIPITY + "@" + i, new SynchronizedDescriptiveStatistics());
                metricStatistics.put(NOVELTY + "@" + i, new SynchronizedDescriptiveStatistics());
                metricStatistics.put(EPC + "@" + i, new SynchronizedDescriptiveStatistics());
                metricStatistics.put(EPD + "@" + i, new SynchronizedDescriptiveStatistics());
                metricStatistics.put(MRR + "@" + i, new SynchronizedDescriptiveStatistics());
            }
            metricStatistics.put(USER_COVERAGE, new SynchronizedDescriptiveStatistics());
            algoStatistics.put(algo, metricStatistics);
        }
    }


    /* Accuracy metrics */
    private static final String NDCG = "nDCG";

    /* Non-accuracy metrics */
    private static final String USER_COVERAGE = "userCoverage";
    private static final String SERENDIPITY = "serendipity";
    private static final String EPC = "novEPC";
    private static final String EPD = "serEPD";
    private static final String DIVERSITY = "diversity";
    private static final String NOVELTY = "novelty";
    private static final String MRR = "MRR";


    /**
     * Calculates evaluation metrics on the recommendation result and adds it to the overall statistics
     * @param user user (or session) for which the recommendation was generated
     * @param result recommendation result
     */

    /**
     * Calculates evaluation metrics on the recommendation result and adds it to the overall statistics
     * 
     * @param user the user (or session) for which the recommendation was generated
     * @param algorithm algorithm that is currently being evaluated
     * @param expectedIds items from the test set that should be predicted
     * @param recommendedIds items that were predicted
     */
    public void addResult(String user, String algorithm, List<String> expectedIds, List<String> recommendedIds, List<String> historyIds) {
        Map<String, SynchronizedDescriptiveStatistics > metricStatistic = algoStatistics.get(algorithm);

        NoveltyCalc nc = new NoveltyCalc();
        DiversityCalc dc = new DiversityCalc();
        SerendipityCalc sc = new SerendipityCalc();

        if (recommendedIds != null && recommendedIds.size() > 0) {
            // calculate for every @k
            for(int i = 1; i <= maxResult; i++){
                List<String> recommendationsAtK;

                if (i > recommendedIds.size()) {
                    recommendationsAtK = recommendedIds;
                } else {
                    recommendationsAtK = recommendedIds.subList(0, i);
                }

                AccuracyCalc pc = new AccuracyCalc(user, expectedIds, recommendationsAtK, i);

                metricStatistic.get(NDCG + "@" + i).addValue(pc.getNDCG());
                metricStatistic.get(NOVELTY + "@" + i).addValue(nc.calculate(recommendationsAtK));
                metricStatistic.get(DIVERSITY + "@" + i).addValue(dc.calculate(recommendationsAtK));
                metricStatistic.get(SERENDIPITY + "@" + i).addValue(sc.calculate(recommendationsAtK, historyIds));
                metricStatistic.get(EPC + "@" + i).addValue(nc.calculateEPC(recommendationsAtK, expectedIds));
                metricStatistic.get(EPD + "@" + i).addValue(sc.calculateEPD(recommendationsAtK, historyIds, expectedIds));
                metricStatistic.get(MRR + "@" + i).addValue(pc.getMRR());
            }
            metricStatistic.get(USER_COVERAGE).addValue(1);
        } else {
            metricStatistic.get(USER_COVERAGE).addValue(0);
        }
    }


    public void printResults(Integer k) {
        Set<String> profileIds = algoStatistics.keySet();

        DecimalFormat df = new DecimalFormat("#.#####"); 
        df.setRoundingMode(RoundingMode.HALF_UP);

        for (String profile : profileIds) {
            Map<String, SynchronizedDescriptiveStatistics> statistics = algoStatistics.get(profile);
            System.out.println("Reporting mean values for: [" + profile + "]");
            System.out.println(
                    df.format(statistics.get(NDCG + "@" + k).getMean())
                    + "    " +
                    df.format(statistics.get(DIVERSITY + "@" + k).getMean())
                    + "    " +
                    df.format(statistics.get(NOVELTY + "@" + k).getMean())
                    + "    " +
                    df.format(statistics.get(SERENDIPITY + "@" + k).getMean())
                    + "    " +
                    df.format(statistics.get(USER_COVERAGE).getMean())
                    );
            System.out.println("----------------------------------" );
        }

    }
    
    public String outputResults(String algoName, String delimiter, Integer cutOff) {
        DecimalFormat df = new DecimalFormat("#.#####"); 
        df.setRoundingMode(RoundingMode.CEILING);
        
        Map<String, SynchronizedDescriptiveStatistics> statistics = algoStatistics.get(algoName);
        
        StringBuilder sb = new StringBuilder();
        
        sb.append("k");
        sb.append(delimiter);
        sb.append("nDCG");
        sb.append(delimiter);
        sb.append("Diversity");
        sb.append(delimiter);
        sb.append("Novelty");
        sb.append(delimiter);
        sb.append("Serendipity");
        sb.append(delimiter);
        sb.append("NovEPC");
        sb.append(delimiter);
        sb.append("SerEPD");
        sb.append(delimiter);
        sb.append("MRR");
        sb.append(delimiter);
        sb.append("UC");
        sb.append("\n");
        
        for (int k = 1; k <= cutOff; k++) {
            sb.append(k);
            sb.append(delimiter);
            sb.append(df.format(statistics.get(NDCG + "@" + k).getMean()));
            sb.append(delimiter);
            sb.append(df.format(statistics.get(DIVERSITY + "@" + k).getMean()));
            sb.append(delimiter);
            sb.append(df.format(statistics.get(NOVELTY + "@" + k).getMean()));
            sb.append(delimiter);
            sb.append(df.format(statistics.get(SERENDIPITY + "@" + k).getMean()));
            sb.append(delimiter);
            sb.append(df.format(statistics.get(EPC + "@" + k).getMean()));
            sb.append(delimiter);
            sb.append(df.format(statistics.get(EPD + "@" + k).getMean()));
            sb.append(delimiter);
            sb.append(df.format(statistics.get(MRR + "@" + k).getMean()));
            sb.append(delimiter);
            sb.append(df.format(statistics.get(USER_COVERAGE).getMean()));

            if (k < cutOff) {
                sb.append("\n");
            }
        }
        
        return sb.toString();
    }

}
