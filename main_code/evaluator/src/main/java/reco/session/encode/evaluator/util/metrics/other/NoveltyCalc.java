package reco.session.encode.evaluator.util.metrics.other;


import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

import reco.session.encode.evaluator.util.NonAccuracyHelper;

/**
 * Calculates the novelty measure on the recommended resources.<br/><br/>
 * Look also at the paper: <br/>
 * <i>Zhang et al., Auralist: Introducing Serendipity into Music Recommendation</i>
 *
 */
public class NoveltyCalc {

    private Map<String, Integer> popularityMap;

    /**
     * Creates a new novelty calculator.
     */
    public NoveltyCalc() {
        // initialize item popularity
        this.popularityMap = NonAccuracyHelper.getInstance().getCachedPopularityMap();
    }

    /**
     * Calculates the novelty of recommended items.<br/><br/> 
     *
     * Novelty is calculated as denoted in the paper: <br/>
     *
     * Zhang et al., Auralist: Introducing Serendipity into Music Recommendation
     *
     * <br/><br/> 
     * The difference is that we add 1 to every item's popularity in order to account for 
     * items with 0 values in the popularity field. We also normalize the novelty value by dividing it with the
     * "novelty" of the most popular item.
     *
     * @param recommendations use the provided recommendations to calculate novelty
     * @return Novelty@k
     */
    public double calculate(List<String> recommendations) {
        Double highestNoveltyScore =
                Math.log10(
                        NonAccuracyHelper.getInstance().getBiggestPopularity()
                ) / Math.log10(2);

        Double noveltySum =
                recommendations.stream().mapToDouble( (item) -> {
                    // add 1 to account for items with no interactions
                    Integer itemPop = popularityMap.get(item);
                    Integer itemPopularity = popularityMap.get(item) + 1;

                    Double noveltyScore = Math.log10(itemPopularity) / Math.log10(2);
                    double normalizedNovelty = noveltyScore / highestNoveltyScore;

                    return  normalizedNovelty;

                }).sum();


        return 1 - noveltySum / recommendations.size();
    }

    /**
     * Calculates the novelty of recommended items.<br/><br/>
     *
     * Novelty is calculated as denoted in the paper: <br/>
     *
     * Vargas, S., & Castells, P. (2011, October).
     * Rank and relevance in novelty and diversity metrics for recommender systems.
     * In Proceedings of the fifth ACM conference on Recommender systems (pp. 109-116). ACM.
     *
     *
     * @param recommendations
     * @return
     */
    public double calculateEPC(List<String> recommendations, List<String> expectedItems) {
        Integer highestNoveltyScore = NonAccuracyHelper.getInstance().getBiggestPopularity() + 1;

        Double noveltySum =
                IntStream.range(0, recommendations.size()).mapToDouble( (itemIndex) -> {
                    String item = recommendations.get(itemIndex);

                    // relevance factor is either 1 or 0
                    if (expectedItems.contains(item)) {
                        // add 1 to account for items with no interactions
                        Integer itemPopularity = popularityMap.get(item) + 1;

                        double pSeen = itemPopularity / (double) highestNoveltyScore;
                        // Alternative:
                        //double itemNovelty = -1 * ( Math.log10(pSeen) / Math.log10(2) );
                        double itemNovelty = 1 - pSeen;


                        // item rank starts with index 0, so we need to add 2 because of the logarithm
                        // Logarithmic discount: disc(k) = 1 / log_2(k + 2).
                        double rankingDiscount = 1 / ( Math.log10(itemIndex + 2.0) / Math.log10(2) );

                        return itemNovelty * rankingDiscount;
                    }

                    return 0.0;
                }).sum();

        return noveltySum / recommendations.size();
    }

}