package reco.session.encode.evaluator.util.metrics.other;


import java.util.List;
import java.util.stream.IntStream;

import reco.session.encode.evaluator.util.NonAccuracyHelper;

/**
 * Calculates the serendipity measure on the recommended resources.<br/><br/>
 * Look also at the paper: <br/>
 * <i>Zhang et al., Auralist: Introducing Serendipity into Music Recommendation</i>
 * 
 */
public class SerendipityCalc {
	
	/**
	 * Calculates the serendipity of recommended items.<br/><br/> 
	 * Serendipity is calculated by adapting the formula in the paper: <br/> 
	 * Zhang et al., Auralist: Introducing Serendipity into Music Recommendation 
	 * 
	 * <br/><br/> 
	 * 
	 * The difference is that we sum the dissimilarity (i.e., 1 - similarity) instead.
	 * 
	 * @return Serendipity@k
	 */
	public double calculate(List<String> recommendations, List<String> sessionHistory) {
        int k = recommendations.size();

        
        Double serendipitySum = IntStream.range(0, k).mapToDouble( (itemIndex) -> {
        	
        	Double partialSerendipitySum = IntStream.range(0, sessionHistory.size()).mapToDouble( (h) -> {
        		String recommendItem = recommendations.get(itemIndex);
        		String historyItem = sessionHistory.get(h);
        		
        		Double dissimilarity = NonAccuracyHelper.getInstance().fetchDissimilarity(recommendItem, historyItem);
        		return dissimilarity;
        	}).sum();
        	
        	
        	return partialSerendipitySum;
        }).sum();
        
        // calculate how many item combinations will compared to each other
        Integer serendipityCalculations = k * sessionHistory.size();

		return serendipitySum / serendipityCalculations;
	}



	public double calculateEPD(List<String> recommendations, List<String> sessionHistory, List<String> expectedItems) {
		int k = recommendations.size();


		Double serendipitySum = IntStream.range(0, k).mapToDouble( (itemIndex) -> {
			String recommendItem = recommendations.get(itemIndex);

			// relevance factor is either 1 or 0
			if (expectedItems.contains(recommendItem)) {
				Double partialSerendipitySum = IntStream.range(0, sessionHistory.size()).mapToDouble( (h) -> {
					String historyItem = sessionHistory.get(h);

					Double dissimilarity = NonAccuracyHelper.getInstance().fetchDissimilarity(recommendItem, historyItem);
					return dissimilarity;
				}).sum();

				// nomralize partial serendipity sum
				partialSerendipitySum = partialSerendipitySum / sessionHistory.size();

				// item rank starts with index 0, so we need to add 2 because of the logarithm
				// Logarithmic discount: disc(k) = 1 / log_2(k + 2).
				double rankingDiscount = 1 / ( Math.log10(itemIndex + 2.0) / Math.log10(2) );

				return partialSerendipitySum * rankingDiscount;
			}

			return 0.0;
		}).sum();

		return serendipitySum / k;
	}



}