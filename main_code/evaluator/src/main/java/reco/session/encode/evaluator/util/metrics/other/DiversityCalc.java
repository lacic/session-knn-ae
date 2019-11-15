package reco.session.encode.evaluator.util.metrics.other;


import java.util.List;
import java.util.stream.IntStream;

import reco.session.encode.evaluator.util.NonAccuracyHelper;


/**
 * Calculates the diversity measure on the recommended resources.<br/><br/>
 * Look also at the paper: <br/>
 * <i>Zhang et al., Auralist: Introducing Serendipity into Music Recommendation</i>
 * 
 */
public class DiversityCalc {

	/**
	 * Calculates the diversity of recommended items.<br/><br/> 
	 * Diversity is calculated by adapting the formula in the paper: <br/> 
	 * Zhang et al., Auralist: Introducing Serendipity into Music Recommendation 
	 * <br/><br/> The difference is that we sum the dissimilarity (i.e., 1 - similarity) instead 
	 * and return the average diversity instead of the sum - this is important when comparing different sizes of
	 * recommendation lists.
	 * 
	 * @param k calculate diversity on top-k recommendations (i.e., Diversity@k) 
	 * @return Diversity@k
	 */
	public double calculate(List<String> recommendations) {
        int k = recommendations.size();
        
        Double diversitySum = IntStream.range(0, k).mapToDouble( (i) -> {
        	
        	Double partialDiversitySum = IntStream.range(i + 1, k).mapToDouble( (j) -> {
        		String itemA = recommendations.get(i);
        		String itemB = recommendations.get(j);
        		
        		return NonAccuracyHelper.getInstance().fetchDissimilarity(itemA, itemB);
        	}).sum();
        	
        	
        	return partialDiversitySum;
        }).sum();
        
        // calculate how many item combinations will compared to each other with n*(n-1)/2
        Integer diversityCalculations = (k * (k - 1)) / 2;
        
        double diversity = 0.0;
        if (diversityCalculations > 0) {
        	diversity = diversitySum / diversityCalculations;
        }

        return diversity;
	}
	
}