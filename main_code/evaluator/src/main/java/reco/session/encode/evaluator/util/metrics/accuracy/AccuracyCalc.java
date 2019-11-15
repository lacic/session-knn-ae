package reco.session.encode.evaluator.util.metrics.accuracy;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;

public class AccuracyCalc{

	private String userID;
	private List<String> realData;
	private List<String> predictionData;
	
	public AccuracyCalc(String userID, List<String> realData, List<String> predictionData, int k) {
		this.userID = userID;
		// in case the same item is contained multiple times in the history
		this.realData = new ArrayList<>(new HashSet<>(realData));
		if (k == 0 || predictionData.size() < k) {
			this.predictionData = predictionData;
		} else {
			this.predictionData = predictionData.subList(0, k);
		}
 	}
	
	/**
	* Compute the normalized discounted cumulative gain (NDCG) of a list of ranked items.
	*
	* @return the NDCG for the given data
	*/
	public double getNDCG() {
		return NDCG.calculateNDCG(realData, predictionData);
	}

	public double getMRR() {
		return MRR.calculateMRR(realData, predictionData);
	}

	// Getter ------------------------------------------------------------------------------------------------
	
	public String getUserID() {
		return this.userID;
	}
	
	public List<String> getRealData() {
		return this.realData;
	}
	
	public List<String> getPredictionData() {
		return this.predictionData;
	}
}
