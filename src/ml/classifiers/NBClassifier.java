package ml.classifiers;

import java.util.*;

import ml.data.CrossValidationSet;
import ml.data.DataSet;
import ml.data.DataSetSplit;
import ml.data.Example;
import ml.utils.*;

/**
 * 
 * Assignment 7
 * 
 * @author Maddie Gordon, Nick Reminder
 *
 */
public class NBClassifier implements Classifier {

	double lambda = 0.1;
	boolean usePosOnly = false; // tells program whether to use approach in
								// which only positive features are used to
								// calculate probabilities

	/**
	 * Set the regularization/smoothing parameter to a new value.
	 * 
	 * @param newLambda
	 *            New lambda value.
	 */
	public void setLambda(double newLambda) {
		lambda = newLambda;
	}

	// Two approaches:
	// **for both, train and calculating individual feature probs is same
	// 1) use all features
	// 2) use only positive/present features (for when examples are sparse)

	/**
	 * Method which allows the user to choose which variant of classification
	 * will be used--calculating probabilities using only positive features or
	 * updating using all features.
	 * 
	 * @param setting
	 *            True if using only positive features, false if using all
	 *            features.
	 */
	public void setUseOnlyPositiveFeatures(boolean setting) {
		usePosOnly = setting;
	}

	/**
	 * Store the counts for each feature in each example.
	 * 
	 * @param data
	 *            DataSet for which we are storing feature counts.
	 */
	@Override
	public void train(DataSet data) {
		// store raw counts
		ArrayList<Example> examples = data.getData();
		//ArrayList<HashMapCounter<Double>> words = new ArrayList<HashMapCounter<Double>>();
		HashMapCounter<Double> hm = new HashMapCounter<Double>();

		for (Example ex : examples) {
			for (int i : ex.getFeatureSet()) {
				hm.put(ex.getFeature(i), 1);
			}
			//words.add(hm);
		}
	}
	
	//count(x_i, y) = run through each example & when has label y & feature x_i, increment count
	public int countFeatureandLabel(ArrayList<Example> examples, double feature, int label) {
		for(Example ex: examples) {
			
		}
		return 0;
	}
	
	//count(y) = run through all examples & when has label y increment count

	@Override
	public double classify(Example example) {
		// TODO Auto-generated method stub
		return 0;
	}

	/**
	 * Return the log probability of the most likely label for the given
	 * example.
	 * 
	 * @param example
	 *            Example for which to calculate the log prob. of the most
	 *            likely label.
	 */
	@Override
	public double confidence(Example example) {
		// TODO Auto-generated method stub
		return 0;
	}

	/**
	 * Return the log probability of the example with the label under the
	 * current trained model.
	 * 
	 * @param ex
	 *            Example for which to calculate log probability.
	 * @param label
	 *            Label to be used to calculate log probability.
	 * @return p(x_1, x_2,...,x_m, y)
	 */
	public double getLogProb(Example ex, double label) {
		return 0; // TODO
	}

	/**
	 * Return the probability of a given feature index given a
	 * 
	 * @param featureIndex
	 * @param label
	 * @return
	 */
	public double getFeatureProb(int featureIndex, double label) {
		return 0; // TODO
	}

	public static void main(String[] args) {
		NBClassifier c = new NBClassifier();
		String file = "/Users/maddie/Documents/FALL2016/MachineLearning/hw5/wines.train.txt";
		DataSet data = new DataSet(file, 1);
		CrossValidationSet cs = new CrossValidationSet(data, 10, true);
		for (int i = 0; i < cs.getNumSplits(); i++) {
			double avg = 0.0;
			DataSetSplit dss = cs.getValidationSet(i);

			for (int iter = 0; iter < 100; iter++) {
				c.train(dss.getTrain());
				double acc = 0.0;
				double size = dss.getTest().getData().size();
				for (Example ex : dss.getTest().getData()) {
					if (c.classify(ex) == ex.getLabel()) {
						acc += 1.0 / size;
					}
				}
				avg += acc / 100.0;
			}
			System.out.println(avg);
		}

	}

}
