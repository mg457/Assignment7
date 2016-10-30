package ml.classifiers;

import ml.data.CrossValidationSet;
import ml.data.DataSet;
import ml.data.DataSetSplit;
import ml.data.Example;
/**
 * 
 * Assignment 7
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

	@Override
	public void train(DataSet data) {
		// TODO Auto-generated method stub

	}

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
		String csv = "/Users/maddie/Documents/FALL2016/MachineLearning/hw4/titanic-train.perc.csv";
		DataSet data = new DataSet(csv, 0);
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
