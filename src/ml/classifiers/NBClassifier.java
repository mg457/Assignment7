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
	ArrayList<HashMapCounter<Integer>> featureLabelCounts;
	HashMapCounter<Double> labelCounts;
	int allExamples;
	DataSet data;

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
	 * Store the counts for each label and feature/label combination in the data
	 * set.
	 * 
	 * @param data
	 *            DataSet for which we are storing label/feature counts.
	 */
	@Override
	public void train(DataSet data) {
		// store raw counts
		allExamples = data.getData().size();
		this.data = data;
		featureLabelCounts = countFeaturesandLabels(data);
		labelCounts = countLabels(data);
	}

	/**
	 * Count the occurences of certain features associated with specific labels.
	 * 
	 * @param data
	 *            Dataset
	 * @return a list of lists of associations between labels, features indices
	 *         associated with each label, and counts associated with this
	 *         combination of labels and feature index
	 */
	public ArrayList<HashMapCounter<Integer>> countFeaturesandLabels(DataSet data) {
		ArrayList<HashMapCounter<Integer>> list = new ArrayList<HashMapCounter<Integer>>();
		ArrayList<Example> examples = data.getData();
		// for each label, store a list of features with associated counts
		for (double l : data.getLabels()) {
			HashMapCounter<Integer> hm = new HashMapCounter<Integer>();
			// for (int f : data.getAllFeatureIndices()) {
			// for (Example ex : examples) {
			// Set<Integer> features = ex.getFeatureSet();
			// for (int f : features) {
			// if (ex.getLabel() == l) {
			// hm.increment(f);
			// }
			// }
			// }
			// }
			list.add(hm);
		}
		for (Example ex : examples) {
			HashMapCounter<Integer> current = list.get((int) ex.getLabel());
			for (int f : ex.getFeatureSet()) {
				current.increment(f);
			}
		}
		return list;
	}

	/**
	 * Store the occurrences of each label within the dataset.
	 * 
	 * @param data
	 *            DataSet
	 * @return HashMap with (label, occurrences) associations
	 */
	public HashMapCounter<Double> countLabels(DataSet data) {
		HashMapCounter<Double> hmc = new HashMapCounter<Double>();
		// for each label, store a count of occurences of this label in dataset
		for (double l : data.getLabels()) {
			for (Example ex : data.getData()) {
				if (ex.getLabel() == l) {

					hmc.increment(l);

				}
			}
		}
		return hmc;
	}

	/**
	 * Method which classifies a given example using the equation: label =
	 * argmax_{y in labels} log(p(y)) + sum(log(p(x_i | y)))
	 * 
	 * @param example
	 *            Example to be classified
	 * @return classification for example
	 */
	@Override
	public double classify(Example example) {
		double max = -100000000000.0;
		double curLabel = 0.0;
		for (double label : labelCounts.keySet()) {
			double logProb = getLogProb(example, label);
			// System.out.println("logprob: " + logProb);
			if (max < logProb) {
				max = logProb;
				curLabel = label;
			}
		}
		// System.out.println("max: "+ max + " curLabel: " + curLabel);
		return curLabel;
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
		// store all log probabilities in an ArrayList and return the maximum
		ArrayList<Double> probs = new ArrayList<Double>();
		for (double label : labelCounts.keySet()) {
			probs.add(getLogProb(example, label));
		}
		return Collections.max(probs);
	}

	/**
	 * Return the log probability of the example with the label under the
	 * current trained model.
	 * 
	 * @param ex
	 *            Example for which to calculate log probability.
	 * @param label
	 *            Label for which to calculate log probability.
	 * @return p(x_1, x_2,...,x_m, y)
	 */
	public double getLogProb(Example ex, double label) {
		double labelCount = labelCounts.get(label);
		double probY = labelCount / (double) allExamples;// labelCounts.keySet().size();
		Set<Integer> features = ex.getFeatureSet();
		if (usePosOnly) {
			double sum = 0.0;
			for (int f : features) {
				sum += Math.log(getFeatureProb(f, label));
			}
			return Math.log(probY) + sum;
		} else {
			// run through all features in dataset and add 1-p(xi|y) to sum if
			// feature not in ex feature set
			double sum = 0.0;
			for (int f : data.getAllFeatureIndices()) {
				if (features.contains(f)) {
					sum += Math.log(getFeatureProb(f, label));
				} else {
					sum += Math.log(1 - getFeatureProb(f, label));
				}
			}
			return Math.log(probY) + sum;
		}
	}

	/**
	 * Return the probability of a given feature index given a label
	 * 
	 * @param featureIndex
	 * @param label
	 * @return
	 */
	public double getFeatureProb(int featureIndex, double label) {
		double labelCount = labelCounts.get(label);
		double featureLabelCount = featureLabelCounts.get((int) label).get(featureIndex);
		// P(x_i | y) = P(x_i and y) + lambda / P(y) + (# of possible values of
		// x_i) * lambda
		double prob = (featureLabelCount + lambda) / (labelCount + 2 * lambda);// (double)
																				// featureLabelCounts.get((int)
																				// label).size()
		// System.out.println("returned: " + prob);
		return prob;
	}

	public static void main(String[] args) {
		NBClassifier c = new NBClassifier();
		String file = "/Users/maddie/Documents/FALL2016/MachineLearning/hw5/wines.train.txt";
		DataSet data = new DataSet(file, 1);
		CrossValidationSet cs = new CrossValidationSet(data, 10, true);
		for (int i = 0; i < cs.getNumSplits(); i++) {
			double avg = 0.0;
			DataSetSplit dss = cs.getValidationSet(i);

			// for (int iter = 0; iter < 100; iter++) {
			c.train(dss.getTrain());
			double acc = 0.0;
			double size = dss.getTest().getData().size();
			for (Example ex : dss.getTest().getData()) {
				if (c.classify(ex) == ex.getLabel()) {
					acc += 1.0;
				}
			}
			System.out.println("acc: " + acc / size);

			// avg += acc / 100.0;
			// }
			// System.out.println(avg);
		}

	}

}
