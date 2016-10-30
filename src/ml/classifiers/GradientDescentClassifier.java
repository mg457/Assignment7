package ml.classifiers;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Set;
import java.util.Random;

import ml.data.*;

/**
 * Gradient descent classifier allowing for two different loss functions and
 * three different regularization settings.
 * 
 * @author Maddie Gordon, Nick Reminder
 *
 */
public class GradientDescentClassifier implements Classifier {
	// constants for the different surrogate loss functions
	public static final int EXPONENTIAL_LOSS = 0;
	public static final int HINGE_LOSS = 1;
	public static final int SQUARED_LOSS = 2;

	// constants for the different regularization parameters
	public static final int NO_REGULARIZATION = 0;
	public static final int L1_REGULARIZATION = 1;
	public static final int L2_REGULARIZATION = 2;

	protected HashMap<Integer, Double> weights; // the feature weights
	protected double b = 0; // the intersect weight

	protected int iterations = 10;

	protected int lossFun = EXPONENTIAL_LOSS; // specifies loss function to use
												// in training
	protected double lambda = 0.1;
	protected int regularization = NO_REGULARIZATION; // specifies
														// regularization method
														// to use in training
	protected double eta = 0.1;

	/**
	 * Select which loss function to use
	 * 
	 * @param functionNum
	 *            the number associated with a specific loss function
	 */
	protected void setLoss(int functionNum) {
		if (functionNum == 0) {
			lossFun = EXPONENTIAL_LOSS;
		} else if (functionNum == 1) {
			lossFun = HINGE_LOSS;
		}
	}

	/**
	 * Set the regularization method to be used.
	 * 
	 * @param regNum
	 *            the number associated with a specific regularization method
	 */
	protected void setRegularization(int regNum) {
		if (regNum == 0)
			regularization = NO_REGULARIZATION;
		else if (regNum == 1)
			regularization = L1_REGULARIZATION;
		else if (regNum == 2)
			regularization = L2_REGULARIZATION;
	}

	/**
	 * Set lambda to a new value
	 * 
	 * @param newVal
	 *            the new value of lambda
	 */
	protected void setLambda(double newVal) {
		lambda = newVal;
	}

	/**
	 * Set the eta value to use
	 * 
	 * @param newEta
	 *            the new value of eta
	 */
	protected void setEta(double newEta) {
		eta = newEta;
	}

	/**
	 * Get a weight vector over the set of features with each weight set to 0
	 * 
	 * @param features
	 *            the set of features to learn over
	 * @return
	 */
	protected HashMap<Integer, Double> getZeroWeights(Set<Integer> features) {
		HashMap<Integer, Double> temp = new HashMap<Integer, Double>();

		for (Integer f : features) {
			temp.put(f, 0.0);
		}

		return temp;
	}

	/**
	 * Initialize the weights and the intersect value
	 * 
	 * @param features
	 */
	protected void initializeWeights(Set<Integer> features) {
		weights = getZeroWeights(features);
		b = 0;
	}

	/**
	 * Set the number of iterations the perceptron should run during training
	 * 
	 * @param iterations
	 */
	public void setIterations(int iterations) {
		this.iterations = iterations;
	}

	/**
	 * Train the classifier on data using the gradient descent method with
	 * specified loss and regularization methods.
	 * 
	 * @param data
	 *            Set of data to train classifier on.
	 */
	public void train(DataSet data) {
		initializeWeights(data.getAllFeatureIndices());

		ArrayList<Example> training = (ArrayList<Example>) data.getData().clone();

		for (int it = 0; it < iterations; it++) {
			Collections.shuffle(training);

			for (Example e : training) {
				double label = e.getLabel();

				double dotProduct = 0.0;
				for (Integer index : e.getFeatureSet()) {
					dotProduct += e.getFeature(index) * weights.get(index);
				}

				// update the weights
				// for( Integer featureIndex: weights.keySet() ){
				double constant = computeConstant(label, dotProduct, b);
				for (Integer featureIndex : e.getFeatureSet()) {
					double oldWeight = weights.get(featureIndex);
					double featureValue = e.getFeature(featureIndex);

					// y_i*x_{ij}
					double update = featureValue * label * constant;
					double regularize = computeReg(oldWeight);
					weights.put(featureIndex, oldWeight + update - regularize);
				}

				// update b
				double bUpdate = label * constant;
				double bRegularize = computeReg(b);
				b += bUpdate - bRegularize;
			}
		}
	}

	/**
	 * Compute the regularization value
	 * 
	 * @param weight
	 *            value to be used in regularization computation
	 * @return regularization value based upon method selected
	 */
	protected double computeReg(double weight) {
		if (regularization == L1_REGULARIZATION)
			return eta * lambda * ((Math.abs(weight) == weight) ? 1 : -1); // return
																			// eta*lambda*sign(w_j)
		else if (regularization == L2_REGULARIZATION)
			return eta * lambda * (weight);
		else
			return 0.0; // no regularization
	}

	/**
	 * Compute the constant based on the chosen loss function for a given
	 * example
	 * 
	 * @param label
	 *            label of the example being considered
	 * @param dotProduct
	 *            (w * x_i)
	 * @param b
	 *            bias term
	 * @return
	 */
	protected double computeConstant(double label, double dotProduct, double b) {
		if (lossFun == EXPONENTIAL_LOSS) {
			// exp(-y_i*(w*x_i + b))
			return eta * Math.exp(-label * (dotProduct + b));
		} else if(lossFun == HINGE_LOSS)// hinge loss
			return eta * (((label * (dotProduct + b)) < 1) ? 1 : 0);
		// return 1 if (yy' < 1), 0 otherwise
		else //squared loss
			return Math.pow(label - (dotProduct + b), 2);
	}

	@Override
	public double classify(Example example) {
		return getPrediction(example);
	}

	@Override
	public double confidence(Example example) {
		return Math.abs(getDistanceFromHyperplane(example, weights, b));
	}

	/**
	 * Get the prediction from the current set of weights on this example
	 * 
	 * @param e
	 *            the example to predict
	 * @return
	 */
	protected double getPrediction(Example e) {
		return getPrediction(e, weights, b);
	}

	/**
	 * Get the prediction from the on this example from using weights w and
	 * inputB
	 * 
	 * @param e
	 *            example to predict
	 * @param w
	 *            the set of weights to use
	 * @param inputB
	 *            the b value to use
	 * @return the prediction
	 */
	protected static double getPrediction(Example e, HashMap<Integer, Double> w, double inputB) {
		double sum = getDistanceFromHyperplane(e, w, inputB);

		if (sum > 0) {
			return 1.0;
		} else if (sum < 0) {
			return -1.0;
		} else {
			return 0;
		}
	}

	protected static double getDistanceFromHyperplane(Example e, HashMap<Integer, Double> w, double inputB) {
		double sum = inputB;

		// for(Integer featureIndex: w.keySet()){
		// only need to iterate over non-zero features
		for (Integer featureIndex : e.getFeatureSet()) {
			sum += w.get(featureIndex) * e.getFeature(featureIndex);
		}

		return sum;
	}

	public String toString() {
		StringBuffer buffer = new StringBuffer();

		ArrayList<Integer> temp = new ArrayList<Integer>(weights.keySet());
		Collections.sort(temp);

		for (Integer index : temp) {
			buffer.append(index + ":" + weights.get(index) + " ");
		}

		return buffer.substring(0, buffer.length() - 1);
	}

	public static void main(String[] args) {
		// EXPERIMENTS
		GradientDescentClassifier c = new GradientDescentClassifier();
		 c.setLoss(1);
		 c.setRegularization(2);
		String csv = "/Users/maddie/Documents/FALL2016/MachineLearning/hw4/titanic-train.perc.csv";
		DataSet data = new DataSet(csv, 0);
		CrossValidationSet cs = new CrossValidationSet(data, 10, true);
		//for (double change1 = 0.01; change1 < 0.1; change1 += 0.01) {
			// double changeacc = 0.0;
		c.setEta(0.007);
		c.setLambda(0.007);
			//c.setEta(change1);
			//for (double change2 = 0.001; change2 < 0.01; change2 += 0.001) {
				double changeacc = 0.0;
			//	c.setLambda(change2);

				// DataSetSplit dss = data.split(.7);
				// for(int iter = 0; iter < 50; iter++) {

				// double avg = 0.0;
				for (int i = 0; i < cs.getNumSplits(); i++) {
					double avg = 0.0;
					DataSetSplit dss = cs.getValidationSet(i);

					for (int iter = 0; iter < 100; iter++) {
						c.train(dss.getTrain());
						double acc = 0.0;
						double size = dss.getTest().getData().size();
						for (Example ex : dss.getTest().getData()) {
							// System.out.println(ex.getLabel()+ " classify: " +
							// c.classify(ex));

							if (c.classify(ex) == ex.getLabel()) {
								acc += 1.0 / size;
							}
						}
						avg += acc / 100.0;
					}
					//changeacc += avg / 10;
					System.out.println(avg);
				}
				//System.out.println(changeacc); // 0.06,

			}

//		}

	//}
}
