/*********************************************************************
* File  : MultilayerPerceptron.cpp
* Date  : 2020
*********************************************************************/

#include "MultilayerPerceptron.h"

#include "util.h"


#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib>  // To establish the seed srand() and generate pseudorandom numbers rand()
#include <limits>
#include <math.h>


using namespace imc;
using namespace std;
using namespace util;

// ------------------------------
// Constructor: Default values for all the parameters
MultilayerPerceptron::MultilayerPerceptron()
{
	nOfLayers = 0;
	layers = nullptr;
	eta = 0.0;
	mu = 0.0;
	online = 0;
	outputFunction = 0;

}


// ------------------------------
// Allocate memory for the data structures
// nl is the number of layers and npl is a vetor containing the number of neurons in every layer
// Give values to Layer* layers
int MultilayerPerceptron::initialize(int nl, int npl[]) {
	nOfLayers = nl;
	layers = new Layer[nl];

	for(int i = 0; i < nl; i++){
		layers[i].nOfNeurons = npl[i];
		// cout << "npl[" << i << "] = " << npl[i] << endl;
		layers[i].neurons = new Neuron[npl[i]];

		for(int j = 0; j < npl[i]; j++){
			if(i > 0){
				layers[i].neurons[j].out = 0.0;
				layers[i].neurons[j].delta = 0.0;
				layers[i].neurons[j].w = new double[npl[i-1]+1];
				layers[i].neurons[j].deltaW = new double[npl[i-1]+1];
				layers[i].neurons[j].lastDeltaW = new double[npl[i-1]+1];
				layers[i].neurons[j].wCopy = new double[npl[i-1]+1];
			}else{
				layers[i].neurons[j].out = 0.0;
				layers[i].neurons[j].delta = 0.0;
				layers[i].neurons[j].w = nullptr;
				layers[i].neurons[j].deltaW = nullptr;
				layers[i].neurons[j].lastDeltaW = nullptr;
				layers[i].neurons[j].wCopy = nullptr;
			}
		}
	}

	return 1;
}


// ------------------------------
// DESTRUCTOR: free memory
MultilayerPerceptron::~MultilayerPerceptron() {
	freeMemory();
}


// ------------------------------
// Free memory for the data structures
void MultilayerPerceptron::freeMemory() {
	for(int i=0; i<nOfLayers;i++){
		for(int j=0; j<layers[i].nOfNeurons; j++){ 
			delete[] layers[i].neurons[j].w; 
			delete[] layers[i].neurons[j].deltaW; 
			delete[] layers[i].neurons[j].wCopy; 
		}
		delete[] layers[i].neurons; 

	}
	delete[] layers; 
}

// ------------------------------
// Fill all the weights (w) with random numbers between -1 and +1
void MultilayerPerceptron::randomWeights() {
	for(int i = 1; i < nOfLayers; i++){
		for(int j = 0; j < layers[i].nOfNeurons; j++){
			for(int k = 0; k < layers[i-1].nOfNeurons+1; k++){
				layers[i].neurons[j].w[k] = util::randomDouble(-1.0, 1.0);
			}
		}
	}
}

// ------------------------------
// Feed the input neurons of the network with a vector passed as an argument
void MultilayerPerceptron::feedInputs(double* input) {
	for(int i = 0; i < layers[0].nOfNeurons; i++){
		layers[0].neurons[i].out = input[i];
	}
}

// ------------------------------
// Get the outputs predicted by the network (out vector of the output layer) and save them in the vector passed as an argument
void MultilayerPerceptron::getOutputs(double* output)
{
	for(int i = 0; i < layers[nOfLayers-1].nOfNeurons; i++){
		output[i] = layers[nOfLayers-1].neurons[i].out;
	}
}

// ------------------------------
// Make a copy of all the weights (copy w in wCopy)
void MultilayerPerceptron::copyWeights() {
	for(int i = 1; i < nOfLayers; i++){
		for(int j = 0; j < layers[i].nOfNeurons; j++){
			for(int k = 0; k < layers[i-1].nOfNeurons+1; k++){
				layers[i].neurons[j].wCopy[k] = layers[i].neurons[j].w[k];
			}
		}
	}
}

// ------------------------------
// Restore a copy of all the weights (copy wCopy in w)
void MultilayerPerceptron::restoreWeights() {
	for(int i = 1; i < nOfLayers; i++){
		for(int j = 0; j < layers[i].nOfNeurons; j++){
			for(int k = 0; k < layers[i-1].nOfNeurons+1; k++){
				layers[i].neurons[j].w[k] = layers[i].neurons[j].wCopy[k];
			}
		}
	}
}

// ------------------------------
// Calculate and propagate the outputs of the neurons, from the first layer until the last one -->-->
void MultilayerPerceptron::forwardPropagate() {
	double net;

	for(int h = 1; h < nOfLayers-1; h++){ // Todas las capas menos la ultima
		for(int j = 0; j < layers[h].nOfNeurons; j++){
			net = layers[h].neurons[j].w[layers[h-1].nOfNeurons];
			for(int i = 0; i < layers[h-1].nOfNeurons; i++){
				net += layers[h].neurons[j].w[i] * layers[h-1].neurons[i].out;
			}

			layers[h].neurons[j].out = 1.0 / (1.0 + exp(-net));
		}
	}

	// Ultima capa
	int H = nOfLayers-1;
	std::vector<double> nets;
	for(int j = 0; j < layers[H].nOfNeurons; j++){
		nets.push_back(layers[H].neurons[j].w[layers[H-1].nOfNeurons]);
		for(int i = 0; i < layers[H-1].nOfNeurons; i++){
			nets[j] += layers[H].neurons[j].w[i] * layers[H-1].neurons[i].out;
		}

		if(outputFunction == 0)
			layers[H].neurons[j].out = 1.0 / (1.0 + exp(-nets[j]));

	}

	if(outputFunction == 1){
		for(int i = 0; i < layers[H].nOfNeurons; i++){
			double exp_sum = 0.0;
			for(int j = 0; j < layers[H].nOfNeurons; j++){
				exp_sum += exp(nets[j]);
			}
			layers[H].neurons[i].out = exp(nets[i]) / exp_sum;
		}
	}
}

// ------------------------------
// Obtain the output error (MSE) of the out vector of the output layer wrt a target vector and return it
// errorFunction=1 => Cross Entropy // errorFunction=0 => MSE
double MultilayerPerceptron::obtainError(double* target, int errorFunction) {
	double error = 0.0;
	
	if(errorFunction == 0){ // MSE
		for(int i = 0; i < layers[nOfLayers-1].nOfNeurons; i++){
			double difference = target[i] - layers[nOfLayers-1].neurons[i].out;
			error += (difference*difference);
		}

		error /= layers[nOfLayers-1].nOfNeurons;
		
	}else if(errorFunction == 1){ // Cross Entropy

	}

	return error;
}


// ------------------------------
// Backpropagate the output error wrt a vector passed as an argument, from the last layer to the first one <--<--
// errorFunction=1 => Cross Entropy // errorFunction=0 => MSE
void MultilayerPerceptron::backpropagateError(double* target, int errorFunction) {

	int lastLayer = nOfLayers-1;
	for(int j = 0; j < layers[lastLayer].nOfNeurons; j++){
		double llout = layers[lastLayer].neurons[j].out;

		if(outputFunction == 0){ // sigmoide
			if(errorFunction == 0) // MSE
				layers[lastLayer].neurons[j].delta = -(target[j] - llout) * llout * (1 - llout);
			else if(errorFunction == 1) // Cross Entropy
				layers[lastLayer].neurons[j].delta = -(target[j] / llout) * llout * (1 - llout);

		}else if(outputFunction == 1){ // softmax
			layers[lastLayer].neurons[j].delta = 0.0;
			for(int i = 0; i < layers[lastLayer].nOfNeurons; i++){
				if(errorFunction == 0) // MSE
					layers[lastLayer].neurons[j].delta -= (target[j] - llout) * llout * ((i == j) - llout);
				else if(errorFunction == 1) // Cross Entropy
					layers[lastLayer].neurons[j].delta -= (target[j] / llout) * llout * ((i == j) - llout);
			}
		}
	}

	for(int h = lastLayer-1; h > 0; h--){
		for(int j = 0; j < layers[h].nOfNeurons; j++){
			double sum = 0.0;
			for(int i = 0; i < layers[h+1].nOfNeurons; i++){
				sum += layers[h+1].neurons[i].w[j] * layers[h+1].neurons[i].delta;
			}
			layers[h].neurons[j].delta = sum * layers[h].neurons[j].out * (1 - layers[h].neurons[j].out);
		}
	}
}

// ------------------------------
// Accumulate the changes produced by one pattern and save them in deltaW
void MultilayerPerceptron::accumulateChange() {
	for(int h = 1; h < nOfLayers; h++){
		for(int j = 0; j < layers[h].nOfNeurons; j++){
			for(int i = 0; i < layers[h-1].nOfNeurons; i++){
				layers[h].neurons[j].deltaW[i] += (layers[h].neurons[j].delta * layers[h-1].neurons[i].out);
			}
			int biasIndex = layers[h-1].nOfNeurons;
			layers[h].neurons[j].deltaW[biasIndex] += layers[h].neurons[j].delta * 1;
		}
	}
}

// ------------------------------
// Update the network weights, from the first layer to the last one
void MultilayerPerceptron::weightAdjustment() {
	for(int h = 1; h < nOfLayers; h++){
		for(int j = 0; j < layers[h].nOfNeurons; j++){
			for(int i = 0; i < layers[h-1].nOfNeurons; i++){
				if(online)
					layers[h].neurons[j].w[i] -= (eta * layers[h].neurons[j].deltaW[i]) + mu * (eta * layers[h].neurons[j].lastDeltaW[i]);
				else
					layers[h].neurons[j].w[i] -= ((eta * layers[h].neurons[j].deltaW[i]) / nOfTrainingPatterns) + (mu * (eta * layers[h].neurons[j].lastDeltaW[i]) / nOfTrainingPatterns);
				layers[h].neurons[j].lastDeltaW[i] = layers[h].neurons[j].deltaW[i];
			}
			int biasIndex = layers[h-1].nOfNeurons;
			if(online)
				layers[h].neurons[j].w[biasIndex] -= (eta * layers[h].neurons[j].deltaW[biasIndex]) + mu * (eta * layers[h].neurons[j].lastDeltaW[biasIndex]);
			else
				layers[h].neurons[j].w[biasIndex] -= ((eta * layers[h].neurons[j].deltaW[biasIndex]) / nOfTrainingPatterns) + (mu * (eta * layers[h].neurons[j].lastDeltaW[biasIndex]) / nOfTrainingPatterns);
			layers[h].neurons[j].lastDeltaW[biasIndex] = layers[h].neurons[j].deltaW[biasIndex];
		}
	}
}

// ------------------------------
// Print the network, i.e. all the weight matrices
void MultilayerPerceptron::printNetwork() {
	for(int i = 1; i < nOfLayers; i++){
		cout << "Layer " << i << " weights: " << endl;

		for(int j = 0; j < layers[i].nOfNeurons; j++){
			cout << "Neuron " << j << " weights: ";

			for(int k = 0; k < layers[i-1].nOfNeurons+1; k++){
				cout << layers[i].neurons[j].w[k] << " ";
			}

			cout << endl;
		}

		cout << endl;
	}
}

// ------------------------------
// Perform an epoch: forward propagate the inputs, backpropagate the error and adjust the weights
// input is the input vector of the pattern and target is the desired output vector of the pattern
// The step of adjusting the weights must be performed only in the online case
// If the algorithm is offline, the weightAdjustment must be performed in the "train" function
// errorFunction=1 => Cross Entropy // errorFunction=0 => MSE
void MultilayerPerceptron::performEpoch(double* input, double* target, int errorFunction) {

	if(online){
		for(int i = 1; i < nOfLayers; i++){
			for(int j = 0; j < layers[i].nOfNeurons; j++){
				for(int k = 0; k < layers[i-1].nOfNeurons+1; k++){
					layers[i].neurons[j].deltaW[k] = 0.0;
				}
			}
		}
	}

	feedInputs(input);
	forwardPropagate();
	backpropagateError(target, errorFunction);
	accumulateChange();

	if(online)
		weightAdjustment();

}

// ------------------------------
// Train the network for a dataset (one iteration of the external loop)
// errorFunction=1 => Cross Entropy // errorFunction=0 => MSE
void MultilayerPerceptron::train(Dataset* trainDataset, int errorFunction) {
	
	if(!online){
		for(int i = 1; i < nOfLayers; i++){
			for(int j = 0; j < layers[i].nOfNeurons; j++){
				for(int k = 0; k < layers[i-1].nOfNeurons+1; k++){
					layers[i].neurons[j].deltaW[k] = 0.0;
				}
			}
		}
	}

	for(int i = 0; i < trainDataset->nOfPatterns; i++){
		performEpoch(trainDataset->inputs[i], trainDataset->outputs[i], errorFunction);
	}

	if(!online)
		weightAdjustment();
}

// ------------------------------
// Test the network with a dataset and return the error
// errorFunction=1 => Cross Entropy // errorFunction=0 => MSE
double MultilayerPerceptron::test(Dataset* dataset, int errorFunction) {
	/* online
	double mse = 0.0, difference;

	for(int i = 0; i < testDataset->nOfPatterns; i++){
		feedInputs(testDataset->inputs[i]);
		forwardPropagate();
		for(int j = 0; j < testDataset->nOfOutputs; j++){
			difference = testDataset->outputs[i][j] - layers[nOfLayers-1].neurons[j].out;
			mse += difference*difference;
		}
	}
	
	double out = mse/testDataset->nOfPatterns;
	return out;
	*/
}


// ------------------------------
// Test the network with a dataset and return the CCR
double MultilayerPerceptron::testClassification(Dataset* dataset) {
	/* online
	double mse = 0.0, difference;

	for(int i = 0; i < testDataset->nOfPatterns; i++){
		feedInputs(testDataset->inputs[i]);
		forwardPropagate();
		for(int j = 0; j < testDataset->nOfOutputs; j++){
			difference = testDataset->outputs[i][j] - layers[nOfLayers-1].neurons[j].out;
			mse += difference*difference;
		}
	}
	
	double out = mse/testDataset->nOfPatterns;
	return out;
	*/
}


// ------------------------------
// Optional Kaggle: Obtain the predicted outputs for a dataset
void MultilayerPerceptron::predict(Dataset* dataset)
{
	int i;
	int j;
	int numSalidas = layers[nOfLayers-1].nOfNeurons;
	double * salidas = new double[numSalidas];
	
	cout << "Id,Category" << endl;
	
	for (i=0; i<dataset->nOfPatterns; i++){

		feedInputs(dataset->inputs[i]);
		forwardPropagate();
		getOutputs(salidas);

		int maxIndex = 0;
		for (j = 0; j < numSalidas; j++)
			if (salidas[j] >= salidas[maxIndex])
				maxIndex = j;
		
		cout << i << "," << maxIndex << endl;

	}
}



// ------------------------------
// Run the traning algorithm for a given number of epochs, using trainDataset
// Once finished, check the performance of the network in testDataset
// Both training and test MSEs should be obtained and stored in errorTrain and errorTest
// Both training and test CCRs should be obtained and stored in ccrTrain and ccrTest
// errorFunction=1 => Cross Entropy // errorFunction=0 => MSE
void MultilayerPerceptron::runBackPropagation(Dataset * trainDataset, Dataset * testDataset, int maxiter, double *errorTrain, double *errorTest, double *ccrTrain, double *ccrTest, int errorFunction)
{
	int countTrain = 0;

	// Random assignment of weights (starting point)
	randomWeights();

	double minTrainError = 0;
	int iterWithoutImproving = 0;
	nOfTrainingPatterns = trainDataset->nOfPatterns;


	// Learning
	do {

		train(trainDataset,errorFunction);

		double trainError = test(trainDataset,errorFunction);
		if(countTrain==0 || trainError < minTrainError){
			minTrainError = trainError;
			copyWeights();
			iterWithoutImproving = 0;
		}
		else if( (trainError-minTrainError) < 0.00001)
			iterWithoutImproving = 0;
		else
			iterWithoutImproving++;

		if(iterWithoutImproving==50){
			cout << "We exit because the training is not improving!!"<< endl;
			restoreWeights();
			countTrain = maxiter;
		}

		countTrain++;

		cout << "Iteration " << countTrain << "\t Training error: " << trainError << endl;

	} while ( countTrain<maxiter );

	if ( iterWithoutImproving!=50)
		restoreWeights();

	cout << "NETWORK WEIGHTS" << endl;
	cout << "===============" << endl;
	printNetwork();

	cout << "Desired output Vs Obtained output (test)" << endl;
	cout << "=========================================" << endl;
	for(int i=0; i<testDataset->nOfPatterns; i++){
		double* prediction = new double[testDataset->nOfOutputs];

		// Feed the inputs and propagate the values
		feedInputs(testDataset->inputs[i]);
		forwardPropagate();
		getOutputs(prediction);
		for(int j=0; j<testDataset->nOfOutputs; j++)
			cout << testDataset->outputs[i][j] << " -- " << prediction[j] << " ";
		cout << endl;
		delete[] prediction;

	}

	*errorTest=test(testDataset,errorFunction);;
	*errorTrain=minTrainError;
	*ccrTest = testClassification(testDataset);
	*ccrTrain = testClassification(trainDataset);

}

// -------------------------
// Optional Kaggle: Save the model weights in a textfile
bool MultilayerPerceptron::saveWeights(const char * fileName)
{
	// Object for writing the file
	ofstream f(fileName);

	if(!f.is_open())
		return false;

	// Write the number of layers and the number of layers in every layer
	f << nOfLayers;

	for(int i = 0; i < nOfLayers; i++)
	{
		f << " " << layers[i].nOfNeurons;
	}
	f << " " << outputFunction;
	f << endl;

	// Write the weight matrix of every layer
	for(int i = 1; i < nOfLayers; i++)
		for(int j = 0; j < layers[i].nOfNeurons; j++)
			for(int k = 0; k < layers[i-1].nOfNeurons + 1; k++)
				if(layers[i].neurons[j].w!=NULL)
				    f << layers[i].neurons[j].w[k] << " ";

	f.close();

	return true;

}


// -----------------------
// Optional Kaggle: Load the model weights from a textfile
bool MultilayerPerceptron::readWeights(const char * fileName)
{
	// Object for reading a file
	ifstream f(fileName);

	if(!f.is_open())
		return false;

	// Number of layers and number of neurons in every layer
	int nl;
	int *npl;

	// Read number of layers
	f >> nl;

	npl = new int[nl];

	// Read number of neurons in every layer
	for(int i = 0; i < nl; i++)
	{
		f >> npl[i];
	}
	f >> outputFunction;

	// Initialize vectors and data structures
	initialize(nl, npl);

	// Read weights
	for(int i = 1; i < nOfLayers; i++)
		for(int j = 0; j < layers[i].nOfNeurons; j++)
			for(int k = 0; k < layers[i-1].nOfNeurons + 1; k++)
				if(!(outputFunction==1 && (i==(nOfLayers-1)) && (j==(layers[i].nOfNeurons-1))))
					f >> layers[i].neurons[j].w[k];

	f.close();
	delete[] npl;

	return true;
}
