#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>  // To establish the seed srand() and generate pseudorandom numbers rand()

#include "util.h"

using namespace std;
using namespace util;


// ------------------------------
// Obtain an integer random number in the range [Low,High]
int util::randomInt(int Low, int High)
{
	return rand() % (High-Low+1) + Low;
}

// ------------------------------
// Obtain a real random number in the range [Low,High]
double util::randomDouble(double Low, double High)
{
	return ((double) rand() / RAND_MAX) * (High-Low) + Low;
}

// ------------------------------
// Read a dataset from a file name and return it
Dataset *util::readData(const char *fileName)
{

    ifstream myFile(fileName); // Create an input stream

    if (!myFile.is_open())
    {
        cout << "ERROR: I cannot open the file " << fileName << endl;
        return NULL;
    }

    Dataset *dataset = new Dataset;
    if (dataset == NULL)
        return NULL;

    string line;
    int i, j;

    if (myFile.good())
    {
        getline(myFile, line); // Read a line
        istringstream iss(line);
        iss >> dataset->nOfInputs;
        iss >> dataset->nOfOutputs;
        iss >> dataset->nOfPatterns;
    }
    dataset->inputs = new double *[dataset->nOfPatterns];
    dataset->outputs = new double *[dataset->nOfPatterns];

    for (i = 0; i < dataset->nOfPatterns; i++)
    {
        dataset->inputs[i] = new double[dataset->nOfInputs];
        dataset->outputs[i] = new double[dataset->nOfOutputs];
    }

    i = 0;
    while (myFile.good())
    {
        getline(myFile, line); // Read a line
        if (!line.empty())
        {
            istringstream iss(line);
            for (j = 0; j < dataset->nOfInputs; j++)
            {
                double value;
                iss >> value;
                if (!iss)
                    return NULL;
                dataset->inputs[i][j] = value;
            }
            for (j = 0; j < dataset->nOfOutputs; j++)
            {
                double value;
                iss >> value;
                if (!iss)
                    return NULL;
                dataset->outputs[i][j] = value;
            }
            i++;
        }
    }

    myFile.close();

    return dataset;
}


// ------------------------------
// Print the dataset
void util::printDataset(Dataset *dataset, int len)
{
    if (len == 0)
        len = dataset->nOfPatterns;

    for (int i = 0; i < len; i++)
    {
        cout << "P" << i << ":" << endl;
        for (int j = 0; j < dataset->nOfInputs; j++)
        {
            cout << dataset->inputs[i][j] << ",";
        }

        for (int j = 0; j < dataset->nOfOutputs; j++)
        {
            cout << dataset->outputs[i][j] << ",";
        }
        cout << endl;
    }
}

// ------------------------------
// Transform an scalar x by scaling it to a given range [minAllowed, maxAllowed] considering the min
// and max values of the feature in the dataset (minData and maxData). 
double util::minMaxScaler(double x, double minAllowed, double maxAllowed, double minData, double maxData)
{
    return minAllowed + (((x - minData)*(maxAllowed - minAllowed)) / (maxData - minData));
}

// ------------------------------
// Scale the dataset inputs to a given range [minAllowed, maxAllowed] considering the min
// and max values of the feature in the dataset (minData and maxData). 
void util::minMaxScalerDataSetInputs(Dataset *dataset, double minAllowed, double maxAllowed,
                                     double *minData, double *maxData)
{
    int nOfPatterns = dataset->nOfPatterns;
    int nOfInputs = dataset->nOfInputs;

    for(int i = 0; i < nOfPatterns; i++){
        for(int j = 0; j < nOfInputs; j++){
            (dataset->inputs)[i][j] = minMaxScaler((dataset->inputs)[i][j], minAllowed, maxAllowed, *minData, *maxData);
        }
    }
}

// ------------------------------
// Scale the dataset output vector to a given range [minAllowed, maxAllowed] considering the min
// and max values of the feature in the dataset (minData and maxData). Only for regression problems. 
void util::minMaxScalerDataSetOutputs(Dataset *dataset, double minAllowed, double maxAllowed,
                                      double minData, double maxData)
{
    int nOfPatterns = dataset->nOfPatterns;
    int nOfOutputs = dataset->nOfOutputs;

    for(int i = 0; i < nOfPatterns; i++){
        for(int j = 0; j < nOfOutputs; j++){
            (dataset->outputs)[i][j] = minMaxScaler((dataset->outputs)[i][j], minAllowed, maxAllowed, minData, maxData);
        }
    }
}

// ------------------------------
// Get a vector of minimum values of the dataset inputs
double *util::minDatasetInputs(Dataset *dataset)
{
    int nOfPatterns = dataset->nOfPatterns;
    int nOfInputs = dataset->nOfInputs;
    double min = (dataset->inputs)[0][0];

    for(int i = 0; i < nOfPatterns; i++){
        for(int j = 0; j < nOfInputs; j++){
            if((dataset->inputs)[i][j] < min){
                min = (dataset->inputs)[i][j];
            }
        }
    }

    double * min_aux = new double(min);

    return min_aux;
}

// ------------------------------
// Get a vector of maximum values of the dataset inputs
double *util::maxDatasetInputs(Dataset *dataset)
{
    int nOfPatterns = dataset->nOfPatterns;
    int nOfInputs = dataset->nOfInputs;
    double max = (dataset->inputs)[0][0];

    for(int i = 0; i < nOfPatterns; i++){
        for(int j = 0; j < nOfInputs; j++){
            if((dataset->inputs)[i][j] > max){
                max = (dataset->inputs)[i][j];
            }
        }
    }

    double * max_aux = new double(max);

    return max_aux;
}

// ------------------------------
// Get the minimum value of the dataset outputs
double util::minDatasetOutputs(Dataset *dataset)
{
    int nOfPatterns = dataset->nOfPatterns;
    int nOfOutputs = dataset->nOfOutputs;
    double min = (dataset->outputs)[0][0];

    for(int i = 0; i < nOfPatterns; i++){
        for(int j = 0; j < nOfOutputs; j++){
            if((dataset->outputs)[i][j] < min){
                min = (dataset->outputs)[i][j];
            }
        }
    }

    return min;
}

// ------------------------------
 // Get the maximum value of the dataset outputs
double util::maxDatasetOutputs(Dataset *dataset)
{
    int nOfPatterns = dataset->nOfPatterns;
    int nOfOutputs = dataset->nOfOutputs;
    double max = (dataset->outputs)[0][0];

    for(int i = 0; i < nOfPatterns; i++){
        for(int j = 0; j < nOfOutputs; j++){
            if((dataset->outputs)[i][j] > max){
                max = (dataset->outputs)[i][j];
            }
        }
    }

    return max;
}

void util::plotData(const std::vector<double>& trainingErrors, const std::vector<double>& testErrors) {
    std::ofstream dataFile("errors_data.dat");
    for (size_t i = 0; i < trainingErrors.size(); ++i) {
        dataFile << i << " " << trainingErrors[i] << " " << testErrors[i] << "\n";
    }
    dataFile.close();

    // Enviar comandos a gnuplot
    FILE *gnuplotPipe = popen("gnuplot -persistent", "w");
    fprintf(gnuplotPipe, "set title 'Training and Test Error'\n");
    fprintf(gnuplotPipe, "set xlabel 'Iterations'\n");
    fprintf(gnuplotPipe, "set ylabel 'Error'\n");
    fprintf(gnuplotPipe, "plot 'errors_data.dat' using 1:2 title 'Training Error' with lines, 'errors_data.dat' using 1:3 title 'Test Error' with lines\n");
    fflush(gnuplotPipe);
}