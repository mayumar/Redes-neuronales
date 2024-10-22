//============================================================================
// Introduction to computational models
// Name        : la1.cpp
// Author      : Pedro A. Gutiérrez
// Version     :
// Copyright   : Universidad de Córdoba
//============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <ctime>    // To obtain current time time()
#include <cstdlib>  // To establish the seed srand() and generate pseudorandom numbers rand()
#include <string.h>
#include <math.h>
#include <float.h>

#include "imc/MultilayerPerceptron.h"
#include "imc/util.h"


using namespace imc;
using namespace std;
using namespace util;

int main(int argc, char **argv) {
    // Process arguments of the command line
    bool tflag=0, Tflag = 0, iflag=0, lflag=0, hflag=0, eflag=0, mflag=0, sflag=0, wflag = 0, pflag = 0;
    char *tvalue=NULL, *Tvalue = NULL, *ivalue=NULL, *lvalue=NULL, *hvalue=NULL, *evalue=NULL, *mvalue=NULL, *wvalue = NULL;
    int c;

    opterr = 0;

    // a: Option that requires an argument
    // a:: The argument required is optional
    while ((c = getopt(argc, argv, "t:T:i:l:h:e:m:sw:p")) != -1)
    {
        // The parameters needed for using the optional prediction mode of Kaggle have been included.
        // You should add the rest of parameters needed for the lab assignment.
        switch(c){
            case 't':
                tflag = true;
                tvalue = optarg;
                break;
            case 'T':
                Tflag = true;
                Tvalue = optarg;
                break;
            case 'i':
                iflag = true;
                ivalue = optarg;
                break;
            case 'l':
                lflag = true;
                lvalue = optarg;
                break;
            case 'h':
                hflag = true;
                hvalue = optarg;
                break;
            case 'e':
                eflag = true;
                evalue = optarg;
                break;
            case 'm':
                mflag = true;
                mvalue = optarg;
                break;
            case 's':
                sflag = true;
                break;
            case 'w':
                wflag = true;
                wvalue = optarg;
                break;
            case 'p':
                pflag = true;
                break;
            case '?':
                if (optopt == 't' || optopt == 'T' || optopt == 'i' || optopt == 'h' || optopt == 'e' || optopt == 'w' || optopt == 'p')
                    fprintf (stderr, "The option -%c requires an argument.\n", optopt);
                else if (isprint (optopt))
                    fprintf (stderr, "Unknown option `-%c'.\n", optopt);
                else
                    fprintf (stderr,
                             "Unknown character `\\x%x'.\n",
                             optopt);
                return EXIT_FAILURE;
            default:
                return EXIT_FAILURE;
        }
    }

    if (!pflag) {
        //////////////////////////////////
        // TRAINING AND EVALUATION MODE //
        //////////////////////////////////

        // Multilayer perceptron object
    	MultilayerPerceptron mlp;

        // Parameters of the mlp.
        int iterations = 1000;
        if(iflag)
            iterations = atoi(ivalue);

        float eta = 0.1;
        if(eflag)
            eta = atof(evalue);

        float mu = 0.9;
        if(mflag)
            mu = atof(mvalue);


        // Read training and test data
        Dataset * trainDataset = NULL;
        if(!tflag){
            fprintf(stderr, "Training data is required.\n");
            return EXIT_FAILURE;
        }else{
        	trainDataset = util::readData(tvalue);
        }

        Dataset * testDataset = NULL;
        if(Tflag)
        	testDataset = util::readData(Tvalue);
        else
            testDataset = util::readData(tvalue);


        // Initialize topology vector
    	int layers = 1;
        if(lflag)
            layers = atoi(lvalue);

    	int * topology = new int[layers+2];
        
        int nin = trainDataset->nOfInputs;
        int nout = trainDataset->nOfOutputs;

        int neurons = 5;
        if(hflag)
            neurons = atoi(hvalue);

        for(int i = 1; i <= layers; i++){
            topology[i] = neurons;
            //cout << "topology[" << i << "] = " << neurons << endl;
        }

        topology[0] = nin;
        topology[layers+1] = nout;

        // Initialize the network using the topology vector
        mlp.initialize(layers+2,topology);
        mlp.eta=eta;
        mlp.mu=mu;

        // Seed for random numbers
        int seeds[] = {1,2,3,4,5};
        double *testErrors = new double[5];
        double *trainErrors = new double[5];
        double bestTestError = DBL_MAX;
        for(int i=0; i<5; i++){
            cout << "**********" << endl;
            cout << "SEED " << seeds[i] << endl;
            cout << "**********" << endl;
            srand(seeds[i]);
            mlp.runOnlineBackPropagation(trainDataset,testDataset,iterations,&(trainErrors[i]),&(testErrors[i]));
            cout << "We end!! => Final test error: " << testErrors[i] << endl;

            // We save the weights every time we find a better model
            if(wflag && testErrors[i] <= bestTestError)
            {
                mlp.saveWeights(wvalue);
                bestTestError = testErrors[i];
            }
        }

        cout << "WE HAVE FINISHED WITH ALL THE SEEDS" << endl;

        double averageTestError = 0, stdTestError = 0;
        double averageTrainError = 0, stdTrainError = 0;
        
        // Obtain training and test averages and standard deviations

        cout << "FINAL REPORT" << endl;
        cout << "************" << endl;
        cout << "Train error (Mean +- SD): " << averageTrainError << " +- " << stdTrainError << endl;
        cout << "Test error (Mean +- SD):          " << averageTestError << " +- " << stdTestError << endl;
        return EXIT_SUCCESS;
    }
    else {

        //////////////////////////////
        // PREDICTION MODE (KAGGLE) //
        //////////////////////////////
        
        // Multilayer perceptron object
        MultilayerPerceptron mlp;

        // Initializing the network with the topology vector
        if(!wflag || !mlp.readWeights(wvalue))
        {
            cerr << "Error while reading weights, we can not continue" << endl;
            exit(-1);
        }

        // Reading training and test data: call to util::readData(...)
        Dataset *testDataset;
        testDataset = readData(Tvalue);
        if(testDataset == NULL)
        {
            cerr << "The test file is not valid, we can not continue" << endl;
            exit(-1);
        }

        mlp.predict(testDataset);

        return EXIT_SUCCESS;
    }

    
}

