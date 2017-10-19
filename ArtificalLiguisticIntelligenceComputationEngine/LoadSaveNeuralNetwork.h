#pragma once

#include <algorithm>
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iterator>


#include "ALICECore.cuh"
#define NumofNNInputs 6

using namespace std;


class NeuralNetworkModule {
public:
	vector<unsigned int> NeuralNetworkSetup;
	vector<float> BufferNodeValues;
	vector<unsigned int> TypeOfNode;
	vector<unsigned int> NeuronNodeStartBuffer;
	vector<unsigned int> NeuronNodeLength;
	vector<unsigned int> NeuronInputNodeIDsBuffer;
	vector<unsigned int> NeuronOutputNodeIDsBuffer;
	vector<float> NeuronOutputNodeBuffer;
	vector<float> NeuronWeightsBuffer;
	vector<float> NeuronBiasBuffer;
};


class NeuralNetwork {
public:
	NeuralNetworkModule NeuralNetworkIndividual;
	NeuralNetwork();
	~NeuralNetwork();
	int LoadNeuralNetworkModuleFromFile(
		unsigned int *NetworkModule,		//0=Emotinal training module, 1=Core module, 2=Offline training module
		unsigned int *GenerationNum,
		unsigned int *NumOfIndividualInGeneration
	);
	int NeuralNetwork::SaveNeuralNetworkModuleFromFile(
		unsigned int *NetworkModule,		//0=Emotinal training module, 1=Core module, 2=Offline training module
		unsigned int *GenerationNum,
		unsigned int *NumOfIndividualInGeneration
	);
	vector<float> InputData;
	vector<float> OutputData;
	int LoadNeuralNetworkModule(NeuralNetworkModule NeuralNetworkToLoad);
	int UnloadNeuralNetworkModule(NeuralNetworkModule &UnNeuralNetworkToLoad);
	int LoadInputData(vector<float> InputDataToLoad);
	int LoadScoringVariables(vector<float> ScoringVariablesToLoad);
	int UnloadOutputData(vector<float> &OutputDataTouUnload);
	int UnloadScore(float &ScoreToUnload);
	int RunAICore();
private:
	
	vector<float> ScoringVariables;
	float NetworkScoreBuffer;
};

int SaveVector(
	std::ostringstream *FilePath,
	vector<float> *VectorToSave
);

//SaveNeuralNetworkModule(&BufferNodeValues,&etc...
int SaveNeuralNetworkModule(
	NeuralNetworkModule *NeuralNetworkToSave,
	unsigned int *NetworkModule,		//0=Emotinal training module, 1=Core module, 2=Offline training module
	unsigned int *GenerationNum,
	unsigned int *NumOfIndividualInGeneration
);

int FindFileSize(
	const char* filePath,
	unsigned int &FileSize
);

int LoadVector(
	std::ostringstream *FilePath,		//0=Emotinal training module, 1=Core module, 2=Offline training module
	vector<float> &VectorToLoad
);

int LoadNeuralNetworkModule(
	NeuralNetworkModule &NeuralNetworkToLoad,
	unsigned int *NetworkModule,		//0=Emotinal training module, 1=Core module, 2=Offline training module
	unsigned int *GenerationNum,
	unsigned int *NumOfIndividualInGeneration
);
