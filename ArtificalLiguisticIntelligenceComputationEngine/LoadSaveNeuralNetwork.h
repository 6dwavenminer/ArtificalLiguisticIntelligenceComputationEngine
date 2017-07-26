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

class OLD_NeuralNetworkModule {
public:
	vector<float> BufferNodeValues;
	vector<unsigned int> LSTMInputNodeIDBuffer;
	vector<unsigned int> LSTMOutputNodeIDBuffer;
	vector<float> LSTMWeightsListBuffer;
	vector<float> LSTMPreviousCellStateValueBuffer;		//ct-1
	vector<float> LSTMPreviousOutputValueBuffer;		//ht-1
	vector<float> LSTMForgetGateValueBuffer;			//ft
	vector<float> LSTMInputGateValueBuffer;				//it
	vector<float> LSTMCandidateValueBuffer;				//~ct
	vector<float> LSTMOutputGateValueBuffer;			//ot
	vector<float> LSTMCellStateValueBuffer;				//ct
	vector<float> LSTMOutputValueBuffer;				//ht

	vector<unsigned int> NeuronNodeStartBuffer;
	vector<unsigned int> NeuronNodeEndBuffer;
	vector<unsigned int> NeuronInputNodeIDsBuffer;
	vector<unsigned int> NeuronOutputNodeIDsBuffer;
	vector<float> NeuronOutputNodeBuffer;
	vector<float> NeuronWeightsBuffer;
};


//SaveNeuralNetworkModule(&BufferNodeValues,&etc...
int OLD_SaveNeuralNetworkModule(
	OLD_NeuralNetworkModule *NeuralNetworkToSave,
	unsigned int *NetworkModule,		//0=Emotinal training module, 1=Core module, 2=Offline training module
	unsigned int *GenerationNum,
	unsigned int *NumOfIndividualInGeneration
);


int OLD_LoadNeuralNetworkModule(
	OLD_NeuralNetworkModule &NeuralNetworkToLoad,
	unsigned int *NetworkModule,		//0=Emotinal training module, 1=Core module, 2=Offline training module
	unsigned int *GenerationNum,
	unsigned int *NumOfIndividualInGeneration
);

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

int OLD_RunAICore(
	NeuralNetworkModule& NeuralNetworkIndividual,
	vector<float>& InputData,
	vector<float>& OutputData,
	vector<float>& ScoringVariables,
	float& NetworkScore
);

class NeuralNetwork {
public:
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
	NeuralNetworkModule NeuralNetworkIndividual;
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
