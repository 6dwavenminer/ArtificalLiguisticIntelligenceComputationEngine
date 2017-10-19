
#include "LoadSaveNeuralNetwork.h"

int NeuralNetwork::LoadNeuralNetworkModuleFromFile(
	unsigned int *NetworkModule,		//0=Emotinal training module, 1=Core module, 2=Offline training module
	unsigned int *GenerationNum,
	unsigned int *NumOfIndividualInGeneration
) {
	vector<float> VectorToLoad;
	unsigned int VecPos = 1;
	while (VectorToLoad.size() != 0) {
		VectorToLoad.clear();
	}

	std::ostringstream FilePath;
	unsigned int FileDataSize = 0;
	unsigned int VectorToLoadSize = 0;
	if (*NetworkModule == 0) {
		FilePath << "C:/ALICE/Emotinal training module/Gen-" << *GenerationNum << "_Individual-" << *NumOfIndividualInGeneration << ".DATA";
	}
	else if (*NetworkModule == 1) {
		FilePath << "C:/ALICE/Offline training module/Gen-" << *GenerationNum << "_Individual-" << *NumOfIndividualInGeneration << ".DATA";
	}
	else {
		FilePath << "C:/ALICE/-Core module/Gen-" << *GenerationNum << "_Individual-" << *NumOfIndividualInGeneration << ".DATA";
	}

	LoadVector(
		&FilePath,
		VectorToLoad
	);

	VecPos += VectorToLoad[0];
	(NeuralNetworkIndividual.NeuralNetworkSetup).clear();
	for (unsigned int i = 0; i < VectorToLoad[1]; i++) {
		(NeuralNetworkIndividual.NeuralNetworkSetup).push_back(VectorToLoad[VecPos + i]);
	}

	VecPos += VectorToLoad[1];
	(NeuralNetworkIndividual.BufferNodeValues).clear();
	for (unsigned int i = 0; i < VectorToLoad[2]; i++) {
		(NeuralNetworkIndividual.BufferNodeValues).push_back(VectorToLoad[VecPos + i]);
	}
	VecPos += VectorToLoad[2];

	(NeuralNetworkIndividual.TypeOfNode).clear();
	for (unsigned int i = 0; i < VectorToLoad[3]; i++) {
		(NeuralNetworkIndividual.TypeOfNode).push_back(VectorToLoad[VecPos + i]);
	}
	VecPos += VectorToLoad[3];

	(NeuralNetworkIndividual.NeuronNodeStartBuffer).clear();
	for (unsigned int i = 0; i < VectorToLoad[4]; i++) {
		(NeuralNetworkIndividual.NeuronNodeStartBuffer).push_back(VectorToLoad[VecPos + i]);
	}
	VecPos += VectorToLoad[4];

	(NeuralNetworkIndividual.NeuronNodeLength).clear();
	for (unsigned int i = 0; i < VectorToLoad[5]; i++) {
		(NeuralNetworkIndividual.NeuronNodeLength).push_back(VectorToLoad[VecPos + i]);
	}
	VecPos += VectorToLoad[5];

	(NeuralNetworkIndividual.NeuronInputNodeIDsBuffer).clear();
	for (unsigned int i = 0; i < VectorToLoad[6]; i++) {
		(NeuralNetworkIndividual.NeuronInputNodeIDsBuffer).push_back(VectorToLoad[VecPos + i]);
	}
	VecPos += VectorToLoad[6];

	(NeuralNetworkIndividual.NeuronOutputNodeIDsBuffer).clear();
	for (unsigned int i = 0; i < VectorToLoad[7]; i++) {//15
		(NeuralNetworkIndividual.NeuronOutputNodeIDsBuffer).push_back(VectorToLoad[VecPos + i]);
	}
	VecPos += VectorToLoad[7];

	(NeuralNetworkIndividual.NeuronOutputNodeBuffer).clear();
	for (unsigned int i = 0; i < VectorToLoad[8]; i++) {//16
		(NeuralNetworkIndividual.NeuronOutputNodeBuffer).push_back(VectorToLoad[VecPos + i]);
	}
	VecPos += VectorToLoad[8];

	(NeuralNetworkIndividual.NeuronWeightsBuffer).clear();
	for (unsigned int i = 0; i < VectorToLoad[9]; i++) {//19
		(NeuralNetworkIndividual.NeuronWeightsBuffer).push_back(VectorToLoad[VecPos + i]);
	}
	VecPos += VectorToLoad[9];

	(NeuralNetworkIndividual.NeuronBiasBuffer).clear();
	for (unsigned int i = 0; i < VectorToLoad[10]; i++) {//19
		(NeuralNetworkIndividual.NeuronBiasBuffer).push_back(VectorToLoad[VecPos + i]);
	}
	VecPos += VectorToLoad[10];


	return 0;
}
int NeuralNetwork::SaveNeuralNetworkModuleFromFile(
	unsigned int *NetworkModule,		//0=Emotinal training module, 1=Core module, 2=Offline training module
	unsigned int *GenerationNum,
	unsigned int *NumOfIndividualInGeneration
) {
	vector<float> VectorToSave;
	unsigned int VecPos = 1;
	while (VectorToSave.size() != 0) {
		VectorToSave.clear();
	}
	//First input the header
	VectorToSave.push_back(10);		//NumOfVectorsToSave
	
									//Size of all the vectors
	VectorToSave.push_back((NeuralNetworkIndividual).NeuralNetworkSetup.size());
	VectorToSave.push_back((NeuralNetworkIndividual).BufferNodeValues.size());
	VectorToSave.push_back((NeuralNetworkIndividual).TypeOfNode.size());
	VectorToSave.push_back((NeuralNetworkIndividual).NeuronNodeStartBuffer.size());
	VectorToSave.push_back((NeuralNetworkIndividual).NeuronNodeLength.size());
	VectorToSave.push_back((NeuralNetworkIndividual).NeuronInputNodeIDsBuffer.size());
	VectorToSave.push_back((NeuralNetworkIndividual).NeuronOutputNodeIDsBuffer.size());
	VectorToSave.push_back((NeuralNetworkIndividual).NeuronOutputNodeBuffer.size());
	VectorToSave.push_back((NeuralNetworkIndividual).NeuronWeightsBuffer.size());
	VectorToSave.push_back((NeuralNetworkIndividual).NeuronBiasBuffer.size());

	//Input the data
	for (unsigned int i = 0; i < (NeuralNetworkIndividual).NeuralNetworkSetup.size(); i++) {
		VectorToSave.push_back((float)(NeuralNetworkIndividual).NeuralNetworkSetup[i]);
	}
	for (unsigned int i = 0; i < (NeuralNetworkIndividual).BufferNodeValues.size(); i++) {
		VectorToSave.push_back((float)(NeuralNetworkIndividual).BufferNodeValues[i]);
	}
	for (unsigned int i = 0; i < (NeuralNetworkIndividual).TypeOfNode.size(); i++) {
		VectorToSave.push_back((float)(NeuralNetworkIndividual).TypeOfNode[i]);
	}
	for (unsigned int i = 0; i < (NeuralNetworkIndividual).NeuronNodeStartBuffer.size(); i++) {
		VectorToSave.push_back((float)(NeuralNetworkIndividual).NeuronNodeStartBuffer[i]);
	}
	for (unsigned int i = 0; i < (NeuralNetworkIndividual).NeuronNodeLength.size(); i++) {
		VectorToSave.push_back((float)(NeuralNetworkIndividual).NeuronNodeLength[i]);
	}
	for (unsigned int i = 0; i < (NeuralNetworkIndividual).NeuronInputNodeIDsBuffer.size(); i++) {
		VectorToSave.push_back((float)(NeuralNetworkIndividual).NeuronInputNodeIDsBuffer[i]);
	}
	for (unsigned int i = 0; i < (NeuralNetworkIndividual).NeuronOutputNodeIDsBuffer.size(); i++) {
		VectorToSave.push_back((float)(NeuralNetworkIndividual).NeuronOutputNodeIDsBuffer[i]);
	}
	for (unsigned int i = 0; i < (NeuralNetworkIndividual).NeuronOutputNodeBuffer.size(); i++) {
		VectorToSave.push_back((float)(NeuralNetworkIndividual).NeuronOutputNodeBuffer[i]);
	}
	for (unsigned int i = 0; i < (NeuralNetworkIndividual).NeuronWeightsBuffer.size(); i++) {
		VectorToSave.push_back((float)(NeuralNetworkIndividual).NeuronWeightsBuffer[i]);
	}
	for (unsigned int i = 0; i < (NeuralNetworkIndividual).NeuronBiasBuffer.size(); i++) {
		VectorToSave.push_back((float)(NeuralNetworkIndividual).NeuronBiasBuffer[i]);
	}
									
	/*
	//Size of all the vectors
	VectorToSave.push_back((*NeuralNetworkIndividual).NeuralNetworkSetup.size());
	VectorToSave.push_back((*NeuralNetworkIndividual).BufferNodeValues.size());
	VectorToSave.push_back((*NeuralNetworkIndividual).TypeOfNode.size());
	VectorToSave.push_back((*NeuralNetworkIndividual).NeuronNodeStartBuffer.size());
	VectorToSave.push_back((*NeuralNetworkIndividual).NeuronNodeLength.size());
	VectorToSave.push_back((*NeuralNetworkIndividual).NeuronInputNodeIDsBuffer.size());
	VectorToSave.push_back((*NeuralNetworkIndividual).NeuronOutputNodeIDsBuffer.size());
	VectorToSave.push_back((*NeuralNetworkIndividual).NeuronOutputNodeBuffer.size());
	VectorToSave.push_back((*NeuralNetworkIndividual).NeuronWeightsBuffer.size());
	VectorToSave.push_back((*NeuralNetworkIndividual).NeuronBiasBuffer.size());

	//Input the data
	for (unsigned int i = 0; i < (*NeuralNetworkIndividual).NeuralNetworkSetup.size(); i++) {
		VectorToSave.push_back((float)(*NeuralNetworkIndividual).NeuralNetworkSetup[i]);
	}
	for (unsigned int i = 0; i < (*NeuralNetworkIndividual).BufferNodeValues.size(); i++) {
		VectorToSave.push_back((float)(*NeuralNetworkIndividual).BufferNodeValues[i]);
	}
	for (unsigned int i = 0; i < (*NeuralNetworkIndividual).TypeOfNode.size(); i++) {
		VectorToSave.push_back((float)(*NeuralNetworkIndividual).TypeOfNode[i]);
	}
	for (unsigned int i = 0; i < (*NeuralNetworkIndividual).NeuronNodeStartBuffer.size(); i++) {
		VectorToSave.push_back((float)(*NeuralNetworkIndividual).NeuronNodeStartBuffer[i]);
	}
	for (unsigned int i = 0; i < (*NeuralNetworkIndividual).NeuronNodeLength.size(); i++) {
		VectorToSave.push_back((float)(*NeuralNetworkIndividual).NeuronNodeLength[i]);
	}
	for (unsigned int i = 0; i < (*NeuralNetworkIndividual).NeuronInputNodeIDsBuffer.size(); i++) {
		VectorToSave.push_back((float)(*NeuralNetworkIndividual).NeuronInputNodeIDsBuffer[i]);
	}
	for (unsigned int i = 0; i < (*NeuralNetworkIndividual).NeuronOutputNodeIDsBuffer.size(); i++) {
		VectorToSave.push_back((float)(*NeuralNetworkIndividual).NeuronOutputNodeIDsBuffer[i]);
	}
	for (unsigned int i = 0; i < (*NeuralNetworkIndividual).NeuronOutputNodeBuffer.size(); i++) {
		VectorToSave.push_back((float)(*NeuralNetworkIndividual).NeuronOutputNodeBuffer[i]);
	}
	for (unsigned int i = 0; i < (*NeuralNetworkIndividual).NeuronWeightsBuffer.size(); i++) {
		VectorToSave.push_back((float)(*NeuralNetworkIndividual).NeuronWeightsBuffer[i]);
	}
	for (unsigned int i = 0; i < (*NeuralNetworkIndividual).NeuronBiasBuffer.size(); i++) {
		VectorToSave.push_back((float)(*NeuralNetworkIndividual).NeuronBiasBuffer[i]);
	}*/

	std::ostringstream FilePath;
	if (*NetworkModule == 0) {
		FilePath << "C:/ALICE/Emotinal training module/Gen-" << *GenerationNum << "_Individual-" << *NumOfIndividualInGeneration << ".DATA";
	}
	else if (*NetworkModule == 1) {
		FilePath << "C:/ALICE/Offline training module/Gen-" << *GenerationNum << "_Individual-" << *NumOfIndividualInGeneration << ".DATA";
	}
	else {
		FilePath << "C:/ALICE/Core module/Gen-" << *GenerationNum << "_Individual-" << *NumOfIndividualInGeneration << ".DATA";
	}

	SaveVector(
		&FilePath,
		&VectorToSave
	);
	return 0;
}


NeuralNetwork::NeuralNetwork() {
	//
}
NeuralNetwork::~NeuralNetwork() {
	//
}
int NeuralNetwork::LoadNeuralNetworkModule(NeuralNetworkModule NeuralNetworkToLoad) {
	NeuralNetworkIndividual = NeuralNetworkToLoad;
	return 0;
}
int NeuralNetwork::UnloadNeuralNetworkModule(NeuralNetworkModule &NeuralNetworkToUnLoad) {
	 NeuralNetworkToUnLoad = NeuralNetworkIndividual;
	 return 0;
}
int NeuralNetwork::LoadInputData(vector<float> InputDataToLoad){
	InputData = InputDataToLoad;
	return 0;
}
int NeuralNetwork::LoadScoringVariables(vector<float> ScoringVariablesToLoad) {
	ScoringVariables = ScoringVariablesToLoad;
	return 0;
}
int NeuralNetwork::UnloadOutputData(vector<float> &OutputDataTouUnload) {
	OutputDataTouUnload = OutputData;
	return 0;
}
int NeuralNetwork::UnloadScore(float &ScoreToUnload) {
	ScoreToUnload = NetworkScoreBuffer;
	return 0;
}
int NeuralNetwork::RunAICore() {
	cudaError_t cudaStatus = RunAICoreOnGPU(
		NeuralNetworkIndividual.NeuralNetworkSetup.size(),
		&NeuralNetworkIndividual.NeuralNetworkSetup[0],
		InputData.size(),
		&InputData[0],
		OutputData.size(),
		&OutputData[0],
		NeuralNetworkIndividual.BufferNodeValues.size(),
		&NeuralNetworkIndividual.BufferNodeValues[0],
		NeuralNetworkIndividual.NeuronOutputNodeBuffer.size(),
		NeuralNetworkIndividual.NeuronInputNodeIDsBuffer.size(),
		&NeuralNetworkIndividual.TypeOfNode[0],
		&NeuralNetworkIndividual.NeuronNodeStartBuffer[0],
		&NeuralNetworkIndividual.NeuronNodeLength[0],
		&NeuralNetworkIndividual.NeuronInputNodeIDsBuffer[0],
		&NeuralNetworkIndividual.NeuronOutputNodeIDsBuffer[0],
		&NeuralNetworkIndividual.NeuronOutputNodeBuffer[0],
		NeuralNetworkIndividual.NeuronWeightsBuffer.size(),
		&NeuralNetworkIndividual.NeuronWeightsBuffer[0],
		&NeuralNetworkIndividual.NeuronBiasBuffer[0],
		ScoringVariables.size(),
		&ScoringVariables[0],
		&NetworkScoreBuffer
	);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "RunALICECore failed!");
		return 1;
	}

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}
	return 1;
}


//SaveNeuralNetworkModule(&BufferNodeValues,&etc...
int SaveVector(
	std::ostringstream *FilePath,		//0=Emotinal training module, 1=Core module, 2=Offline training module
	vector<float> *VectorToSave
) {

	ofstream OUTFILE((*FilePath).str(), std::ios_base::binary);
	OUTFILE.write(
		reinterpret_cast<char*>(((*VectorToSave).data())),//reinterpret_cast<char*>(&((*VectorToSave).begin())),
		(*VectorToSave).size() * sizeof(float)
	);

	OUTFILE.close();

	//std::copy((*VectorToSave).begin(), (*VectorToSave).end(), std::ostreambuf_iterator<char>(outfile));
	return 0;
}

int SaveNeuralNetworkModule(
	NeuralNetworkModule *NeuralNetworkToSave,
	unsigned int *NetworkModule,		//0=Emotinal training module, 1=Core module, 2=Offline training module
	unsigned int *GenerationNum,
	unsigned int *NumOfIndividualInGeneration
) {
	vector<float> VectorToSave;
	unsigned int VecPos = 1;
	while (VectorToSave.size() != 0) {
		VectorToSave.clear();
	}	
	//First input the header
	VectorToSave.push_back(10);		//NumOfVectorsToSave
	//Size of all the vectors
	VectorToSave.push_back((*NeuralNetworkToSave).NeuralNetworkSetup.size());
	VectorToSave.push_back((*NeuralNetworkToSave).BufferNodeValues.size());
	VectorToSave.push_back((*NeuralNetworkToSave).TypeOfNode.size());
	VectorToSave.push_back((*NeuralNetworkToSave).NeuronNodeStartBuffer.size());
	VectorToSave.push_back((*NeuralNetworkToSave).NeuronNodeLength.size());
	VectorToSave.push_back((*NeuralNetworkToSave).NeuronInputNodeIDsBuffer.size());
	VectorToSave.push_back((*NeuralNetworkToSave).NeuronOutputNodeIDsBuffer.size());
	VectorToSave.push_back((*NeuralNetworkToSave).NeuronOutputNodeBuffer.size());
	VectorToSave.push_back((*NeuralNetworkToSave).NeuronWeightsBuffer.size());
	VectorToSave.push_back((*NeuralNetworkToSave).NeuronBiasBuffer.size());

	//Input the data
	for (unsigned int i = 0; i < (*NeuralNetworkToSave).NeuralNetworkSetup.size(); i++) {
		VectorToSave.push_back((float)(*NeuralNetworkToSave).NeuralNetworkSetup[i]);
	}
	for (unsigned int i = 0; i < (*NeuralNetworkToSave).BufferNodeValues.size(); i++) {
		VectorToSave.push_back((float)(*NeuralNetworkToSave).BufferNodeValues[i]);
	}
	for (unsigned int i = 0; i < (*NeuralNetworkToSave).TypeOfNode.size(); i++) {
		VectorToSave.push_back((float)(*NeuralNetworkToSave).TypeOfNode[i]);
	}
	for (unsigned int i = 0; i < (*NeuralNetworkToSave).NeuronNodeStartBuffer.size(); i++) {
		VectorToSave.push_back((float)(*NeuralNetworkToSave).NeuronNodeStartBuffer[i]);
	}
	for (unsigned int i = 0; i < (*NeuralNetworkToSave).NeuronNodeLength.size(); i++) {
		VectorToSave.push_back((float)(*NeuralNetworkToSave).NeuronNodeLength[i]);
	}
	for (unsigned int i = 0; i < (*NeuralNetworkToSave).NeuronInputNodeIDsBuffer.size(); i++) {
		VectorToSave.push_back((float)(*NeuralNetworkToSave).NeuronInputNodeIDsBuffer[i]);
	}
	for (unsigned int i = 0; i < (*NeuralNetworkToSave).NeuronOutputNodeIDsBuffer.size(); i++) {
		VectorToSave.push_back((float)(*NeuralNetworkToSave).NeuronOutputNodeIDsBuffer[i]);
	}
	for (unsigned int i = 0; i < (*NeuralNetworkToSave).NeuronOutputNodeBuffer.size(); i++) {
		VectorToSave.push_back((float)(*NeuralNetworkToSave).NeuronOutputNodeBuffer[i]);
	}
	for (unsigned int i = 0; i < (*NeuralNetworkToSave).NeuronWeightsBuffer.size(); i++) {
		VectorToSave.push_back((float)(*NeuralNetworkToSave).NeuronWeightsBuffer[i]);
	}
	for (unsigned int i = 0; i < (*NeuralNetworkToSave).NeuronBiasBuffer.size(); i++) {
		VectorToSave.push_back((float)(*NeuralNetworkToSave).NeuronBiasBuffer[i]);
	}

	std::ostringstream FilePath;
	if (*NetworkModule == 0) {
		FilePath << "C:/ALICE/Emotinal training module/Gen-" << *GenerationNum << "_Individual-" << *NumOfIndividualInGeneration << ".DATA";
	}else if (*NetworkModule == 1) {
		FilePath << "C:/ALICE/Offline training module/Gen-" << *GenerationNum << "_Individual-" << *NumOfIndividualInGeneration << ".DATA";
	}else {
		FilePath << "C:/ALICE/Core module/Gen-" << *GenerationNum << "_Individual-" << *NumOfIndividualInGeneration << ".DATA";
	}

	SaveVector(
		&FilePath,
		&VectorToSave
	);
	return 0;
}

int FindFileSize(
	const char* filePath,
	unsigned int &FileSize
) {
	ifstream INFILE(filePath, std::ios::in | std::ifstream::binary);
	INFILE.ignore(std::numeric_limits<std::streamsize>::max());
	//std::streamsize FileSize = INFILE.gcount();
	FileSize = INFILE.gcount();
	INFILE.clear();   //  Since ignore will have set eof.
	INFILE.seekg(0, std::ios_base::beg);
	return 1;
}


int LoadVector(
	std::ostringstream *FilePath,		//0=Emotinal training module, 1=Core module, 2=Offline training module
	vector<float> &VectorToLoad
) {
	//string FilePath;
	unsigned int FileDataSize = 0;
	unsigned int VectorToLoadSize = 0;

	FindFileSize(
		((*FilePath).str()).c_str(),
		FileDataSize
	);
	VectorToLoadSize = FileDataSize / sizeof(float);
	VectorToLoad.resize(VectorToLoadSize);
	ifstream INFILE((*FilePath).str(), std::ios::in | std::ifstream::binary);

	INFILE.read(
		reinterpret_cast<char*>(&(VectorToLoad[0])),
		FileDataSize
	);
	//std::istreambuf_iterator<char> iter(INFILE);
	//std::copy(iter, std::istreambuf_iterator<char>{}, std::back_inserter(VectorToLoad));
	INFILE.close();
	return 0;
}

int LoadNeuralNetworkModule(
	NeuralNetworkModule &NeuralNetworkToLoad,
	unsigned int *NetworkModule,		//0=Emotinal training module, 1=Core module, 2=Offline training module
	unsigned int *GenerationNum,
	unsigned int *NumOfIndividualInGeneration
) {
	vector<float> VectorToLoad;
	unsigned int VecPos = 1;
	while (VectorToLoad.size() != 0) {
		VectorToLoad.clear();
	}

	std::ostringstream FilePath;
	unsigned int FileDataSize = 0;
	unsigned int VectorToLoadSize = 0;
	if (*NetworkModule == 0) {
		FilePath << "C:/ALICE/Emotinal training module/Gen-" << *GenerationNum << "_Individual-" << *NumOfIndividualInGeneration << ".DATA";
	}else if (*NetworkModule == 1) {
		FilePath << "C:/ALICE/Offline training module/Gen-" << *GenerationNum << "_Individual-" << *NumOfIndividualInGeneration << ".DATA";
	}else {
		FilePath << "C:/ALICE/Core module/Gen-" << *GenerationNum << "_Individual-" << *NumOfIndividualInGeneration << ".DATA";
	}

	LoadVector(
		&FilePath,
		VectorToLoad
	); 

	VecPos += VectorToLoad[0];
	(NeuralNetworkToLoad.NeuralNetworkSetup).clear();
	for (unsigned int i = 0; i < VectorToLoad[1]; i++) {
		(NeuralNetworkToLoad.NeuralNetworkSetup).push_back(VectorToLoad[VecPos + i]);
	}

	VecPos += VectorToLoad[1];
	(NeuralNetworkToLoad.BufferNodeValues).clear();
	for (unsigned int i = 0; i < VectorToLoad[2]; i++) {
		(NeuralNetworkToLoad.BufferNodeValues).push_back(VectorToLoad[VecPos + i]);
	}
	VecPos += VectorToLoad[2];

	(NeuralNetworkToLoad.TypeOfNode).clear();
	for (unsigned int i = 0; i < VectorToLoad[3]; i++) {
		(NeuralNetworkToLoad.TypeOfNode).push_back(VectorToLoad[VecPos + i]);
	}
	VecPos += VectorToLoad[3];

	(NeuralNetworkToLoad.NeuronNodeStartBuffer).clear();
	for (unsigned int i = 0; i < VectorToLoad[4]; i++) {
		(NeuralNetworkToLoad.NeuronNodeStartBuffer).push_back(VectorToLoad[VecPos + i]);
	}
	VecPos += VectorToLoad[4];

	(NeuralNetworkToLoad.NeuronNodeLength).clear();
	for (unsigned int i = 0; i < VectorToLoad[5]; i++) {
		(NeuralNetworkToLoad.NeuronNodeLength).push_back(VectorToLoad[VecPos + i]);
	}
	VecPos += VectorToLoad[5];

	(NeuralNetworkToLoad.NeuronInputNodeIDsBuffer).clear();
	for (unsigned int i = 0; i < VectorToLoad[6]; i++) {
		(NeuralNetworkToLoad.NeuronInputNodeIDsBuffer).push_back(VectorToLoad[VecPos + i]);
	}
	VecPos += VectorToLoad[6];

	(NeuralNetworkToLoad.NeuronOutputNodeIDsBuffer).clear();
	for (unsigned int i = 0; i < VectorToLoad[7]; i++) {//15
		(NeuralNetworkToLoad.NeuronOutputNodeIDsBuffer).push_back(VectorToLoad[VecPos + i]);
	}
	VecPos += VectorToLoad[7];

	(NeuralNetworkToLoad.NeuronOutputNodeBuffer).clear();
	for (unsigned int i = 0; i < VectorToLoad[8]; i++) {//16
		(NeuralNetworkToLoad.NeuronOutputNodeBuffer).push_back(VectorToLoad[VecPos + i]);
	}
	VecPos += VectorToLoad[8];

	(NeuralNetworkToLoad.NeuronWeightsBuffer).clear();
	for (unsigned int i = 0; i < VectorToLoad[9]; i++) {//19
		(NeuralNetworkToLoad.NeuronWeightsBuffer).push_back(VectorToLoad[VecPos + i]);
	}
	VecPos += VectorToLoad[9];

	(NeuralNetworkToLoad.NeuronBiasBuffer).clear();
	for (unsigned int i = 0; i < VectorToLoad[10]; i++) {//19
		(NeuralNetworkToLoad.NeuronBiasBuffer).push_back(VectorToLoad[VecPos + i]);
	}
	VecPos += VectorToLoad[10];


	return 0;
}
