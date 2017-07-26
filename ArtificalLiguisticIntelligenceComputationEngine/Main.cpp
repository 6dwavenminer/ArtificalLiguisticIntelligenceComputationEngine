/*
ArtificalLiguisticIntelligenceComputationEngine
Author: Oliver Thurgood
Version: 0.2.4
Created on: 04/08/2016
Last edited on: 26/07/2017
*/



#include "ALICECore.cuh"
#include "AudioIO.h"
#include "LoadSaveNeuralNetwork.h"
#include <atomic>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <vector>
#include <fstream>
#include <limits>

#include <chrono>
#include <mutex>
#include <future>
#include <utility>

//#define FRAMES_PER_BUFFER (2205)
#define TEST_RUNTIME_IN_SECONDS (120)

using namespace std;
using namespace std::chrono;

void SelectSurvivors(
	vector<float> VectorGenerationScores, 
	vector<unsigned int> &FiveSurviorsList
) {
	float LowestScore = std::numeric_limits<float>::max();
	for (unsigned int i = 0; i < VectorGenerationScores.size(); i++) {
		if (VectorGenerationScores[i] < LowestScore) {
			LowestScore = VectorGenerationScores[i];
		}
	}

	float TotalModifiedScore = 0;
	for (int i = 0; i < VectorGenerationScores.size(); i++) {
		TotalModifiedScore += VectorGenerationScores[i] - LowestScore;
	}

	vector<float> NeuralNetworkChanceOfSurvivalVector;

	srand(time(0));
	for (unsigned int i = 0; i < VectorGenerationScores.size(); i++) {
		float NeuralNetworkChanceOfSurvival = 0;
		float RandomChance = 0;
		const float MinWeight = 0;
		const float MaxWeight = 0.025;//0.085
									  //NeuralNetworkChanceOfSurvival = TotalModifiedScore / (VectorGenerationScores[i] - LowestScore);
		NeuralNetworkChanceOfSurvival = (VectorGenerationScores[i] - LowestScore) / TotalModifiedScore;
		RandomChance = ((MaxWeight - MinWeight) * (float)rand() / (float)RAND_MAX) + MinWeight;
		NeuralNetworkChanceOfSurvival += RandomChance;
		NeuralNetworkChanceOfSurvivalVector.push_back(NeuralNetworkChanceOfSurvival);
	}

	FiveSurviorsList.resize(5);
	for (unsigned int j = 0; j < 5; j++) {
		float HighestSurvivalChance = 0;
		for (unsigned int i = 0; i < NeuralNetworkChanceOfSurvivalVector.size(); i++) {
			if (HighestSurvivalChance < NeuralNetworkChanceOfSurvivalVector[i]) {//VectorGenerationScores
				bool AlreadySurvived = false;
				for (unsigned int k = 0; k < j; k++) {
					if (FiveSurviorsList[k] == i) {
						AlreadySurvived = true;
					}
				}
				if (AlreadySurvived == false) {
					HighestSurvivalChance = NeuralNetworkChanceOfSurvivalVector[i];
					FiveSurviorsList[j] = i;
				}
			}
		}
	}
}


int LoadAudioVector(
	string *FilePathString,
	vector<float> &VectorToLoad
) {
	vector<char> BufferVectorToLoad;
	vector<float> VectorToLoad2;
	float result = 0;
	float result2 = 0;
	std::fstream::pos_type FileSize = 0;
	std::ostringstream FilePath;
	FilePath << "C:/ALICE/" << *FilePathString << ".RAW";
	VectorToLoad.clear();
	ifstream INFILE(FilePath.str(), std::ios::in | std::ifstream::binary);
	std::istreambuf_iterator<char> iter(INFILE);
	std::copy(iter, std::istreambuf_iterator<char>{}, std::back_inserter(BufferVectorToLoad));
	
	for (unsigned int n = 0; n < BufferVectorToLoad.size() / 8; n++) {
		result = 0;
		for (unsigned int i = 0; i < 4; i++) {

			*((char*)(&result) + i) = BufferVectorToLoad[(n * 8) + i + 0]; //for left channel float only need first 4 bytes of every 8
		}
		VectorToLoad.push_back(result);
	}
	result = 0;
	BufferVectorToLoad.clear();
	return 0;
}

int LoadAudio(
	string *FilePathString,
	atomic<unsigned int>& AudioIOToggledUpdateFromMicFlag,
	vector<float>& AudioInputBufferFromMicThread
){
	vector<float> FullAudioVectorBuffer;
	unsigned int FullAudioVectorBufferSize;
	vector<float> AudioVectorSegmentBuffer;
	unsigned int VectorPos = 0;
	double CalcTimeDiff_nseconds = 0.0;
	std::chrono::steady_clock::time_point StartTime;
	std::chrono::steady_clock::time_point EndTime;

	LoadAudioVector(
		*(&FilePathString),
		FullAudioVectorBuffer
	);
	FullAudioVectorBufferSize = FullAudioVectorBuffer.size();
	if (FullAudioVectorBufferSize != 0) {
		while (VectorPos < FullAudioVectorBufferSize) {
			//Start of section
			StartTime = std::chrono::high_resolution_clock::now();
			while (AudioVectorSegmentBuffer.size() != 0) {
				AudioVectorSegmentBuffer.clear();
			}
			if ((VectorPos + FRAMES_PER_BUFFER) < FullAudioVectorBufferSize) {
				AudioVectorSegmentBuffer.resize(FRAMES_PER_BUFFER);
				std::copy(FullAudioVectorBuffer.begin() + VectorPos, FullAudioVectorBuffer.begin() + VectorPos + FRAMES_PER_BUFFER, AudioVectorSegmentBuffer.begin());
			}else {
				AudioVectorSegmentBuffer.resize(FullAudioVectorBufferSize - VectorPos);
				std::copy(FullAudioVectorBuffer.begin() + VectorPos, FullAudioVectorBuffer.end(), AudioVectorSegmentBuffer.begin());
				while (AudioVectorSegmentBuffer.size() < FRAMES_PER_BUFFER) {
					AudioVectorSegmentBuffer.push_back(0);
				}
			}
			VectorPos += FRAMES_PER_BUFFER + 1;
			while (AudioInputBufferFromMicThread.size() != 0) {
				AudioInputBufferFromMicThread.clear();
			}
			AudioInputBufferFromMicThread.resize(AudioVectorSegmentBuffer.size());
			std::copy(AudioVectorSegmentBuffer.begin(), AudioVectorSegmentBuffer.end(),AudioInputBufferFromMicThread.begin());
			AudioIOToggledUpdateFromMicFlag ^= 1;
			do {
				EndTime = std::chrono::high_resolution_clock::now();
				CalcTimeDiff_nseconds = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(EndTime - StartTime).count() / 1e9;
			} while(CalcTimeDiff_nseconds <= ((double)FRAMES_PER_BUFFER / (double)SAMPLE_RATE));
			//End of section
		}
	}
	return 0;
}

int SilentAudio(
	double SecondsOfSilence,
	atomic<unsigned int>& AudioIOToggledUpdateFromMicFlag,
	vector<float>& AudioInputBufferFromMicThread
) {
	unsigned int FullAudioVectorBufferSize;
	vector<float> AudioVectorSegmentBuffer;
	unsigned int VectorPos = 0;
	std::chrono::steady_clock::time_point OverallStartTime;
	std::chrono::steady_clock::time_point OverallEndTime;
	std::chrono::steady_clock::time_point StartTime;
	std::chrono::steady_clock::time_point EndTime;
	double OverallCalcTimeDiff_nseconds = 0.0;
	double CalcTimeDiff_nseconds = 0.0;

	while (AudioVectorSegmentBuffer.size() != 0) {
		AudioVectorSegmentBuffer.clear();
	}
	while (AudioVectorSegmentBuffer.size() < FRAMES_PER_BUFFER) {
		AudioVectorSegmentBuffer.push_back(0);
	}

	OverallStartTime = std::chrono::high_resolution_clock::now();
	StartTime = std::chrono::high_resolution_clock::now();
	while (OverallCalcTimeDiff_nseconds < SecondsOfSilence) {
		//Start of section
		while (AudioInputBufferFromMicThread.size() != 0) {
			AudioInputBufferFromMicThread.clear();
		}
		AudioInputBufferFromMicThread.resize(AudioVectorSegmentBuffer.size());
		std::copy(AudioVectorSegmentBuffer.begin(), AudioVectorSegmentBuffer.end(), AudioInputBufferFromMicThread.begin());
		AudioIOToggledUpdateFromMicFlag ^= 1;
		do{
			EndTime = std::chrono::high_resolution_clock::now();
			CalcTimeDiff_nseconds = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(EndTime - StartTime).count() / 1e9;
		}while (CalcTimeDiff_nseconds < (FRAMES_PER_BUFFER / SAMPLE_RATE));
		//End of section
		StartTime = std::chrono::high_resolution_clock::now();
		OverallEndTime = std::chrono::high_resolution_clock::now();
		OverallCalcTimeDiff_nseconds = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(OverallEndTime - OverallStartTime).count() / 1e9;
	}
	return 0;
}

int RandomAudio(
	double SecondsOfSilence,
	atomic<unsigned int>& AudioIOToggledUpdateFromMicFlag,
	vector<float>& AudioInputBufferFromMicThread
) {
	const float MinValue = -1.0;
	const float MaxValue = 1.0;
	unsigned int FullAudioVectorBufferSize;
	vector<float> AudioVectorSegmentBuffer;
	unsigned int VectorPos = 0;
	std::chrono::steady_clock::time_point OverallStartTime;
	std::chrono::steady_clock::time_point OverallEndTime;
	std::chrono::steady_clock::time_point StartTime;
	std::chrono::steady_clock::time_point EndTime;
	double OverallCalcTimeDiff_nseconds = 0.0;
	double CalcTimeDiff_nseconds = 0.0;

	srand(time(0));
	OverallStartTime = std::chrono::high_resolution_clock::now();
	StartTime = std::chrono::high_resolution_clock::now();
	while (OverallCalcTimeDiff_nseconds < SecondsOfSilence) {
		//Start of section
		while (AudioVectorSegmentBuffer.size() != 0) {
			AudioVectorSegmentBuffer.clear();
		}
		while (AudioVectorSegmentBuffer.size() < FRAMES_PER_BUFFER) {
			AudioVectorSegmentBuffer.push_back(((MaxValue - MinValue) * (float)rand() / (float)RAND_MAX) + MinValue);
		}
		while (AudioInputBufferFromMicThread.size() != 0) {
			AudioInputBufferFromMicThread.clear();
		}
		AudioInputBufferFromMicThread.resize(AudioVectorSegmentBuffer.size());
		std::copy(AudioVectorSegmentBuffer.begin(), AudioVectorSegmentBuffer.end(), AudioInputBufferFromMicThread.begin());
		AudioIOToggledUpdateFromMicFlag ^= 1;
		EndTime = std::chrono::high_resolution_clock::now();
		CalcTimeDiff_nseconds = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(EndTime - StartTime).count() / 1e9;
		double TimeToWait = (FRAMES_PER_BUFFER / SAMPLE_RATE) - CalcTimeDiff_nseconds;
		StartTime = std::chrono::high_resolution_clock::now();
		EndTime = std::chrono::high_resolution_clock::now();
		CalcTimeDiff_nseconds = 0.0;
		while (CalcTimeDiff_nseconds < TimeToWait) {
			double CalcTimeDiff_nseconds = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(EndTime - StartTime).count() / 1e9;
		}
		//End of section
		StartTime = std::chrono::high_resolution_clock::now();
		OverallEndTime = std::chrono::high_resolution_clock::now();
		OverallCalcTimeDiff_nseconds = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(OverallEndTime - OverallStartTime).count() / 1e9;
	}
	return 0;
}

/*
Selects and loads audio with associated scoring variables
*/
int SupervisorThread(
	atomic<unsigned int> *RunProgram,
	atomic<unsigned int> *ProgramStartup,
	unsigned int TrainingMode,
	atomic<unsigned int>& AudioIOToggledUpdateFromMicFlag,
	vector<float>& AudioInputBufferFromMicThread,
	vector<float>& ScoringVariables
) {
	const int MinValue = 0;
	const int MaxValue = 14;//12 + Admin leaves, admin becomes busy(15in total)
	int RandomValue;
	int Group = 0;
	int SubGroup = 0;
	int MemberOfGroup = 0;

	bool OtherPersonPresent = false;
	bool AdminPresent = true;
	bool AdminLeaving = false;
	bool AdminBusy = false;
	bool IsAdmin = true;

	/*
	ScoringVariables
	[0]TrainingMode
	[1]NumOfNodesToScore
	[2 + (n * 2)]NodeIDsToScore
	[3 + (n * 2)]ScoringVariables
	*/

	while (*ProgramStartup == 1) {
		//Wait for startup to finish
	}
	while ((*RunProgram == 1) && (TrainingMode == 0)) {
		string FilePathString;
		std::ostringstream FilePathOStringStream;
		FilePathOStringStream.clear();
		if (TrainingMode == 0) {
			ScoringVariables.clear();
			ScoringVariables.resize(11);
			RandomValue = ((99 - MinValue) * rand() / RAND_MAX) + MinValue;
			if (((RandomValue < 85) && (AdminPresent == true))||((RandomValue >= 85) && (AdminPresent == false))) {
				IsAdmin = true;
			}else {
				IsAdmin = false;
			}
			RandomValue = ((99 - MinValue) * rand() / RAND_MAX) + MinValue;
			if (((RandomValue < 80) && (OtherPersonPresent == true)) || ((RandomValue >= 80) && (OtherPersonPresent == false))) {
				OtherPersonPresent = true;
			}else {
				OtherPersonPresent = false;
			}
			/*
			0=Admiration(1-274)
			1=Amazement(1-71)
			2=Ecstasy(1-261)
			3=Grief(1-185)
			4=Loathing(1-290)
			5=Neutral(1-593)
			6=Rage(1-136)
			7=Terror(1-6)
			8=Vigilance(1-25)
			9=High admiration and esctasy(1-103)
			10=High grief and admiration(1-290)
			11=Low grief, loathing, rage and amazement(1-82)
			12=Low Vigilance, esctasy, admiration and amazement(1-86)
			13=Admin leaves/comes back
				leaves
				0=Ecstasy(1-21)
				1=Grief(1-18)
				2=Loathing(1-9)
				3=Neutral(1-37)
				4=Vigilance(1-10)
				comes back
				0=Admiration(1-31)
				1=Ecstasy(1-23)
				2=Grief(1-20)
				3=Loathing(1-22)
				4=Neutral(1-31)
				5=Rage(1-6)
			14=Admin busy/not busy
				Busy
				0=Ecstasy(1-19)
				1=Grief(1-18)
				2=Loathing(1-29)
				3=Neutral(1-32)
				4-Rage(1-25)
				5-Terror(1-29)
				6=Vigilance(1-22)
				Not busy
				0=Admiration(1-20)
				1=Ecstasy(1-27)
				2=Grief(1-7)
				3=Loathing(1-23)
				4=Neutral(1-35)
				5=Rage(1-14)
			*/
			/*
			ScoringVariables
			0=Ecstasy
			1=Admiration
			2=Terror
			3=Amazement
			4=Grief
			5=Loathing
			6=Rage
			7=Vigilance
			8=ID
			9=Busy
			10=Present
			*/
			if (AdminPresent == true) {
				if (IsAdmin == true) {
					FilePathOStringStream << "Emotinal training module/Training data/From Admin/";
					RandomValue = ((14 - MinValue) * rand() / RAND_MAX) + MinValue;
					switch (RandomValue) {
					case 0:
						ScoringVariables[0] = 0.1;
						ScoringVariables[1] = 0.5;
						ScoringVariables[2] = -0.7;
						ScoringVariables[3] = 0.1;
						ScoringVariables[4] = -0.7;
						ScoringVariables[5] = -0.7;
						ScoringVariables[6] = -0.6;
						ScoringVariables[7] = 0;
						RandomValue = ((274 - 1) * rand() / RAND_MAX) + 1;
						FilePathOStringStream << "Admiration/" << RandomValue;
						break;
					case 1:
						ScoringVariables[0] = 0.1;
						ScoringVariables[1] = 0.35;
						ScoringVariables[2] = -0.1;
						ScoringVariables[3] = 0.6;
						ScoringVariables[4] = -0.7;
						ScoringVariables[5] = -0.7;
						ScoringVariables[6] = -0.5;
						ScoringVariables[7] = 0.5;
						RandomValue = ((71 - 1) * rand() / RAND_MAX) + 1;
						FilePathOStringStream << "Amazement/" << RandomValue;
						break;
					case 2:
						ScoringVariables[0] = 0.5;
						ScoringVariables[1] = 0.1;
						ScoringVariables[2] = -0.7;
						ScoringVariables[3] = 0;
						ScoringVariables[4] = -0.6;
						ScoringVariables[5] = -0.7;
						ScoringVariables[6] = -0.65;
						ScoringVariables[7] = -0.1;
						RandomValue = ((261 - 1) * rand() / RAND_MAX) + 1;
						FilePathOStringStream << "Ecstasy/" << RandomValue;
						break;
					case 3:
						ScoringVariables[0] = -0.1;
						ScoringVariables[1] = 0.2;
						ScoringVariables[2] = -0.1;
						ScoringVariables[3] = -0.1;
						ScoringVariables[4] = 0.5;
						ScoringVariables[5] = -0.1;
						ScoringVariables[6] = 0;
						ScoringVariables[7] = -0.5;
						RandomValue = ((185 - 1) * rand() / RAND_MAX) + 1;
						FilePathOStringStream << "Grief/" << RandomValue;
						break;
					case 4:
						ScoringVariables[0] = -0.2;
						ScoringVariables[1] = -0.5;
						ScoringVariables[2] = 0.0;
						ScoringVariables[3] = -0.6;
						ScoringVariables[4] = -0.2;
						ScoringVariables[5] = 0.5;
						ScoringVariables[6] = 0.1;
						ScoringVariables[7] = 0.2;
						RandomValue = ((290 - 1) * rand() / RAND_MAX) + 1;
						FilePathOStringStream << "Loathing/" << RandomValue;
						break;
					case 5:
						ScoringVariables[0] = 0.05;
						ScoringVariables[1] = 0.0;
						ScoringVariables[2] = -0.5;
						ScoringVariables[3] = -0.5;
						ScoringVariables[4] = -0.5;
						ScoringVariables[5] = -0.3;
						ScoringVariables[6] = -0.4;
						ScoringVariables[7] = 0.1;
						RandomValue = ((593 - 1) * rand() / RAND_MAX) + 1;
						FilePathOStringStream << "Neutral/" << RandomValue;
						break;
					case 6:
						ScoringVariables[0] = -0.5;
						ScoringVariables[1] = -0.5;
						ScoringVariables[2] = -0.1;
						ScoringVariables[3] = -0.2;
						ScoringVariables[4] = 0;
						ScoringVariables[5] = 0.2;
						ScoringVariables[6] = 0.5;
						ScoringVariables[7] = -0.5;
						RandomValue = ((136 - 1) * rand() / RAND_MAX) + 1;
						FilePathOStringStream << "Rage/" << RandomValue;
						break;
					case 7:
						ScoringVariables[0] = -0.6;
						ScoringVariables[1] = -0.6;
						ScoringVariables[2] = 0.5;
						ScoringVariables[3] = 0.2;
						ScoringVariables[4] = -0.2;
						ScoringVariables[5] = 0.2;
						ScoringVariables[6] = -0.3;
						ScoringVariables[7] = 0.4;
						RandomValue = ((6 - 1) * rand() / RAND_MAX) + 1;
						FilePathOStringStream << "Terror/" << RandomValue;
						break;
					case 8:
						ScoringVariables[0] = -0.5;
						ScoringVariables[1] = -0.3;
						ScoringVariables[2] = 0.2;
						ScoringVariables[3] = 0.1;
						ScoringVariables[4] = -0.5;
						ScoringVariables[5] = 0.0;
						ScoringVariables[6] = -0.3;
						ScoringVariables[7] = 0.5;
						RandomValue = ((25 - 1) * rand() / RAND_MAX) + 1;
						FilePathOStringStream << "Vigilance/" << RandomValue;
						break;
					case 9:
						ScoringVariables[0] = 0.9;
						ScoringVariables[1] = 0.9;
						ScoringVariables[2] = -0.9;
						ScoringVariables[3] = 0.1;
						ScoringVariables[4] = -0.8;
						ScoringVariables[5] = -0.9;
						ScoringVariables[6] = -0.8;
						ScoringVariables[7] = 0.0;
						RandomValue = ((103 - 1) * rand() / RAND_MAX) + 1;
						FilePathOStringStream << "High admiration & esctasy/" << RandomValue;
						break;
					case 10:
						ScoringVariables[0] = 0;
						ScoringVariables[1] = 0.8;
						ScoringVariables[2] = -0.6;
						ScoringVariables[3] = -0.2;
						ScoringVariables[4] = 0.9;
						ScoringVariables[5] = -0.8;
						ScoringVariables[6] = -0.7;
						ScoringVariables[7] = -0.7;
						RandomValue = ((290 - 1) * rand() / RAND_MAX) + 1;
						FilePathOStringStream << "High grief & admiration/" << RandomValue;
						break;
					case 11:
						ScoringVariables[0] = -0.05;
						ScoringVariables[1] = -0.1;
						ScoringVariables[2] = -0.5;
						ScoringVariables[3] = 0;
						ScoringVariables[4] = -0.3;
						ScoringVariables[5] = 0.1;
						ScoringVariables[6] = -0.2;
						ScoringVariables[7] = 0.12;
						RandomValue = ((82 - 1) * rand() / RAND_MAX) + 1;
						FilePathOStringStream << "Low grief & loathing & rage  & amazement/" << RandomValue;
						break;
					case 12:
						ScoringVariables[0] = 0.15;
						ScoringVariables[1] = 0.1;
						ScoringVariables[2] = -0.5;
						ScoringVariables[3] = 0;
						ScoringVariables[4] = -0.55;
						ScoringVariables[5] = -0.4;
						ScoringVariables[6] = -0.5;
						ScoringVariables[7] = 0.12;
						RandomValue = ((86 - 1) * rand() / RAND_MAX) + 1;
						FilePathOStringStream << "Low Vigilance & ecstasy & admiration & amazement/" << RandomValue;
						break;
					case 13:
						AdminLeaving = true;
						FilePathOStringStream << "Gone away/";
						RandomValue = ((4 - MinValue) * rand() / RAND_MAX) + MinValue;
						switch (RandomValue) {
						case 0:
							ScoringVariables[0] = 0.5;
							ScoringVariables[1] = 0.1;
							ScoringVariables[2] = -0.7;
							ScoringVariables[3] = 0;
							ScoringVariables[4] = -0.6;
							ScoringVariables[5] = -0.7;
							ScoringVariables[6] = -0.65;
							ScoringVariables[7] = -0.1;
							RandomValue = ((21 - 1) * rand() / RAND_MAX) + 1;
							FilePathOStringStream << "Ecstasy/" << RandomValue;
							break;
						case 1:
							ScoringVariables[0] = -0.1;
							ScoringVariables[1] = 0.2;
							ScoringVariables[2] = -0.1;
							ScoringVariables[3] = -0.1;
							ScoringVariables[4] = 0.5;
							ScoringVariables[5] = -0.1;
							ScoringVariables[6] = 0;
							ScoringVariables[7] = -0.5;
							RandomValue = ((18 - 1) * rand() / RAND_MAX) + 1;
							FilePathOStringStream << "Grief/" << RandomValue;
							break;
						case 2:
							ScoringVariables[0] = -0.2;
							ScoringVariables[1] = -0.5;
							ScoringVariables[2] = 0;
							ScoringVariables[3] = -0.6;
							ScoringVariables[4] = -0.2;
							ScoringVariables[5] = 0.5;
							ScoringVariables[6] = 0.1;
							ScoringVariables[7] = 0.2;
							RandomValue = ((9 - 1) * rand() / RAND_MAX) + 1;
							FilePathOStringStream << "Loathing/" << RandomValue;
							break;
						case 3:
							ScoringVariables[0] = 0.05;
							ScoringVariables[1] = 0;
							ScoringVariables[2] = -0.5;
							ScoringVariables[3] = -0.5;
							ScoringVariables[4] = -0.5;
							ScoringVariables[5] = -0.3;
							ScoringVariables[6] = -0.4;
							ScoringVariables[7] = 0.1;
							RandomValue = ((37 - 1) * rand() / RAND_MAX) + 1;
							FilePathOStringStream << "Neutral/" << RandomValue;
							break;
						case 4:
							ScoringVariables[0] = -0.5;
							ScoringVariables[1] = -0.3;
							ScoringVariables[2] = 0.2;
							ScoringVariables[3] = 0.1;
							ScoringVariables[4] = -0.5;
							ScoringVariables[5] = 0;
							ScoringVariables[6] = -0.3;
							ScoringVariables[7] = 0.5;
							RandomValue = ((10 - 1) * rand() / RAND_MAX) + 1;
							FilePathOStringStream << "Vigilance/" << RandomValue;
							break;
						}
						break;
					case 14:
						if (AdminBusy == false) {
							AdminBusy = true;
							FilePathOStringStream << "Start busy/";
							RandomValue = ((6 - MinValue) * rand() / RAND_MAX) + MinValue;
							switch (RandomValue) {
							case 0:
								ScoringVariables[0] = 0.5;
								ScoringVariables[1] = 0.1;
								ScoringVariables[2] = -0.7;
								ScoringVariables[3] = 0;
								ScoringVariables[4] = -0.6;
								ScoringVariables[5] = -0.7;
								ScoringVariables[6] = -0.65;
								ScoringVariables[7] = -0.1;
								RandomValue = ((19 - 1) * rand() / RAND_MAX) + 1;
								FilePathOStringStream << "Ecstasy/" << RandomValue;
								break;
							case 1:
								ScoringVariables[0] = -0.1;
								ScoringVariables[1] = 0.2;
								ScoringVariables[2] = -0.1;
								ScoringVariables[3] = -0.1;
								ScoringVariables[4] = 0.5;
								ScoringVariables[5] = -0.1;
								ScoringVariables[6] = 0;
								ScoringVariables[7] = -0.5;
								RandomValue = ((18 - 1) * rand() / RAND_MAX) + 1;
								FilePathOStringStream << "Grief/" << RandomValue;
								break;
							case 2:
								ScoringVariables[0] = -0.2;
								ScoringVariables[1] = -0.5;
								ScoringVariables[2] = 0;
								ScoringVariables[3] = -0.6;
								ScoringVariables[4] = -0.2;
								ScoringVariables[5] = 0.5;
								ScoringVariables[6] = 0.1;
								ScoringVariables[7] = 0.2;
								RandomValue = ((29 - 1) * rand() / RAND_MAX) + 1;
								FilePathOStringStream << "Loathing/" << RandomValue;
								break;
							case 3:
								ScoringVariables[0] = 0.05;
								ScoringVariables[1] = 0;
								ScoringVariables[2] = -0.5;
								ScoringVariables[3] = -0.5;
								ScoringVariables[4] = -0.5;
								ScoringVariables[5] = -0.3;
								ScoringVariables[6] = -0.4;
								ScoringVariables[7] = 0.1;
								RandomValue = ((32 - 1) * rand() / RAND_MAX) + 1;
								FilePathOStringStream << "Neutral/" << RandomValue;
								break;
							case 4:
								ScoringVariables[0] = -0.5;
								ScoringVariables[1] = -0.5;
								ScoringVariables[2] = -0.1;
								ScoringVariables[3] = -0.2;
								ScoringVariables[4] = 0;
								ScoringVariables[5] = 0.2;
								ScoringVariables[6] = 0.5;
								ScoringVariables[7] = -0.5;
								RandomValue = ((25 - 1) * rand() / RAND_MAX) + 1;
								FilePathOStringStream << "Rage/" << RandomValue;
								break;
							case 5:
								ScoringVariables[0] = -0.6;
								ScoringVariables[1] = -0.6;
								ScoringVariables[2] = 0.5;
								ScoringVariables[3] = 0.2;
								ScoringVariables[4] = -0.2;
								ScoringVariables[5] = 0.2;
								ScoringVariables[6] = -0.3;
								ScoringVariables[7] = 0.4;
								RandomValue = ((29 - 1) * rand() / RAND_MAX) + 1;
								FilePathOStringStream << "Terror/" << RandomValue;
								break;
							case 6:
								ScoringVariables[0] = -0.5;
								ScoringVariables[1] = -0.3;
								ScoringVariables[2] = 0.2;
								ScoringVariables[3] = 0.1;
								ScoringVariables[4] = -0.5;
								ScoringVariables[5] = 0;
								ScoringVariables[6] = -0.3;
								ScoringVariables[7] = 0.5;
								RandomValue = ((22 - 1) * rand() / RAND_MAX) + 1;
								FilePathOStringStream << "Vigilance/" << RandomValue;
								break;
							}
						}
						else {
							AdminBusy = false;
							FilePathOStringStream << "End busy/";
							RandomValue = ((5 - MinValue) * rand() / RAND_MAX) + MinValue;
							switch (RandomValue) {
							case 0:
								ScoringVariables[0] = 0.1;
								ScoringVariables[1] = 0.5;
								ScoringVariables[2] = -0.7;
								ScoringVariables[3] = 0.1;
								ScoringVariables[4] = -0.7;
								ScoringVariables[5] = -0.7;
								ScoringVariables[6] = -0.6;
								ScoringVariables[7] = 0;
								RandomValue = ((20 - 1) * rand() / RAND_MAX) + 1;
								FilePathOStringStream << "Admiration/" << RandomValue;
								break;
							case 1:
								ScoringVariables[0] = 0.5;
								ScoringVariables[1] = 0.1;
								ScoringVariables[2] = -0.7;
								ScoringVariables[3] = 0;
								ScoringVariables[4] = -0.6;
								ScoringVariables[5] = -0.7;
								ScoringVariables[6] = -0.65;
								ScoringVariables[7] = -0.1;
								RandomValue = ((27 - 1) * rand() / RAND_MAX) + 1;
								FilePathOStringStream << "Ecstasy/" << RandomValue;
								break;
							case 2:
								ScoringVariables[0] = -0.1;
								ScoringVariables[1] = 0.2;
								ScoringVariables[2] = -0.1;
								ScoringVariables[3] = -0.1;
								ScoringVariables[4] = 0.5;
								ScoringVariables[5] = -0.1;
								ScoringVariables[6] = 0;
								ScoringVariables[7] = -0.5;
								RandomValue = ((7 - 1) * rand() / RAND_MAX) + 1;
								FilePathOStringStream << "Grief/" << RandomValue;
								break;
							case 3:
								ScoringVariables[0] = -0.2;
								ScoringVariables[1] = -0.5;
								ScoringVariables[2] = 0;
								ScoringVariables[3] = -0.6;
								ScoringVariables[4] = -0.2;
								ScoringVariables[5] = 0.5;
								ScoringVariables[6] = 0.1;
								ScoringVariables[7] = 0.2;
								RandomValue = ((23 - 1) * rand() / RAND_MAX) + 1;
								FilePathOStringStream << "Loathing/" << RandomValue;
								break;
							case 4:
								ScoringVariables[0] = 0.05;
								ScoringVariables[1] = 0;
								ScoringVariables[2] = -0.5;
								ScoringVariables[3] = -0.5;
								ScoringVariables[4] = -0.5;
								ScoringVariables[5] = -0.3;
								ScoringVariables[6] = -0.4;
								ScoringVariables[7] = 0.1;
								RandomValue = ((35 - 1) * rand() / RAND_MAX) + 1;
								FilePathOStringStream << "Neutral/" << RandomValue;
								break;
							case 5:
								ScoringVariables[0] = -0.5;
								ScoringVariables[1] = -0.5;
								ScoringVariables[2] = -0.1;
								ScoringVariables[3] = -0.2;
								ScoringVariables[4] = 0;
								ScoringVariables[5] = 0.2;
								ScoringVariables[6] = 0.5;
								ScoringVariables[7] = -0.5;
								RandomValue = ((14 - 1) * rand() / RAND_MAX) + 1;
								FilePathOStringStream << "Rage/" << RandomValue;
								break;
							}
						}
						break;
					}
				}else if (OtherPersonPresent == true) {
					//not admin
					FilePathOStringStream << "Emotinal training module/Training data/From Other/";
					RandomValue = ((8 - MinValue) * rand() / RAND_MAX) + MinValue;
					switch (RandomValue) {
					case 0:
						ScoringVariables[0] = 0.1;
						ScoringVariables[1] = 0.5;
						ScoringVariables[2] = -0.7;
						ScoringVariables[3] = 0.1;
						ScoringVariables[4] = -0.7;
						ScoringVariables[5] = -0.7;
						ScoringVariables[6] = -0.6;
						ScoringVariables[7] = 0.0;
						RandomValue = ((58 - 1) * rand() / RAND_MAX) + 1;
						FilePathOStringStream << "Admiration/" << RandomValue;
						break;
					case 1:
						ScoringVariables[0] = 0.1;
						ScoringVariables[1] = 0.35;
						ScoringVariables[2] = -0.1;
						ScoringVariables[3] = 0.6;
						ScoringVariables[4] = -0.7;
						ScoringVariables[5] = -0.7;
						ScoringVariables[6] = -0.5;
						ScoringVariables[7] = 0.5;
						RandomValue = ((43 - 1) * rand() / RAND_MAX) + 1;
						FilePathOStringStream << "Amazement/" << RandomValue;
						break;
					case 2:
						ScoringVariables[0] = 0.5;
						ScoringVariables[1] = 0.1;
						ScoringVariables[2] = -0.7;
						ScoringVariables[3] = 0;
						ScoringVariables[4] = -0.6;
						ScoringVariables[5] = -0.7;
						ScoringVariables[6] = -0.65;
						ScoringVariables[7] = -0.1;
						RandomValue = ((108 - 1) * rand() / RAND_MAX) + 1;
						FilePathOStringStream << "Ecstasy/" << RandomValue;
						break;
					case 3:
						ScoringVariables[0] = -0.1;
						ScoringVariables[1] = 0.2;
						ScoringVariables[2] = -0.1;
						ScoringVariables[3] = -0.1;
						ScoringVariables[4] = 0.5;
						ScoringVariables[5] = -0.1;
						ScoringVariables[6] = 0.0;
						ScoringVariables[7] = -0.5;
						RandomValue = ((77 - 1) * rand() / RAND_MAX) + 1;
						FilePathOStringStream << "Grief/" << RandomValue;
						break;
					case 4:
						ScoringVariables[0] = -0.2;
						ScoringVariables[1] = -0.5;
						ScoringVariables[2] = 0;
						ScoringVariables[3] = -0.6;
						ScoringVariables[4] = -0.2;
						ScoringVariables[5] = 0.5;
						ScoringVariables[6] = 0.1;
						ScoringVariables[7] = 0.2;
						RandomValue = ((63 - 1) * rand() / RAND_MAX) + 1;
						FilePathOStringStream << "Loathing/" << RandomValue;
						break;
					case 5:
						ScoringVariables[0] = 0.05;
						ScoringVariables[1] = 0.0;
						ScoringVariables[2] = -0.5;
						ScoringVariables[3] = -0.5;
						ScoringVariables[4] = -0.5;
						ScoringVariables[5] = -0.3;
						ScoringVariables[6] = -0.4;
						ScoringVariables[7] = 0.1;
						RandomValue = ((102 - 1) * rand() / RAND_MAX) + 1;
						FilePathOStringStream << "Neutral/" << RandomValue;
						break;
					case 6:
						ScoringVariables[0] = -0.5;
						ScoringVariables[1] = -0.5;
						ScoringVariables[2] = -0.1;
						ScoringVariables[3] = -0.2;
						ScoringVariables[4] = 0.0;
						ScoringVariables[5] = 0.2;
						ScoringVariables[6] = 0.5;
						ScoringVariables[7] = -0.5;
						RandomValue = ((83 - 1) * rand() / RAND_MAX) + 1;
						FilePathOStringStream << "Rage/" << RandomValue;
						break;
					case 7:
						ScoringVariables[0] = -0.6;
						ScoringVariables[1] = -0.6;
						ScoringVariables[2] = 0.5;
						ScoringVariables[3] = 0.2;
						ScoringVariables[4] = -0.2;
						ScoringVariables[5] = 0.2;
						ScoringVariables[6] = -0.3;
						ScoringVariables[7] = 0.4;
						RandomValue = ((50 - 1) * rand() / RAND_MAX) + 1;
						FilePathOStringStream << "Terror/" << RandomValue;
						break;
					case 8:
						ScoringVariables[0] = -0.5;
						ScoringVariables[1] = -0.3;
						ScoringVariables[2] = 0.2;
						ScoringVariables[3] = 0.1;
						ScoringVariables[4] = -0.5;
						ScoringVariables[5] = 0.0;
						ScoringVariables[6] = -0.3;
						ScoringVariables[7] = 0.5;
						RandomValue = ((66 - 1) * rand() / RAND_MAX) + 1;
						FilePathOStringStream << "Vigilance/" << RandomValue;
						break;
					}
				}
			}else {
				if (IsAdmin == true) {
					AdminPresent = true;
					FilePathOStringStream << "Emotinal training module/Training data/From Admin/Came back/";
					RandomValue = ((5 - MinValue) * rand() / RAND_MAX) + MinValue;
					switch (RandomValue) {
					case 0:
						ScoringVariables[0] = 0.1;
						ScoringVariables[1] = 0.5;
						ScoringVariables[2] = -0.7;
						ScoringVariables[3] = 0.1;
						ScoringVariables[4] = -0.7;
						ScoringVariables[5] = -0.7;
						ScoringVariables[6] = -0.6;
						ScoringVariables[7] = 0;
						RandomValue = ((31 - 1) * rand() / RAND_MAX) + 1;
						FilePathOStringStream << "Admiration/" << RandomValue;
						break;
					case 1:
						ScoringVariables[0] = 0.5;
						ScoringVariables[1] = 0.1;
						ScoringVariables[2] = -0.7;
						ScoringVariables[3] = 0;
						ScoringVariables[4] = -0.6;
						ScoringVariables[5] = -0.7;
						ScoringVariables[6] = -0.65;
						ScoringVariables[7] = -0.1;
						RandomValue = ((23 - 1) * rand() / RAND_MAX) + 1;
						FilePathOStringStream << "Ecstasy/" << RandomValue;
						break;
					case 2:
						ScoringVariables[0] = -0.1;
						ScoringVariables[1] = 0.2;
						ScoringVariables[2] = -0.1;
						ScoringVariables[3] = -0.1;
						ScoringVariables[4] = 0.5;
						ScoringVariables[5] = -0.1;
						ScoringVariables[6] = 0;
						ScoringVariables[7] = -0.5;
						RandomValue = ((20 - 1) * rand() / RAND_MAX) + 1;
						FilePathOStringStream << "Grief/" << RandomValue;
						break;
					case 3:
						ScoringVariables[0] = -0.2;
						ScoringVariables[1] = -0.5;
						ScoringVariables[2] = 0;
						ScoringVariables[3] = -0.6;
						ScoringVariables[4] = -0.2;
						ScoringVariables[5] = 0.5;
						ScoringVariables[6] = 0.1;
						ScoringVariables[7] = 0.2;
						RandomValue = ((22 - 1) * rand() / RAND_MAX) + 1;
						FilePathOStringStream << "Loathing/" << RandomValue;
						break;
					case 4:
						ScoringVariables[0] = 0.05;
						ScoringVariables[1] = 0;
						ScoringVariables[2] = -0.5;
						ScoringVariables[3] = -0.5;
						ScoringVariables[4] = -0.5;
						ScoringVariables[5] = -0.3;
						ScoringVariables[6] = -0.4;
						ScoringVariables[7] = 0.1;
						RandomValue = ((31 - 1) * rand() / RAND_MAX) + 1;
						FilePathOStringStream << "Neutral/" << RandomValue;
						break;
					case 5:
						ScoringVariables[0] = -0.5;
						ScoringVariables[1] = -0.5;
						ScoringVariables[2] = -0.1;
						ScoringVariables[3] = -0.2;
						ScoringVariables[4] = 0;
						ScoringVariables[5] = 0.2;
						ScoringVariables[6] = 0.5;
						ScoringVariables[7] = -0.5;
						RandomValue = ((6 - 1) * rand() / RAND_MAX) + 1;
						FilePathOStringStream << "Rage/" << RandomValue;
						break;
					}
				}else if (OtherPersonPresent == true) {
					//not admin
					FilePathOStringStream << "Emotinal training module/Training data/From Other/";
					RandomValue = ((8 - MinValue) * rand() / RAND_MAX) + MinValue;
					switch (RandomValue) {
					case 0:
						ScoringVariables[0] = 0.1;
						ScoringVariables[1] = 0.5;
						ScoringVariables[2] = -0.7;
						ScoringVariables[3] = 0.1;
						ScoringVariables[4] = -0.7;
						ScoringVariables[5] = -0.7;
						ScoringVariables[6] = -0.6;
						ScoringVariables[7] = 0;
						RandomValue = ((58 - 1) * rand() / RAND_MAX) + 1;
						FilePathOStringStream << "Admiration/" << RandomValue;
						break;
					case 1:
						ScoringVariables[0] = 0.1;
						ScoringVariables[1] = 0.35;
						ScoringVariables[2] = -0.1;
						ScoringVariables[3] = 0.6;
						ScoringVariables[4] = -0.7;
						ScoringVariables[5] = -0.7;
						ScoringVariables[6] = -0.5;
						ScoringVariables[7] = 0.5;
						RandomValue = ((43 - 1) * rand() / RAND_MAX) + 1;
						FilePathOStringStream << "Amazement/" << RandomValue;
						break;
					case 2:
						ScoringVariables[0] = 0.5;
						ScoringVariables[1] = 0.1;
						ScoringVariables[2] = -0.7;
						ScoringVariables[3] = 0;
						ScoringVariables[4] = -0.6;
						ScoringVariables[5] = -0.7;
						ScoringVariables[6] = -0.65;
						ScoringVariables[7] = -0.1;
						RandomValue = ((108 - 1) * rand() / RAND_MAX) + 1;
						FilePathOStringStream << "Ecstasy/" << RandomValue;
						break;
					case 3:
						ScoringVariables[0] = -0.1;
						ScoringVariables[1] = 0.2;
						ScoringVariables[2] = -0.1;
						ScoringVariables[3] = -0.1;
						ScoringVariables[4] = 0.5;
						ScoringVariables[5] = -0.1;
						ScoringVariables[6] = 0;
						ScoringVariables[7] = -0.5;
						RandomValue = ((77 - 1) * rand() / RAND_MAX) + 1;
						FilePathOStringStream << "Grief/" << RandomValue;
						break;
					case 4:
						ScoringVariables[0] = -0.2;
						ScoringVariables[1] = -0.5;
						ScoringVariables[2] = 0;
						ScoringVariables[3] = -0.6;
						ScoringVariables[4] = -0.2;
						ScoringVariables[5] = 0.5;
						ScoringVariables[6] = 0.1;
						ScoringVariables[7] = 0.2;
						RandomValue = ((63 - 1) * rand() / RAND_MAX) + 1;
						FilePathOStringStream << "Loathing/" << RandomValue;
						break;
					case 5:
						ScoringVariables[0] = 0.05;
						ScoringVariables[1] = 0;
						ScoringVariables[2] = -0.5;
						ScoringVariables[3] = -0.5;
						ScoringVariables[4] = -0.5;
						ScoringVariables[5] = -0.3;
						ScoringVariables[6] = -0.4;
						ScoringVariables[7] = 0.1;
						RandomValue = ((102 - 1) * rand() / RAND_MAX) + 1;
						FilePathOStringStream << "Neutral/" << RandomValue;
						break;
					case 6:
						ScoringVariables[0] = -0.5;
						ScoringVariables[1] = -0.5;
						ScoringVariables[2] = -0.1;
						ScoringVariables[3] = -0.2;
						ScoringVariables[4] = 0;
						ScoringVariables[5] = 0.2;
						ScoringVariables[6] = 0.5;
						ScoringVariables[7] = -0.5;
						RandomValue = ((83 - 1) * rand() / RAND_MAX) + 1;
						FilePathOStringStream << "Rage/" << RandomValue;
						break;
					case 7:
						ScoringVariables[0] = -0.6;
						ScoringVariables[1] = -0.6;
						ScoringVariables[2] = 0.5;
						ScoringVariables[3] = 0.2;
						ScoringVariables[4] = -0.2;
						ScoringVariables[5] = 0.2;
						ScoringVariables[6] = -0.3;
						ScoringVariables[7] = 0.4;
						RandomValue = ((50 - 1) * rand() / RAND_MAX) + 1;
						FilePathOStringStream << "Terror/" << RandomValue;
						break;
					case 8:
						ScoringVariables[0] = -0.5;
						ScoringVariables[1] = -0.3;
						ScoringVariables[2] = 0.2;
						ScoringVariables[3] = 0.1;
						ScoringVariables[4] = -0.5;
						ScoringVariables[5] = 0;
						ScoringVariables[6] = -0.3;
						ScoringVariables[7] = 0.5;
						RandomValue = ((66 - 1) * rand() / RAND_MAX) + 1;
						FilePathOStringStream << "Vigilance/" << RandomValue;
						break;
					}
				}
			}
		}
		//Is admin
		if ((IsAdmin == true) || (OtherPersonPresent == false)) {
			ScoringVariables[8] = 1;
		}else {
			ScoringVariables[8] = -1;
		}
		//Admin busy
		if (AdminBusy == true) {
			ScoringVariables[9] = 1;
		}else {
			ScoringVariables[9] = -1;
		}
		//Person present
		if ((AdminPresent == true) || (OtherPersonPresent == true)) {
			ScoringVariables[10] = 1;
		}else {
			ScoringVariables[10] = -1;
		}

		if ((IsAdmin == true)||(OtherPersonPresent == true)) {
			
			FilePathString = FilePathOStringStream.str();
			LoadAudio(
				&FilePathString,
				*(&AudioIOToggledUpdateFromMicFlag),
				*(&AudioInputBufferFromMicThread)
			);
		}

		if (AdminLeaving == true) {
			AdminPresent = false;
			AdminLeaving = false;
		}

		RandomValue = ((1 - MinValue) * rand() / RAND_MAX) + MinValue;
		if (RandomValue == 0) {
			RandomValue = ((2 - 10) * rand() / RAND_MAX) + 10;
			SilentAudio(
				RandomValue,
				*(&AudioIOToggledUpdateFromMicFlag),
				*(&AudioInputBufferFromMicThread)
			);
		}else {
			RandomValue = ((2 - 10) * rand() / RAND_MAX) + 10;
			RandomAudio(
				RandomValue,
				*(&AudioIOToggledUpdateFromMicFlag),
				*(&AudioInputBufferFromMicThread)
			);
		}



	}
	return 0;
}

int AudioBufferFromMicThread(
	atomic<unsigned int> *RunProgram,
	atomic<unsigned int> *ProgramStartup,
	atomic<unsigned int> *AudioIOToggledUpdateFlag,
	atomic<unsigned int>& AudioBufferToCoreUpdateFlag,
	atomic<unsigned int> *AudioBufferFromCoreAcknowledgementFlag,
	vector<float> *AudioInputBufferFromAudioIOThread,
	vector<float>& AudioInputBufferToCoreThread
) {
	atomic<unsigned int> LastAudioIOToggledUpdateFlag = 0;
	while (*ProgramStartup == 1) {
		//Wait for startup to finish
	}
	while (*RunProgram == 1) {
		if (*AudioIOToggledUpdateFlag != LastAudioIOToggledUpdateFlag) {
			AudioInputBufferToCoreThread = *AudioInputBufferFromAudioIOThread;
			if (AudioInputBufferToCoreThread.size() != FRAMES_PER_BUFFER) {
				AudioInputBufferToCoreThread.resize(FRAMES_PER_BUFFER);
			}
			AudioBufferToCoreUpdateFlag = 1;
			while (*AudioBufferFromCoreAcknowledgementFlag == 0) {
				//Wait for acknowledgement of data transfer
			}
			AudioBufferToCoreUpdateFlag = 0;
			while (*AudioBufferFromCoreAcknowledgementFlag == 1) {
				//Wait for acknowledgement has reset
			}

			LastAudioIOToggledUpdateFlag = (unsigned int)*AudioIOToggledUpdateFlag;
		}
	}
	return 1;
}

int AudioBufferToSpeakerThread(
	atomic<unsigned int> *RunProgram,
	atomic<unsigned int> *ProgramStartup,
	atomic<unsigned int> *AudioIOToggledUpdateFlag,
	atomic<unsigned int>& AudioBufferToCoreUpdateFlag,/////////////
	atomic<unsigned int> *AudioBufferFromCoreAcknowledgementFlag,
	vector<float>& AudioOutputBufferToAudioIOThread,
	vector<float> *AudioOutputBufferFromCoreThread
) {
	atomic<unsigned int> LastAudioIOToggledUpdateFlag = 0;
	while (*ProgramStartup == 1) {
		//Wait for startup to finish
	}
	while (*RunProgram == 1) {
		if ((*AudioIOToggledUpdateFlag) != LastAudioIOToggledUpdateFlag) {
			AudioOutputBufferToAudioIOThread = *AudioOutputBufferFromCoreThread;
			AudioBufferToCoreUpdateFlag = 1;
			while (*AudioBufferFromCoreAcknowledgementFlag == 0) {
				//Wait for acknowledgement of data transfer
			}
			AudioBufferToCoreUpdateFlag = 0;
			while (*AudioBufferFromCoreAcknowledgementFlag == 1) {
				//Wait for acknowledgement has reset
			}
			LastAudioIOToggledUpdateFlag = (unsigned int)*AudioIOToggledUpdateFlag;
		}
	}
	return 1;
}


int OLD_ALICECoreThread(
	atomic<unsigned int> *RunProgram,
	atomic<unsigned int> *ProgramStartup,
	unsigned int TrainingMode, //0=Emotinal training, 1=Offline trainer training , 2=Offline training, 3=Online training
	vector<float> *ScoringVariables,
	atomic<unsigned int> *AudioBufferFromAudioBufferUpdateFromMicFlag,
	atomic<unsigned int>& AudioBufferToAudioBufferAcknowledgementFromMicFlag,
	atomic<unsigned int> *AudioBufferFromAudioBufferUpdateToSpeakerFlag,
	atomic<unsigned int>& AudioBufferToAudioBufferAcknowledgementToSpeakerFlag,
	vector<float> *AudioInputBufferFromAudioBufferThread,
	vector<float>& AudioOutputBufferToAudioBufferThread,
	vector<float>& EmotinoalOutputsToUI,
	unsigned int& NumOfIndividualInGeneration,
	unsigned int& GenerationNum,
	float &NetworkScore
) {
	
	vector<float> AudioInputBuffer;
	vector<float> AudioInputSecondaryBuffer;
	vector<float> TimeDateInput;

	float NetworkScoreBuffer = 0;
	NetworkScore = 0;
	vector<float> AudioOutputBuffer;
	vector<float> EmotinoalOutputs;

	vector<OLD_NeuralNetworkModule> NeuralNetworkGeneration;
	OLD_NeuralNetworkModule NeuralNetwork;
	unsigned int NetworkModule = 0;
	GenerationNum;
	NumOfIndividualInGeneration;

vector<float> GenerationScoreVector;

AudioInputBuffer.resize(FRAMES_PER_BUFFER);
if (AudioInputBuffer.size() == 0) {
	AudioInputBuffer.resize(FRAMES_PER_BUFFER);
}
if (AudioOutputBuffer.size() == 0) {
	AudioOutputBuffer.resize(FRAMES_PER_BUFFER);
}
/*
BufferNodeValues.resize(33);///
for (unsigned int i = 0; i < 22; i++) {
	if (i < 11) {
		LSTMInputNodeIDBuffer.push_back(0);
	}else {
		LSTMInputNodeIDBuffer.push_back(i);
	}
	LSTMOutputNodeIDBuffer.push_back(i + 11);
}

//LSTMWeightsListBuffer.resize(22*8);
for (unsigned int i = 0; i < 22*8; i++) {
	LSTMWeightsListBuffer.push_back(3.1);///
}

LSTMPreviousCellStateValueBuffer.resize(22);//ct-1///
LSTMPreviousOutputValueBuffer.resize(22);	//ht-1///
LSTMForgetGateValueBuffer.resize(22);		//ft///
LSTMInputGateValueBuffer.resize(22);		//it///
LSTMCandidateValueBuffer.resize(22);		//~ct///
LSTMOutputGateValueBuffer.resize(22);		//ot///
LSTMCellStateValueBuffer.resize(22);		//ct///
LSTMOutputValueBuffer.resize(22);			//ht///
//	NeuronNodeStartBuffer.resize(1);
//	NeuronNodeEndBuffer.resize(1);
//	NeuronInputNodeIDsBuffer.resize(1);
//	NeuronOutputNodeIDsBuffer.resize(1);
//	NeuronOutputNodeBuffer.resize(1);///
//	NeuronWeightsBuffer.resize(1);
	(*ScoringVariables).resize(11);
	*/


TrainingMode = 0;		//0=Emotinal training, 1=Offline training, 2=Online training
//	float NetworkScoreBuffer;
//	float NetworkScore;
std::chrono::steady_clock::time_point TimeStamp;
std::chrono::steady_clock::time_point OldTimeStamp;
float CalcTimeDiff_nseconds = 40.0;
TimeStamp = std::chrono::high_resolution_clock::now();
OldTimeStamp = TimeStamp;
while (*ProgramStartup == 1) {
	//Wait for startup to finish
}
NeuralNetworkGeneration.resize(25);
for (NumOfIndividualInGeneration = 0; NumOfIndividualInGeneration < 25; NumOfIndividualInGeneration++) {
	OLD_LoadNeuralNetworkModule(
		NeuralNetworkGeneration[NumOfIndividualInGeneration],
		&NetworkModule,		//0=Emotinal training module, 1=Core module, 2=Offline training module
		&GenerationNum,
		&NumOfIndividualInGeneration
	);
}
NumOfIndividualInGeneration = 0;
bool InitialRun = true;
while (*RunProgram == 1) {
	/*
	OldTimeStamp = TimeStamp;
	TimeStamp = std::chrono::high_resolution_clock::now();
	CalcTimeDiff_nseconds = (float)(1 / std::chrono::duration_cast<std::chrono::nanoseconds>(OldTimeStamp - TimeStamp).count() / 1e9);
	system("CLS");
	cout << FPSCalcTimeDiff_nseconds  << endl;
	*/
	if (InitialRun == true) {
		InitialRun = false;
		//Load current generation
		NeuralNetwork = NeuralNetworkGeneration[NumOfIndividualInGeneration];
		//NumOfIndividualInGeneration++;
		NetworkScoreBuffer = 0;
		NetworkScore = 0;
	}
	TimeStamp = std::chrono::high_resolution_clock::now();
	//CalcTimeDiff_nseconds = (float)(1 / std::chrono::duration_cast<std::chrono::nanoseconds>(OldTimeStamp - TimeStamp).count() / 1e9);
	CalcTimeDiff_nseconds = (float)(std::chrono::duration_cast<std::chrono::seconds>(TimeStamp - OldTimeStamp).count());
	if (CalcTimeDiff_nseconds > TEST_RUNTIME_IN_SECONDS) {
		OldTimeStamp = TimeStamp;

		//Save score of last individual
		GenerationScoreVector.push_back(NetworkScore / TEST_RUNTIME_IN_SECONDS); //To make score time indepdant, as usually longer tests will almost always result in much higher scores
		NumOfIndividualInGeneration++;
		NetworkScoreBuffer = 0;
		NetworkScore = 0;

		if (NumOfIndividualInGeneration < 25) {	//Load next indivudal
			NeuralNetwork = NeuralNetworkGeneration[NumOfIndividualInGeneration];
		}else {		//If finished with current generation, generate next one
			//Save scores of last generation
			std::ostringstream FilePath;
			if (NetworkModule == 0) {
				FilePath << "C:/ALICE/Emotinal training module/Gen-" << GenerationNum << "_Scores-" << ".DATA";
			}
			else if (NetworkModule == 1) {
				FilePath << "C:/ALICE/Offline training module/Gen-" << GenerationNum << "_Scores-" << ".DATA";
			}
			else {
				FilePath << "C:/ALICE/Core module/Gen-" << GenerationNum << "_Scores-" << ".DATA";
			}


			SaveVector(
				&FilePath,
				&GenerationScoreVector
			);

			vector<OLD_NeuralNetworkModule> BufferNeuralNetworkNewGeneration;
			vector<unsigned int> FiveSurviorsList;
			//Check vector is empty
			while (BufferNeuralNetworkNewGeneration.size() != 0) {
				BufferNeuralNetworkNewGeneration.clear();
			}
			FiveSurviorsList.resize(5);
			SelectSurvivors(GenerationScoreVector, FiveSurviorsList);
			srand(time(0));
			//Cycle through each survivor
			for (unsigned int n = 0; n < 5; n++) {
				//Cycle through mating partners
				for (unsigned int i = 0; i < 5; i++) {
					//Copy the network, will be modified as needed later
					BufferNeuralNetworkNewGeneration.push_back(NeuralNetworkGeneration[FiveSurviorsList[n]]);
					if (n != i) {						//Check for self-mating
						const float MinWeight = 0.0;
						const float MaxWeight = 1.0;
						float RandomChance;
						for (unsigned int j = 0; j < BufferNeuralNetworkNewGeneration[(n * 5) + i].LSTMWeightsListBuffer.size(); j++) {
							RandomChance = ((MaxWeight - MinWeight) * (float)rand() / (float)RAND_MAX) + MinWeight;
							if (RandomChance < 0.5) {//Use the weights from the other survivor
								BufferNeuralNetworkNewGeneration[(n * 5) + i].LSTMWeightsListBuffer[j] = NeuralNetworkGeneration[FiveSurviorsList[i]].LSTMWeightsListBuffer[j];
							}
							RandomChance = ((MaxWeight - MinWeight) * (float)rand() / (float)RAND_MAX) + MinWeight;
							if (RandomChance < 0.1) {//Mutate the weight?
								BufferNeuralNetworkNewGeneration[(n * 5) + i].LSTMWeightsListBuffer[j] *= ((5.0 - -5.0) * (float)rand() / (float)RAND_MAX) + -5.0; //Mutate
																																								   //Clamp result
								if (BufferNeuralNetworkNewGeneration[(n * 5) + i].LSTMWeightsListBuffer[j] > 10.0f) {
									BufferNeuralNetworkNewGeneration[(n * 5) + i].LSTMWeightsListBuffer[j] = 10.0f;
								}else if (BufferNeuralNetworkNewGeneration[(n * 5) + i].LSTMWeightsListBuffer[j] < -10.0f) {
									BufferNeuralNetworkNewGeneration[(n * 5) + i].LSTMWeightsListBuffer[j] = -10.0f;
								}
							}
						}
						for (unsigned int j = 0; j < BufferNeuralNetworkNewGeneration[(n * 5) + i].NeuronWeightsBuffer.size(); j++) {
							RandomChance = ((MaxWeight - MinWeight) * (float)rand() / (float)RAND_MAX) + MinWeight;
							if (RandomChance < 0.5) {//Use the weights from the other survivor
								BufferNeuralNetworkNewGeneration[(n * 5) + i].NeuronWeightsBuffer[j] = NeuralNetworkGeneration[FiveSurviorsList[i]].NeuronWeightsBuffer[j];
							}else if (BufferNeuralNetworkNewGeneration[(n * 5) + i].NeuronWeightsBuffer[j] < 0.0f) {
								BufferNeuralNetworkNewGeneration[(n * 5) + i].NeuronWeightsBuffer[j] = 0.0f;
							}
							RandomChance = ((MaxWeight - MinWeight) * (float)rand() / (float)RAND_MAX) + MinWeight;
							if (RandomChance < 0.1) {//Mutate the weight?
								BufferNeuralNetworkNewGeneration[(n * 5) + i].NeuronWeightsBuffer[j] *= ((5.0 - -5.0) * (float)rand() / (float)RAND_MAX) + -5.0; //Mutate
																																								 //Clamp result
								if (BufferNeuralNetworkNewGeneration[(n * 5) + i].NeuronWeightsBuffer[j] > 10.0f) {
									BufferNeuralNetworkNewGeneration[(n * 5) + i].NeuronWeightsBuffer[j] = 10.0f;
								}else if (BufferNeuralNetworkNewGeneration[(n * 5) + i].NeuronWeightsBuffer[j] < 0.0f) {
									BufferNeuralNetworkNewGeneration[(n * 5) + i].NeuronWeightsBuffer[j] = 0.0f;
								}
							}
						}
					}

				}
			}
			NeuralNetworkGeneration = BufferNeuralNetworkNewGeneration;
			GenerationNum++;
			for (NumOfIndividualInGeneration = 0; NumOfIndividualInGeneration < 25; NumOfIndividualInGeneration++) {
				OLD_SaveNeuralNetworkModule(
					&NeuralNetworkGeneration[NumOfIndividualInGeneration],
					&NetworkModule,		//0=Emotinal training module, 1=Core module, 2=Offline training module
					&GenerationNum,
					&NumOfIndividualInGeneration
				);
			}
			NumOfIndividualInGeneration = 0;
			GenerationScoreVector.clear();
		}
		//End of next generation creation

	}

	//	while(AudioInputBuffer.size() > 0){
		//	AudioInputBuffer.clear();
	//	}

		time_t	t = time(0);   // get time now
		struct	tm * now = localtime(&t);
		while (TimeDateInput.size() != 0) {
			TimeDateInput.clear();
		}
		TimeDateInput.push_back(now->tm_sec);
		TimeDateInput.push_back(now->tm_min);
		TimeDateInput.push_back(now->tm_hour);
		TimeDateInput.push_back(now->tm_mday);
		TimeDateInput.push_back(now->tm_mon + 1);
		//TimeDateInput.push_back(now->tm_year + 1900);+

		AudioInputSecondaryBuffer = *AudioInputBufferFromAudioBufferThread;
		if (AudioInputBuffer.size() == FRAMES_PER_BUFFER) {
			AudioInputBuffer.resize(AudioInputSecondaryBuffer.size());
			std::copy(AudioInputSecondaryBuffer.begin(), AudioInputSecondaryBuffer.end(), AudioInputBuffer.begin());
		}
		if (AudioInputBuffer.size() != FRAMES_PER_BUFFER) {
			AudioInputBuffer.resize(FRAMES_PER_BUFFER);
		}
		if(NeuralNetwork.NeuronNodeStartBuffer.size() == 1){
			OLD_RunALICECoreNoStandardNeurons(
				AudioInputBuffer,
				TimeDateInput,
				NeuralNetwork.BufferNodeValues,
				NeuralNetwork.LSTMInputNodeIDBuffer,
				NeuralNetwork.LSTMOutputNodeIDBuffer,
				NeuralNetwork.LSTMWeightsListBuffer,
				NeuralNetwork.LSTMPreviousCellStateValueBuffer,	//ct-1
				NeuralNetwork.LSTMPreviousOutputValueBuffer,		//ht-1
				NeuralNetwork.LSTMForgetGateValueBuffer,			//ft
				NeuralNetwork.LSTMInputGateValueBuffer,			//it
				NeuralNetwork.LSTMCandidateValueBuffer,			//~ct
				NeuralNetwork.LSTMOutputGateValueBuffer,			//ot
				NeuralNetwork.LSTMCellStateValueBuffer,			//ct
				NeuralNetwork.LSTMOutputValueBuffer,				//ht
				(*ScoringVariables),
				TrainingMode,		//0=Emotinal training, 1=Offline training, 2=Online training
				NetworkScoreBuffer,
				AudioOutputBuffer
			);
		}else {
			OLD_RunALICECore(
				AudioInputBuffer,
				TimeDateInput,
				NeuralNetwork.BufferNodeValues,
				NeuralNetwork.LSTMInputNodeIDBuffer,
				NeuralNetwork.LSTMOutputNodeIDBuffer,
				NeuralNetwork.LSTMWeightsListBuffer,
				NeuralNetwork.LSTMPreviousCellStateValueBuffer,	//ct-1
				NeuralNetwork.LSTMPreviousOutputValueBuffer,		//ht-1
				NeuralNetwork.LSTMForgetGateValueBuffer,			//ft
				NeuralNetwork.LSTMInputGateValueBuffer,			//it
				NeuralNetwork.LSTMCandidateValueBuffer,			//~ct
				NeuralNetwork.LSTMOutputGateValueBuffer,			//ot
				NeuralNetwork.LSTMCellStateValueBuffer,			//ct
				NeuralNetwork.LSTMOutputValueBuffer,				//ht
				NeuralNetwork.NeuronNodeStartBuffer,
				NeuralNetwork.NeuronNodeEndBuffer,
				NeuralNetwork.NeuronInputNodeIDsBuffer,
				NeuralNetwork.NeuronOutputNodeIDsBuffer,
				NeuralNetwork.NeuronOutputNodeBuffer,
				NeuralNetwork.NeuronWeightsBuffer,
				(*ScoringVariables),
				TrainingMode,		//0=Emotinal training, 1=Offline training, 2=Online training
				NetworkScoreBuffer,
				AudioOutputBuffer
			);
		}
		NetworkScore += NetworkScoreBuffer;
		EmotinoalOutputsToUI.resize(11);
		std::copy(NeuralNetwork.BufferNodeValues.begin() + 22, NeuralNetwork.BufferNodeValues.begin() + 33,EmotinoalOutputsToUI.begin());
		//EmotinoalOutputsToUI = (BufferNodeValues.begin(), BufferNodeValues.begin() + 11);


		while (*AudioBufferFromAudioBufferUpdateToSpeakerFlag == 0) {
			//Wait for update flag to begin data tranfer
		}
		while (AudioOutputBufferToAudioBufferThread.size() > 0) {
			AudioOutputBufferToAudioBufferThread.clear();
		}
		AudioOutputBufferToAudioBufferThread.resize(AudioOutputBuffer.size());
		std::copy(AudioOutputBuffer.begin(), AudioOutputBuffer.end(), AudioOutputBufferToAudioBufferThread.begin());
		AudioBufferToAudioBufferAcknowledgementToSpeakerFlag = 1;
		while (*AudioBufferFromAudioBufferUpdateToSpeakerFlag == 1) {
			//Wait for update flag to reset
		}
		AudioBufferToAudioBufferAcknowledgementToSpeakerFlag = 0;
	}
	return 1;
}

int ALICECoreThread(
	atomic<unsigned int> *RunProgram,
	atomic<unsigned int> *ProgramStartup,
	unsigned int TrainingMode, //0=Emotinal training, 1=Offline trainer training , 2=Offline training, 3=Online training
	vector<float> *ScoringVariables,
	atomic<unsigned int> *AudioBufferFromAudioBufferUpdateFromMicFlag,
	atomic<unsigned int>& AudioBufferToAudioBufferAcknowledgementFromMicFlag,
	atomic<unsigned int> *AudioBufferFromAudioBufferUpdateToSpeakerFlag,
	atomic<unsigned int>& AudioBufferToAudioBufferAcknowledgementToSpeakerFlag,
	vector<float> *AudioInputBufferFromAudioBufferThread,
	vector<float>& AudioOutputBufferToAudioBufferThread,
	vector<float>& EmotinoalOutputsToUI,
	unsigned int& NumOfIndividualInGeneration,
	unsigned int& GenerationNum,
	float &NetworkScore
) {

	vector<float> AudioInputBuffer;
	vector<float> AudioInputSecondaryBuffer;
	vector<float> TimeDateInput;

	float NetworkScoreBuffer = 0;
	NetworkScore = 0;
	vector<float> AudioOutputBuffer;
	vector<float> EmotinoalOutputs;

	vector<NeuralNetworkModule> NeuralNetworkGeneration;
	NeuralNetworkModule NeuralNetworkCombinedModule;
	unsigned int NetworkModule = 0;
	GenerationNum;
	NumOfIndividualInGeneration;

	vector<float> GenerationScoreVector;

	AudioInputBuffer.resize(FRAMES_PER_BUFFER);
	if (AudioInputBuffer.size() == 0) {
		AudioInputBuffer.resize(FRAMES_PER_BUFFER);
	}
	if (AudioOutputBuffer.size() == 0) {
		AudioOutputBuffer.resize(FRAMES_PER_BUFFER);
	}

	TrainingMode = 0;		//0=Emotinal training, 1=Offline training, 2=Online training
							//	float NetworkScoreBuffer;
							//	float NetworkScore;
	std::chrono::steady_clock::time_point TimeStamp;
	std::chrono::steady_clock::time_point OldTimeStamp;
	float CalcTimeDiff_nseconds = 40.0;
	TimeStamp = std::chrono::high_resolution_clock::now();
	OldTimeStamp = TimeStamp;
	while (*ProgramStartup == 1) {
		//Wait for startup to finish
	}
	NeuralNetworkGeneration.resize(25);
	for (NumOfIndividualInGeneration = 0; NumOfIndividualInGeneration < 25; NumOfIndividualInGeneration++) {
		LoadNeuralNetworkModule(
			NeuralNetworkGeneration[NumOfIndividualInGeneration],
			&NetworkModule,		//0=Emotinal training module, 1=Core module, 2=Offline training module
			&GenerationNum,
			&NumOfIndividualInGeneration
		);
	}
	NumOfIndividualInGeneration = 0;
	bool InitialRun = true;
	while (*RunProgram == 1) {
		/*
		OldTimeStamp = TimeStamp;
		TimeStamp = std::chrono::high_resolution_clock::now();
		CalcTimeDiff_nseconds = (float)(1 / std::chrono::duration_cast<std::chrono::nanoseconds>(OldTimeStamp - TimeStamp).count() / 1e9);
		system("CLS");
		cout << FPSCalcTimeDiff_nseconds  << endl;
		*/
		if (InitialRun == true) {
			InitialRun = false;
			//Load current generation
			NeuralNetworkCombinedModule = NeuralNetworkGeneration[NumOfIndividualInGeneration];
			//NumOfIndividualInGeneration++;
			NetworkScoreBuffer = 0;
			NetworkScore = 0;
		}
		TimeStamp = std::chrono::high_resolution_clock::now();
		//CalcTimeDiff_nseconds = (float)(1 / std::chrono::duration_cast<std::chrono::nanoseconds>(OldTimeStamp - TimeStamp).count() / 1e9);
		CalcTimeDiff_nseconds = (float)(std::chrono::duration_cast<std::chrono::seconds>(TimeStamp - OldTimeStamp).count());
		if (CalcTimeDiff_nseconds > TEST_RUNTIME_IN_SECONDS) {
			OldTimeStamp = TimeStamp;

			//Save score of last individual
			GenerationScoreVector.push_back(NetworkScore / TEST_RUNTIME_IN_SECONDS); //To make score time indepdant, as usually longer tests will almost always result in much higher scores
			NumOfIndividualInGeneration++;
			NetworkScoreBuffer = 0;
			NetworkScore = 0;

			if (NumOfIndividualInGeneration < 25) {	//Load next indivudal
				NeuralNetworkCombinedModule = NeuralNetworkGeneration[NumOfIndividualInGeneration];
			}
			else {		//If finished with current generation, generate next one
						//Save scores of last generation
				std::ostringstream FilePath;
				if (NetworkModule == 0) {
					FilePath << "C:/ALICE/Emotinal training module/Gen-" << GenerationNum << "_Scores-" << ".DATA";
				}else if (NetworkModule == 1) {
					FilePath << "C:/ALICE/Offline training module/Gen-" << GenerationNum << "_Scores-" << ".DATA";
				}else {
					FilePath << "C:/ALICE/Core module/Gen-" << GenerationNum << "_Scores-" << ".DATA";
				}


				SaveVector(
					&FilePath,
					&GenerationScoreVector
				);

				vector<NeuralNetworkModule> BufferNeuralNetworkNewGeneration;
				vector<unsigned int> FiveSurviorsList;
				//Check vector is empty
				while (BufferNeuralNetworkNewGeneration.size() != 0) {
					BufferNeuralNetworkNewGeneration.clear();
				}
				FiveSurviorsList.resize(5);
				SelectSurvivors(GenerationScoreVector, FiveSurviorsList);
				srand(time(0));
				//Cycle through each survivor
				for (unsigned int n = 0; n < 5; n++) {
					//Cycle through mating partners
					for (unsigned int i = 0; i < 5; i++) {
						//Copy the network, will be modified as needed later
						BufferNeuralNetworkNewGeneration.push_back(NeuralNetworkGeneration[FiveSurviorsList[n]]);
						if (n != i) {						//Check for self-mating
							const float MinWeight = 0.0;
							const float MaxWeight = 1.0;
							float RandomChance;
							for (unsigned int j = 0; j < BufferNeuralNetworkNewGeneration[(n * 5) + i].NeuronWeightsBuffer.size(); j++) {
								RandomChance = ((MaxWeight - MinWeight) * (float)rand() / (float)RAND_MAX) + MinWeight;
								if (RandomChance < 0.5) {//Use the weights from the other survivor
									BufferNeuralNetworkNewGeneration[(n * 5) + i].NeuronWeightsBuffer[j] = NeuralNetworkGeneration[FiveSurviorsList[i]].NeuronWeightsBuffer[j];
								}else if (BufferNeuralNetworkNewGeneration[(n * 5) + i].NeuronWeightsBuffer[j] < 0.0f) {
									BufferNeuralNetworkNewGeneration[(n * 5) + i].NeuronWeightsBuffer[j] = 0.0f;
								}
								RandomChance = ((MaxWeight - MinWeight) * (float)rand() / (float)RAND_MAX) + MinWeight;
								if (RandomChance < 0.1) {//Mutate the weight?
									BufferNeuralNetworkNewGeneration[(n * 5) + i].NeuronWeightsBuffer[j] *= ((5.0 - -5.0) * (float)rand() / (float)RAND_MAX) + -5.0; //Mutate
																																									 //Clamp result
									if (BufferNeuralNetworkNewGeneration[(n * 5) + i].NeuronWeightsBuffer[j] > 10.0f) {
										BufferNeuralNetworkNewGeneration[(n * 5) + i].NeuronWeightsBuffer[j] = 10.0f;
									}else if (BufferNeuralNetworkNewGeneration[(n * 5) + i].NeuronWeightsBuffer[j] < 0.0f) {
										BufferNeuralNetworkNewGeneration[(n * 5) + i].NeuronWeightsBuffer[j] = 0.0f;
									}
								}
							}
						}

					}
				}
				NeuralNetworkGeneration = BufferNeuralNetworkNewGeneration;
				GenerationNum++;
				for (NumOfIndividualInGeneration = 0; NumOfIndividualInGeneration < 25; NumOfIndividualInGeneration++) {
					SaveNeuralNetworkModule(
						&NeuralNetworkGeneration[NumOfIndividualInGeneration],
						&NetworkModule,		//0=Emotinal training module, 1=Core module, 2=Offline training module
						&GenerationNum,
						&NumOfIndividualInGeneration
					);
				}
				NumOfIndividualInGeneration = 0;
				GenerationScoreVector.clear();
			}
			//End of next generation creation

		}

		//	while(AudioInputBuffer.size() > 0){
		//	AudioInputBuffer.clear();
		//	}

		time_t	t = time(0);   // get time now
		struct	tm * now = localtime(&t);
		while (TimeDateInput.size() != 0) {
			TimeDateInput.clear();
		}
		TimeDateInput.push_back(now->tm_sec);
		TimeDateInput.push_back(now->tm_min);
		TimeDateInput.push_back(now->tm_hour);
		TimeDateInput.push_back(now->tm_mday);
		TimeDateInput.push_back(now->tm_mon + 1);
		//TimeDateInput.push_back(now->tm_year + 1900);+

		AudioInputSecondaryBuffer = *AudioInputBufferFromAudioBufferThread;
		if (AudioInputBuffer.size() == FRAMES_PER_BUFFER) {
			AudioInputBuffer.resize(AudioInputSecondaryBuffer.size());
			std::copy(AudioInputSecondaryBuffer.begin(), AudioInputSecondaryBuffer.end(), AudioInputBuffer.begin());
		}
		if (AudioInputBuffer.size() != FRAMES_PER_BUFFER) {
			AudioInputBuffer.resize(FRAMES_PER_BUFFER);
		}

		vector<float> InputData;
		InputData.resize(FRAMES_PER_BUFFER + 5);
		std::copy(AudioInputSecondaryBuffer.begin(), AudioInputSecondaryBuffer.end(), InputData.begin());
		std::copy(TimeDateInput.begin(), TimeDateInput.end(), InputData.end() - 5);
		NeuralNetwork AICore;
		AICore.LoadInputData(InputData);
		AICore.OutputData.resize(1);
		AICore.LoadScoringVariables((*ScoringVariables));
		AICore.LoadNeuralNetworkModule(NeuralNetworkCombinedModule);
		AICore.RunAICore();
		AICore.UnloadOutputData(AudioOutputBuffer);
		AICore.UnloadScore(NetworkScoreBuffer);

		NetworkScore += NetworkScoreBuffer;
		EmotinoalOutputsToUI.resize(11);
		std::copy(NeuralNetworkCombinedModule.BufferNodeValues.begin() + 5, NeuralNetworkCombinedModule.BufferNodeValues.begin() + 16, EmotinoalOutputsToUI.begin());
		//std::copy(NeuralNetworkCombinedModule.BufferNodeValues.begin() + 22, NeuralNetworkCombinedModule.BufferNodeValues.begin() + 33, EmotinoalOutputsToUI.begin());

		while (*AudioBufferFromAudioBufferUpdateToSpeakerFlag == 0) {
			//Wait for update flag to begin data tranfer
		}
		while (AudioOutputBufferToAudioBufferThread.size() > 0) {
			AudioOutputBufferToAudioBufferThread.clear();
		}
		AudioOutputBufferToAudioBufferThread.resize(AudioOutputBuffer.size());
		std::copy(AudioOutputBuffer.begin(), AudioOutputBuffer.end(), AudioOutputBufferToAudioBufferThread.begin());
		AudioBufferToAudioBufferAcknowledgementToSpeakerFlag = 1;
		while (*AudioBufferFromAudioBufferUpdateToSpeakerFlag == 1) {
			//Wait for update flag to reset
		}
		AudioBufferToAudioBufferAcknowledgementToSpeakerFlag = 0;
	}
	return 1;
}

int UserInterfaceOutputsThread(
	atomic<unsigned int> *RunProgram,
	atomic<unsigned int> *ProgramStartup,
	vector<float> *EmotinoalOutputsToUI,
	unsigned int *NumOfIndividualInGeneration,
	unsigned int *GenerationNum,
	float *NetworkScore,
	vector<float> *ScoringVariables
) {
	vector<float> EmotinoalOutputs;
	unsigned int LocalNumOfIndividualInGeneration;
	unsigned int LocalGenerationNum;
	unsigned int LocalNetworkScore;
	vector<float> LocalEmotinoalOutputs;
	

	while (*ProgramStartup == 1) {
		//Wait for startup to finish
	}
	while (*RunProgram == 1) {
		LocalNumOfIndividualInGeneration = *NumOfIndividualInGeneration;
		LocalNetworkScore = *NetworkScore;
		LocalGenerationNum = *GenerationNum;
		EmotinoalOutputs.resize((*EmotinoalOutputsToUI).size());
		LocalEmotinoalOutputs.resize((*ScoringVariables).size());
		std::copy((*EmotinoalOutputsToUI).begin(),(*EmotinoalOutputsToUI).end(), EmotinoalOutputs.begin());
		std::copy((*ScoringVariables).begin(), (*ScoringVariables).end(), LocalEmotinoalOutputs.begin());
		if (EmotinoalOutputs.size() == 11) {
			system("CLS");
			cout.precision(3);
			cout << "Ecstasy: " << EmotinoalOutputs[0] << "		" << LocalEmotinoalOutputs[0] << endl;
			cout << "Admiration: " << EmotinoalOutputs[1] << "		" << LocalEmotinoalOutputs[1] << endl;
			cout << "Terror: " << EmotinoalOutputs[2] << "		" << LocalEmotinoalOutputs[2] << endl;
			cout << "Amazement: " << EmotinoalOutputs[3] << "		" << LocalEmotinoalOutputs[3] << endl;
			cout << "Grief: " << EmotinoalOutputs[4] << "		" << LocalEmotinoalOutputs[4] << endl;
			cout << "Loathing: " << EmotinoalOutputs[5] << "		" << LocalEmotinoalOutputs[5] << endl;
			cout << "Rage: " << EmotinoalOutputs[6] << "		" << LocalEmotinoalOutputs[6] << endl;
			cout << "Vigilance: " << EmotinoalOutputs[7] << "		" << LocalEmotinoalOutputs[7] << endl;
			cout << "ID: " << EmotinoalOutputs[8] << "		" << LocalEmotinoalOutputs[8] << endl;
			cout << "Busy: " << EmotinoalOutputs[9] << "		" << LocalEmotinoalOutputs[9] << endl;
			cout << "Present: " << EmotinoalOutputs[10] << "		" << LocalEmotinoalOutputs[10] << endl;
			cout << "Generation: " << LocalGenerationNum << endl;
			cout << "Individual: " << LocalNumOfIndividualInGeneration << endl;
			cout << "Current score: " << LocalNetworkScore << endl;
		}
	}
	return 1;
}

void main() {
	atomic<unsigned int> RunProgram = 0;
	atomic<unsigned int> ProgramStartup = 1;

	unsigned int NumOfIndividualInGeneration = 0;
	unsigned int GenerationNum = 0;//0
	float NetworkScore = 0;

	atomic<unsigned int> AudioIOToggledUpdateFromMicFlag = 0;
	atomic<unsigned int> AudioIOToggledUpdateToSpeakerFlag = 0;
	vector<float> AudioInputBufferFromMic;
	vector<float> AudioInputBufferToSpeaker;

	atomic<unsigned int> AudioBufferToCoreUpdateFromMicFlag;
	atomic<unsigned int> AudioBufferFromCoreAcknowledgementFromMicFlag;
	vector<float> AudioInputBufferToCore;

	atomic<unsigned int> AudioBufferToCoreUpdateToSpeakerFlag;
	atomic<unsigned int> AudioBufferFromCoreAcknowledgementToSpeakerFlag;
	vector<float> AudioOutputBufferFromCore;
	vector<float> EmotinoalOutputsToUI;

	vector<float> ScoringVariables;

	unsigned int TrainingMode = 0;

	AudioInputBufferToSpeaker.resize(FRAMES_PER_BUFFER);
	//if (TrainingMode != 0) {
	auto AudioIOThreadID = async(
		AudioIOThread,
		&RunProgram,
		&ProgramStartup,
		TrainingMode,
		ref(AudioIOToggledUpdateFromMicFlag),
		ref(AudioIOToggledUpdateToSpeakerFlag),
		ref(AudioInputBufferFromMic),
		&AudioInputBufferToSpeaker
	);
	//}

	auto AudioBufferFromMicThreadID = async(
		AudioBufferFromMicThread,
		&RunProgram,
		&ProgramStartup,
		&AudioIOToggledUpdateFromMicFlag,
		ref(AudioBufferToCoreUpdateFromMicFlag),//
		&AudioBufferFromCoreAcknowledgementFromMicFlag,//
		&AudioInputBufferFromMic,
		ref(AudioInputBufferToCore)
	);



	auto AudioBufferToSpeakerThreadID = async(
		AudioBufferToSpeakerThread,
		&RunProgram,
		&ProgramStartup,
		&AudioIOToggledUpdateToSpeakerFlag,
		ref(AudioBufferToCoreUpdateToSpeakerFlag),//
		&AudioBufferFromCoreAcknowledgementToSpeakerFlag,
		ref(AudioInputBufferToSpeaker),
		&AudioOutputBufferFromCore
	);
	auto ALICECoreThreadID = async(
		ALICECoreThread,
		&RunProgram,
		&ProgramStartup,
		TrainingMode,
		&ScoringVariables,
		&AudioBufferToCoreUpdateFromMicFlag,//
		ref(AudioBufferFromCoreAcknowledgementFromMicFlag),//
		&AudioBufferToCoreUpdateToSpeakerFlag,
		ref(AudioBufferFromCoreAcknowledgementToSpeakerFlag),
		&AudioInputBufferToCore,
		ref(AudioOutputBufferFromCore),
		ref(EmotinoalOutputsToUI),
		ref(NumOfIndividualInGeneration),
		ref(GenerationNum),
		ref(NetworkScore)
	);
	//if (TrainingMode == 0) {
	auto SupervisorThreadID = async(
		SupervisorThread,
		&RunProgram,
		&ProgramStartup,
		TrainingMode,
		ref(AudioIOToggledUpdateFromMicFlag),
		ref(AudioInputBufferFromMic),
		ref(ScoringVariables)
	);
	//}
	auto UserInterfaceOutputsThreadID = async(
		UserInterfaceOutputsThread,
		&RunProgram,
		&ProgramStartup,
		&EmotinoalOutputsToUI,
		&NumOfIndividualInGeneration,
		&GenerationNum,
		&NetworkScore,
		&ScoringVariables
	);

	//Setup finished, it's showtime
	RunProgram = 1;
	ProgramStartup = 0;
	while (RunProgram == 1) {
		//Run program
	}
}

void OLD_main6() {
	const float MinWeight = -10.0;
	const float MaxWeight = 10.0;
	float RandomChance;

	unsigned int NetworkModule = 0;
	unsigned int GenerationNum = 0;
	vector<OLD_NeuralNetworkModule> NeuralNetworkGeneration;
	NeuralNetworkGeneration.resize(25);
	for (unsigned int NumOfIndividualInGeneration = 0; NumOfIndividualInGeneration < 25; NumOfIndividualInGeneration++) {
		
		NeuralNetworkGeneration[NumOfIndividualInGeneration].BufferNodeValues.resize(33);
		for (unsigned int i = 0; i < 22; i++) {
			if (i < 11) {
				NeuralNetworkGeneration[NumOfIndividualInGeneration].LSTMInputNodeIDBuffer.push_back(0);
			}else {
				NeuralNetworkGeneration[NumOfIndividualInGeneration].LSTMInputNodeIDBuffer.push_back(i);
			}
			NeuralNetworkGeneration[NumOfIndividualInGeneration].LSTMOutputNodeIDBuffer.push_back(i + 11);
		}
		for (unsigned int i = 0; i < 22 * 8; i++) {
			RandomChance = ((MaxWeight - MinWeight) * (float)rand() / (float)RAND_MAX) + MinWeight;
			NeuralNetworkGeneration[NumOfIndividualInGeneration].LSTMWeightsListBuffer.push_back(RandomChance);
		}
		NeuralNetworkGeneration[NumOfIndividualInGeneration].LSTMPreviousCellStateValueBuffer.resize(22);//ct-1///
		NeuralNetworkGeneration[NumOfIndividualInGeneration].LSTMPreviousOutputValueBuffer.resize(22);	//ht-1///
		NeuralNetworkGeneration[NumOfIndividualInGeneration].LSTMForgetGateValueBuffer.resize(22);		//ft///
		NeuralNetworkGeneration[NumOfIndividualInGeneration].LSTMInputGateValueBuffer.resize(22);		//it///
		NeuralNetworkGeneration[NumOfIndividualInGeneration].LSTMCandidateValueBuffer.resize(22);		//~ct///
		NeuralNetworkGeneration[NumOfIndividualInGeneration].LSTMOutputGateValueBuffer.resize(22);		//ot///
		NeuralNetworkGeneration[NumOfIndividualInGeneration].LSTMCellStateValueBuffer.resize(22);		//ct///
		NeuralNetworkGeneration[NumOfIndividualInGeneration].LSTMOutputValueBuffer.resize(22);			//ht///
		NeuralNetworkGeneration[NumOfIndividualInGeneration].NeuronNodeStartBuffer.resize(1);
		NeuralNetworkGeneration[NumOfIndividualInGeneration].NeuronNodeEndBuffer.resize(1);
		NeuralNetworkGeneration[NumOfIndividualInGeneration].NeuronInputNodeIDsBuffer.resize(1);
		NeuralNetworkGeneration[NumOfIndividualInGeneration].NeuronOutputNodeIDsBuffer.resize(1);
		NeuralNetworkGeneration[NumOfIndividualInGeneration].NeuronOutputNodeBuffer.resize(1);///
		NeuralNetworkGeneration[NumOfIndividualInGeneration].NeuronWeightsBuffer.resize(1);
		//LSTMWeightsListBuffer
		OLD_SaveNeuralNetworkModule(
			&NeuralNetworkGeneration[NumOfIndividualInGeneration],
			&NetworkModule,		//0=Emotinal training module, 1=Core module, 2=Offline training module
			&GenerationNum,
			&NumOfIndividualInGeneration
		);
	}
}

void CreateLSTMUnit(
	NeuralNetworkModule &NeuralNetworkIndividual,
	unsigned int InputNodeID
) {
	const float MinWeight = -10.0;
	const float MaxWeight = 10.0;
	float RandomChance;
	unsigned int StartID = NeuralNetworkIndividual.BufferNodeValues.size();

	//NeuralNetworkIndividual.BufferNodeValues.resize(9 + NeuralNetworkIndividual.BufferNodeValues.size());
	NeuralNetworkIndividual.BufferNodeValues.resize(9 + NeuralNetworkIndividual.BufferNodeValues.size());
	//TypeOfNode
	NeuralNetworkIndividual.TypeOfNode.push_back(1);
	NeuralNetworkIndividual.TypeOfNode.push_back(1);
	NeuralNetworkIndividual.TypeOfNode.push_back(2);
	NeuralNetworkIndividual.TypeOfNode.push_back(1);
	NeuralNetworkIndividual.TypeOfNode.push_back(3);
	NeuralNetworkIndividual.TypeOfNode.push_back(3);
	NeuralNetworkIndividual.TypeOfNode.push_back(0);
	NeuralNetworkIndividual.TypeOfNode.push_back(3);
	NeuralNetworkIndividual.TypeOfNode.push_back(2);

	//NeuronNodeStartBuffer / NeuronNodeLength
	unsigned int WeightLength = 0;
	unsigned int CurrentWeightLength = 0;

	for (unsigned int i = 0; i < 7; i++) {
		WeightLength = 2;
		NeuralNetworkIndividual.NeuronNodeStartBuffer.push_back(CurrentWeightLength);
		NeuralNetworkIndividual.NeuronNodeLength.push_back(WeightLength);
		CurrentWeightLength += WeightLength;
	}
	WeightLength = 1;
	NeuralNetworkIndividual.NeuronNodeStartBuffer.push_back(CurrentWeightLength);
	NeuralNetworkIndividual.NeuronNodeLength.push_back(WeightLength);
	CurrentWeightLength += WeightLength;

	WeightLength = 2;
	NeuralNetworkIndividual.NeuronNodeStartBuffer.push_back(CurrentWeightLength);
	NeuralNetworkIndividual.NeuronNodeLength.push_back(WeightLength);
	CurrentWeightLength += WeightLength;

	//NeuronInputNodeIDsBuffer
	for (unsigned int i = 0; i < 4; i++) {
		NeuralNetworkIndividual.NeuronInputNodeIDsBuffer.push_back(InputNodeID);
		NeuralNetworkIndividual.NeuronInputNodeIDsBuffer.push_back(8 + StartID);
	}

	NeuralNetworkIndividual.NeuronInputNodeIDsBuffer.push_back(0 + StartID);
	NeuralNetworkIndividual.NeuronInputNodeIDsBuffer.push_back(6 + StartID);

	NeuralNetworkIndividual.NeuronInputNodeIDsBuffer.push_back(1 + StartID);
	NeuralNetworkIndividual.NeuronInputNodeIDsBuffer.push_back(2 + StartID);

	NeuralNetworkIndividual.NeuronInputNodeIDsBuffer.push_back(4 + StartID);
	NeuralNetworkIndividual.NeuronInputNodeIDsBuffer.push_back(5 + StartID);

	NeuralNetworkIndividual.NeuronInputNodeIDsBuffer.push_back(6 + StartID);

	NeuralNetworkIndividual.NeuronInputNodeIDsBuffer.push_back(3 + StartID);
	NeuralNetworkIndividual.NeuronInputNodeIDsBuffer.push_back(7 + StartID);
	//NeuronOutputNodeBuffer
	//NeuronOutputNodeIDsBuffer
	for (unsigned int i = 0; i < 9; i++) {
		NeuralNetworkIndividual.NeuronOutputNodeIDsBuffer.push_back(i + StartID);
	}

	//NeuronOutputNodeBuffer
	NeuralNetworkIndividual.NeuronOutputNodeBuffer.resize(NeuralNetworkIndividual.NeuronOutputNodeIDsBuffer.size());
	//NeuralNetworkIndividual.NeuronOutputNodeBuffer.resize(9 + NeuralNetworkIndividual.NeuronOutputNodeIDsBuffer.size());

	//NeuronWeightsBuffer / NeuronBiasBuffer
	for (unsigned int i = 0; i < 17; i++) {
		RandomChance = ((MaxWeight - MinWeight) * (float)rand() / (float)RAND_MAX) + MinWeight;
		NeuralNetworkIndividual.NeuronWeightsBuffer.push_back(RandomChance);

		RandomChance = ((MaxWeight - MinWeight) * (float)rand() / (float)RAND_MAX) + MinWeight;
		NeuralNetworkIndividual.NeuronBiasBuffer.push_back(RandomChance);
	}
}

void GenerateInitialGeneration() {
	const float MinWeight = -10.0;
	const float MaxWeight = 10.0;
	float RandomChance;

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


	unsigned int NetworkModule = 0;
	unsigned int GenerationNum = 0;
	vector<NeuralNetworkModule> NeuralNetworkGeneration;
	NeuralNetworkGeneration.resize(25); 
	//NeuronOutputNodeBuffer
	for (unsigned int NumOfIndividualInGeneration = 0; NumOfIndividualInGeneration < 25; NumOfIndividualInGeneration++) {
		NeuralNetworkGeneration[NumOfIndividualInGeneration].NeuralNetworkSetup.push_back(FRAMES_PER_BUFFER);//NumOfCycles
		NeuralNetworkGeneration[NumOfIndividualInGeneration].NeuralNetworkSetup.push_back(6);//NumOfInputs
		NeuralNetworkGeneration[NumOfIndividualInGeneration].NeuralNetworkSetup.push_back(1);//NumOfOutputs

		//AudioIn
		NeuralNetworkGeneration[NumOfIndividualInGeneration].NeuralNetworkSetup.push_back(FRAMES_PER_BUFFER);//NumOfCycles
		NeuralNetworkGeneration[NumOfIndividualInGeneration].NeuralNetworkSetup.push_back(0);//OutputInputToNodeID
		NeuralNetworkGeneration[NumOfIndividualInGeneration].NeuralNetworkSetup.push_back(1);//NumOfCyclesTillIncrement
		//Timein
		for (unsigned int i = 0; i < 5; i++) {
			NeuralNetworkGeneration[NumOfIndividualInGeneration].NeuralNetworkSetup.push_back(1);//NumOfCycles
			NeuralNetworkGeneration[NumOfIndividualInGeneration].NeuralNetworkSetup.push_back(i + 1);//OutputInputToNodeID
			NeuralNetworkGeneration[NumOfIndividualInGeneration].NeuralNetworkSetup.push_back(0);//NumOfCyclesTillIncrement
		}
		//AudioOut
		NeuralNetworkGeneration[NumOfIndividualInGeneration].NeuralNetworkSetup.push_back(FRAMES_PER_BUFFER);//NumOfCycles
		NeuralNetworkGeneration[NumOfIndividualInGeneration].NeuralNetworkSetup.push_back(6);//InputNodeIDValueToOutput
		NeuralNetworkGeneration[NumOfIndividualInGeneration].NeuralNetworkSetup.push_back(1);//NumOfCyclesTillIncrement

		//23

		/*
		0 = Sum ((input[] * weight[]) + bias[])
		1 = Sigmoid (Sum ((input[] * weight[]) + bias[]))
		2 = Tanh (Sum ((input[] * weight[]) + bias[]))
		3 = Multiply ((input[] * weight[]) + bias[])
		4 = Min ((input[] * weight[]) + bias[])
		5 = Max ((input[] * weight[]) + bias[])
		*/
		/*
		ScoringVariables
		[0]TrainingMode
		[1]NumOfNodesToScore
		[2 + (n * 2)]NodeIDsToScore
		[3 + (n * 2)]ScoringVariables
		*/
		NeuralNetworkGeneration[NumOfIndividualInGeneration].BufferNodeValues.resize(6);
		CreateLSTMUnit(
			NeuralNetworkGeneration[NumOfIndividualInGeneration],
			0
		);
		CreateLSTMUnit(
			NeuralNetworkGeneration[NumOfIndividualInGeneration],
			15
		);

		SaveNeuralNetworkModule(
			&NeuralNetworkGeneration[NumOfIndividualInGeneration],
			&NetworkModule,		//0=Emotinal training module, 1=Core module, 2=Offline training module
			&GenerationNum,
			&NumOfIndividualInGeneration
		);
	}
}

void LoadScoreToConsole() {
	unsigned int NetworkModule = 0;
	unsigned int GenerationNum = 40;
	vector<float> GenerationScoreVector;
	float TotalGenerationScore = 0;
	float AverageGenerationScore = 0;
	GenerationScoreVector.resize(25);
	for (GenerationNum = 0; GenerationNum <= 40; GenerationNum++) {
		std::ostringstream FilePath;
		if (NetworkModule == 0) {
			FilePath << "C:/ALICE/Emotinal training module/Gen-" << GenerationNum << "_Scores-" << ".DATA";
		}
		else if (NetworkModule == 1) {
			FilePath << "C:/ALICE/Offline training module/Gen-" << GenerationNum << "_Scores-" << ".DATA";
		}else {
			FilePath << "C:/ALICE/Core module/Gen-" << GenerationNum << "_Scores-" << ".DATA";
		}
		GenerationScoreVector.resize(25);
		LoadVector(
			&FilePath,
			GenerationScoreVector
		);
		TotalGenerationScore = 0;
		for (unsigned int i = 0; i < 25; i++) {
			//cout << GenerationScoreVector[i] << endl;
			TotalGenerationScore += GenerationScoreVector[i];
		//	cout <<"Gen: " << GenerationNum  << "		Indivdual: " << i << "		Score: " << GenerationScoreVector[i] << endl;
		}
		cout << endl;
		AverageGenerationScore = TotalGenerationScore / 25;
		cout <<"Gen: " << GenerationNum  << "		Avg: " << AverageGenerationScore << endl;
	}
}




