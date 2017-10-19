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
#include <iomanip> 
#include <conio.h>
#include <consoleapi.h>
#undef max

#include <iostream>
#include <string>
#include <sstream>
#include <fstream>

#include <iterator>
#include <vector>
#include <limits>

#include <chrono>
#include <mutex>
#include <future>
#include <utility>

//#define FRAMES_PER_BUFFER (2205)
#define TEST_RUNTIME_IN_SECONDS (40)

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
	atomic<unsigned int> *PauseNeuralNetwork,
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
		Sleep(1);
	}
	while ((*RunProgram == 1) && (TrainingMode == 0)) {
		while ((*PauseNeuralNetwork == 1) && (*RunProgram == 1)) {
			//Loop while paused
			Sleep(100);
		}
		string FilePathString;
		std::ostringstream FilePathOStringStream;
		FilePathOStringStream.clear();
		if (TrainingMode == 0) {
			ScoringVariables.clear();
			ScoringVariables.resize(24);
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
			ScoringVariables
			[0]TrainingMode
			[1]NumOfNodesToScore
			[2 + (n * 2)]NodeIDsToScore
			[3 + (n * 2)]ScoringVariables
			*/

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

			ScoringVariables[0] = 0;
			ScoringVariables[1] = 11;
			ScoringVariables[2] = 23 + 21;
			ScoringVariables[4] = 41 + 21;
			ScoringVariables[6] = 59 + 21;
			ScoringVariables[8] = 77 + 21;
			ScoringVariables[10] = 95 + 21;
			ScoringVariables[12] = 113 + 21;
			ScoringVariables[14] = 131 + 21;
			ScoringVariables[16] = 149 + 21;
			ScoringVariables[18] = 167 + 21;
			ScoringVariables[20] = 185 + 21;
			ScoringVariables[22] = 203 + 21;

			if (AdminPresent == true) {
				if (IsAdmin == true) {
					FilePathOStringStream << "Emotinal training module/Training data/From Admin/";
					RandomValue = ((14 - MinValue) * rand() / RAND_MAX) + MinValue;
					switch (RandomValue) {
					case 0:
						ScoringVariables[3] = 0.1;
						ScoringVariables[5] = 0.5;
						ScoringVariables[7] = -0.7;
						ScoringVariables[9] = 0.1;
						ScoringVariables[11] = -0.7;
						ScoringVariables[13] = -0.7;
						ScoringVariables[15] = -0.6;
						ScoringVariables[17] = 0;
						RandomValue = ((274 - 1) * rand() / RAND_MAX) + 1;
						FilePathOStringStream << "Admiration/" << RandomValue;
						break;
					case 1:
						ScoringVariables[3] = 0.1;
						ScoringVariables[5] = 0.35;
						ScoringVariables[7] = -0.1;
						ScoringVariables[9] = 0.6;
						ScoringVariables[11] = -0.7;
						ScoringVariables[13] = -0.7;
						ScoringVariables[15] = -0.5;
						ScoringVariables[17] = 0.5;
						RandomValue = ((71 - 1) * rand() / RAND_MAX) + 1;
						FilePathOStringStream << "Amazement/" << RandomValue;
						break;
					case 2:
						ScoringVariables[3] = 0.5;
						ScoringVariables[5] = 0.1;
						ScoringVariables[7] = -0.7;
						ScoringVariables[9] = 0;
						ScoringVariables[11] = -0.6;
						ScoringVariables[13] = -0.7;
						ScoringVariables[15] = -0.65;
						ScoringVariables[17] = -0.1;
						RandomValue = ((261 - 1) * rand() / RAND_MAX) + 1;
						FilePathOStringStream << "Ecstasy/" << RandomValue;
						break;
					case 3:
						ScoringVariables[3] = -0.1;
						ScoringVariables[5] = 0.2;
						ScoringVariables[7] = -0.1;
						ScoringVariables[9] = -0.1;
						ScoringVariables[11] = 0.5;
						ScoringVariables[13] = -0.1;
						ScoringVariables[15] = 0;
						ScoringVariables[17] = -0.5;
						RandomValue = ((185 - 1) * rand() / RAND_MAX) + 1;
						FilePathOStringStream << "Grief/" << RandomValue;
						break;
					case 4:
						ScoringVariables[3] = -0.2;
						ScoringVariables[5] = -0.5;
						ScoringVariables[7] = 0.0;
						ScoringVariables[9] = -0.6;
						ScoringVariables[11] = -0.2;
						ScoringVariables[13] = 0.5;
						ScoringVariables[15] = 0.1;
						ScoringVariables[17] = 0.2;
						RandomValue = ((290 - 1) * rand() / RAND_MAX) + 1;
						FilePathOStringStream << "Loathing/" << RandomValue;
						break;
					case 5:
						ScoringVariables[3] = 0.05;
						ScoringVariables[5] = 0.0;
						ScoringVariables[7] = -0.5;
						ScoringVariables[9] = -0.5;
						ScoringVariables[11] = -0.5;
						ScoringVariables[13] = -0.3;
						ScoringVariables[15] = -0.4;
						ScoringVariables[17] = 0.1;
						RandomValue = ((593 - 1) * rand() / RAND_MAX) + 1;
						FilePathOStringStream << "Neutral/" << RandomValue;
						break;
					case 6:
						ScoringVariables[3] = -0.5;
						ScoringVariables[5] = -0.5;
						ScoringVariables[7] = -0.1;
						ScoringVariables[9] = -0.2;
						ScoringVariables[11] = 0;
						ScoringVariables[13] = 0.2;
						ScoringVariables[15] = 0.5;
						ScoringVariables[17] = -0.5;
						RandomValue = ((136 - 1) * rand() / RAND_MAX) + 1;
						FilePathOStringStream << "Rage/" << RandomValue;
						break;
					case 7:
						ScoringVariables[3] = -0.6;
						ScoringVariables[5] = -0.6;
						ScoringVariables[7] = 0.5;
						ScoringVariables[9] = 0.2;
						ScoringVariables[11] = -0.2;
						ScoringVariables[13] = 0.2;
						ScoringVariables[15] = -0.3;
						ScoringVariables[17] = 0.4;
						RandomValue = ((6 - 1) * rand() / RAND_MAX) + 1;
						FilePathOStringStream << "Terror/" << RandomValue;
						break;
					case 8:
						ScoringVariables[3] = -0.5;
						ScoringVariables[5] = -0.3;
						ScoringVariables[7] = 0.2;
						ScoringVariables[9] = 0.1;
						ScoringVariables[11] = -0.5;
						ScoringVariables[13] = 0.0;
						ScoringVariables[15] = -0.3;
						ScoringVariables[17] = 0.5;
						RandomValue = ((25 - 1) * rand() / RAND_MAX) + 1;
						FilePathOStringStream << "Vigilance/" << RandomValue;
						break;
					case 9:
						ScoringVariables[3] = 0.9;
						ScoringVariables[5] = 0.9;
						ScoringVariables[7] = -0.9;
						ScoringVariables[9] = 0.1;
						ScoringVariables[11] = -0.8;
						ScoringVariables[13] = -0.9;
						ScoringVariables[15] = -0.8;
						ScoringVariables[17] = 0.0;
						RandomValue = ((103 - 1) * rand() / RAND_MAX) + 1;
						FilePathOStringStream << "High admiration & esctasy/" << RandomValue;
						break;
					case 10:
						ScoringVariables[3] = 0;
						ScoringVariables[5] = 0.8;
						ScoringVariables[7] = -0.6;
						ScoringVariables[9] = -0.2;
						ScoringVariables[11] = 0.9;
						ScoringVariables[13] = -0.8;
						ScoringVariables[15] = -0.7;
						ScoringVariables[17] = -0.7;
						RandomValue = ((290 - 1) * rand() / RAND_MAX) + 1;
						FilePathOStringStream << "High grief & admiration/" << RandomValue;
						break;
					case 11:
						ScoringVariables[3] = -0.05;
						ScoringVariables[5] = -0.1;
						ScoringVariables[7] = -0.5;
						ScoringVariables[9] = 0;
						ScoringVariables[11] = -0.3;
						ScoringVariables[13] = 0.1;
						ScoringVariables[15] = -0.2;
						ScoringVariables[17] = 0.12;
						RandomValue = ((82 - 1) * rand() / RAND_MAX) + 1;
						FilePathOStringStream << "Low grief & loathing & rage  & amazement/" << RandomValue;
						break;
					case 12:
						ScoringVariables[3] = 0.15;
						ScoringVariables[5] = 0.1;
						ScoringVariables[7] = -0.5;
						ScoringVariables[9] = 0;
						ScoringVariables[11] = -0.55;
						ScoringVariables[13] = -0.4;
						ScoringVariables[15] = -0.5;
						ScoringVariables[17] = 0.12;
						RandomValue = ((86 - 1) * rand() / RAND_MAX) + 1;
						FilePathOStringStream << "Low Vigilance & ecstasy & admiration & amazement/" << RandomValue;
						break;
					case 13:
						AdminLeaving = true;
						FilePathOStringStream << "Gone away/";
						RandomValue = ((4 - MinValue) * rand() / RAND_MAX) + MinValue;
						switch (RandomValue) {
						case 0:
							ScoringVariables[3] = 0.5;
							ScoringVariables[5] = 0.1;
							ScoringVariables[7] = -0.7;
							ScoringVariables[9] = 0;
							ScoringVariables[11] = -0.6;
							ScoringVariables[13] = -0.7;
							ScoringVariables[15] = -0.65;
							ScoringVariables[17] = -0.1;
							RandomValue = ((21 - 1) * rand() / RAND_MAX) + 1;
							FilePathOStringStream << "Ecstasy/" << RandomValue;
							break;
						case 1:
							ScoringVariables[3] = -0.1;
							ScoringVariables[5] = 0.2;
							ScoringVariables[7] = -0.1;
							ScoringVariables[9] = -0.1;
							ScoringVariables[11] = 0.5;
							ScoringVariables[13] = -0.1;
							ScoringVariables[15] = 0;
							ScoringVariables[17] = -0.5;
							RandomValue = ((18 - 1) * rand() / RAND_MAX) + 1;
							FilePathOStringStream << "Grief/" << RandomValue;
							break;
						case 2:
							ScoringVariables[3] = -0.2;
							ScoringVariables[5] = -0.5;
							ScoringVariables[7] = 0;
							ScoringVariables[9] = -0.6;
							ScoringVariables[11] = -0.2;
							ScoringVariables[13] = 0.5;
							ScoringVariables[15] = 0.1;
							ScoringVariables[17] = 0.2;
							RandomValue = ((9 - 1) * rand() / RAND_MAX) + 1;
							FilePathOStringStream << "Loathing/" << RandomValue;
							break;
						case 3:
							ScoringVariables[3] = 0.05;
							ScoringVariables[5] = 0;
							ScoringVariables[7] = -0.5;
							ScoringVariables[9] = -0.5;
							ScoringVariables[11] = -0.5;
							ScoringVariables[13] = -0.3;
							ScoringVariables[15] = -0.4;
							ScoringVariables[17] = 0.1;
							RandomValue = ((37 - 1) * rand() / RAND_MAX) + 1;
							FilePathOStringStream << "Neutral/" << RandomValue;
							break;
						case 4:
							ScoringVariables[3] = -0.5;
							ScoringVariables[5] = -0.3;
							ScoringVariables[7] = 0.2;
							ScoringVariables[9] = 0.1;
							ScoringVariables[11] = -0.5;
							ScoringVariables[13] = 0;
							ScoringVariables[15] = -0.3;
							ScoringVariables[17] = 0.5;
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
								ScoringVariables[3] = 0.5;
								ScoringVariables[5] = 0.1;
								ScoringVariables[7] = -0.7;
								ScoringVariables[9] = 0;
								ScoringVariables[11] = -0.6;
								ScoringVariables[13] = -0.7;
								ScoringVariables[15] = -0.65;
								ScoringVariables[17] = -0.1;
								RandomValue = ((19 - 1) * rand() / RAND_MAX) + 1;
								FilePathOStringStream << "Ecstasy/" << RandomValue;
								break;
							case 1:
								ScoringVariables[3] = -0.1;
								ScoringVariables[5] = 0.2;
								ScoringVariables[7] = -0.1;
								ScoringVariables[9] = -0.1;
								ScoringVariables[11] = 0.5;
								ScoringVariables[13] = -0.1;
								ScoringVariables[15] = 0;
								ScoringVariables[17] = -0.5;
								RandomValue = ((18 - 1) * rand() / RAND_MAX) + 1;
								FilePathOStringStream << "Grief/" << RandomValue;
								break;
							case 2:
								ScoringVariables[3] = -0.2;
								ScoringVariables[5] = -0.5;
								ScoringVariables[7] = 0;
								ScoringVariables[9] = -0.6;
								ScoringVariables[11] = -0.2;
								ScoringVariables[13] = 0.5;
								ScoringVariables[15] = 0.1;
								ScoringVariables[17] = 0.2;
								RandomValue = ((29 - 1) * rand() / RAND_MAX) + 1;
								FilePathOStringStream << "Loathing/" << RandomValue;
								break;
							case 3:
								ScoringVariables[3] = 0.05;
								ScoringVariables[5] = 0;
								ScoringVariables[7] = -0.5;
								ScoringVariables[9] = -0.5;
								ScoringVariables[11] = -0.5;
								ScoringVariables[13] = -0.3;
								ScoringVariables[15] = -0.4;
								ScoringVariables[17] = 0.1;
								RandomValue = ((32 - 1) * rand() / RAND_MAX) + 1;
								FilePathOStringStream << "Neutral/" << RandomValue;
								break;
							case 4:
								ScoringVariables[3] = -0.5;
								ScoringVariables[5] = -0.5;
								ScoringVariables[7] = -0.1;
								ScoringVariables[9] = -0.2;
								ScoringVariables[11] = 0;
								ScoringVariables[13] = 0.2;
								ScoringVariables[15] = 0.5;
								ScoringVariables[17] = -0.5;
								RandomValue = ((25 - 1) * rand() / RAND_MAX) + 1;
								FilePathOStringStream << "Rage/" << RandomValue;
								break;
							case 5:
								ScoringVariables[3] = -0.6;
								ScoringVariables[5] = -0.6;
								ScoringVariables[7] = 0.5;
								ScoringVariables[9] = 0.2;
								ScoringVariables[11] = -0.2;
								ScoringVariables[13] = 0.2;
								ScoringVariables[15] = -0.3;
								ScoringVariables[17] = 0.4;
								RandomValue = ((29 - 1) * rand() / RAND_MAX) + 1;
								FilePathOStringStream << "Terror/" << RandomValue;
								break;
							case 6:
								ScoringVariables[3] = -0.5;
								ScoringVariables[5] = -0.3;
								ScoringVariables[7] = 0.2;
								ScoringVariables[9] = 0.1;
								ScoringVariables[11] = -0.5;
								ScoringVariables[13] = 0;
								ScoringVariables[15] = -0.3;
								ScoringVariables[17] = 0.5;
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
								ScoringVariables[3] = 0.1;
								ScoringVariables[5] = 0.5;
								ScoringVariables[7] = -0.7;
								ScoringVariables[9] = 0.1;
								ScoringVariables[11] = -0.7;
								ScoringVariables[13] = -0.7;
								ScoringVariables[15] = -0.6;
								ScoringVariables[17] = 0;
								RandomValue = ((20 - 1) * rand() / RAND_MAX) + 1;
								FilePathOStringStream << "Admiration/" << RandomValue;
								break;
							case 1:
								ScoringVariables[3] = 0.5;
								ScoringVariables[5] = 0.1;
								ScoringVariables[7] = -0.7;
								ScoringVariables[9] = 0;
								ScoringVariables[11] = -0.6;
								ScoringVariables[13] = -0.7;
								ScoringVariables[15] = -0.65;
								ScoringVariables[17] = -0.1;
								RandomValue = ((27 - 1) * rand() / RAND_MAX) + 1;
								FilePathOStringStream << "Ecstasy/" << RandomValue;
								break;
							case 2:
								ScoringVariables[3] = -0.1;
								ScoringVariables[5] = 0.2;
								ScoringVariables[7] = -0.1;
								ScoringVariables[9] = -0.1;
								ScoringVariables[11] = 0.5;
								ScoringVariables[13] = -0.1;
								ScoringVariables[15] = 0;
								ScoringVariables[17] = -0.5;
								RandomValue = ((7 - 1) * rand() / RAND_MAX) + 1;
								FilePathOStringStream << "Grief/" << RandomValue;
								break;
							case 3:
								ScoringVariables[3] = -0.2;
								ScoringVariables[5] = -0.5;
								ScoringVariables[7] = 0;
								ScoringVariables[9] = -0.6;
								ScoringVariables[11] = -0.2;
								ScoringVariables[13] = 0.5;
								ScoringVariables[15] = 0.1;
								ScoringVariables[17] = 0.2;
								RandomValue = ((23 - 1) * rand() / RAND_MAX) + 1;
								FilePathOStringStream << "Loathing/" << RandomValue;
								break;
							case 4:
								ScoringVariables[3] = 0.05;
								ScoringVariables[5] = 0;
								ScoringVariables[7] = -0.5;
								ScoringVariables[9] = -0.5;
								ScoringVariables[11] = -0.5;
								ScoringVariables[13] = -0.3;
								ScoringVariables[15] = -0.4;
								ScoringVariables[17] = 0.1;
								RandomValue = ((35 - 1) * rand() / RAND_MAX) + 1;
								FilePathOStringStream << "Neutral/" << RandomValue;
								break;
							case 5:
								ScoringVariables[3] = -0.5;
								ScoringVariables[5] = -0.5;
								ScoringVariables[7] = -0.1;
								ScoringVariables[9] = -0.2;
								ScoringVariables[11] = 0;
								ScoringVariables[13] = 0.2;
								ScoringVariables[15] = 0.5;
								ScoringVariables[17] = -0.5;
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
						ScoringVariables[3] = 0.1;
						ScoringVariables[5] = 0.5;
						ScoringVariables[7] = -0.7;
						ScoringVariables[9] = 0.1;
						ScoringVariables[11] = -0.7;
						ScoringVariables[13] = -0.7;
						ScoringVariables[15] = -0.6;
						ScoringVariables[17] = 0.0;
						RandomValue = ((58 - 1) * rand() / RAND_MAX) + 1;
						FilePathOStringStream << "Admiration/" << RandomValue;
						break;
					case 1:
						ScoringVariables[3] = 0.1;
						ScoringVariables[5] = 0.35;
						ScoringVariables[7] = -0.1;
						ScoringVariables[9] = 0.6;
						ScoringVariables[11] = -0.7;
						ScoringVariables[13] = -0.7;
						ScoringVariables[15] = -0.5;
						ScoringVariables[17] = 0.5;
						RandomValue = ((43 - 1) * rand() / RAND_MAX) + 1;
						FilePathOStringStream << "Amazement/" << RandomValue;
						break;
					case 2:
						ScoringVariables[3] = 0.5;
						ScoringVariables[5] = 0.1;
						ScoringVariables[7] = -0.7;
						ScoringVariables[9] = 0;
						ScoringVariables[11] = -0.6;
						ScoringVariables[13] = -0.7;
						ScoringVariables[15] = -0.65;
						ScoringVariables[17] = -0.1;
						RandomValue = ((108 - 1) * rand() / RAND_MAX) + 1;
						FilePathOStringStream << "Ecstasy/" << RandomValue;
						break;
					case 3:
						ScoringVariables[3] = -0.1;
						ScoringVariables[5] = 0.2;
						ScoringVariables[7] = -0.1;
						ScoringVariables[9] = -0.1;
						ScoringVariables[11] = 0.5;
						ScoringVariables[13] = -0.1;
						ScoringVariables[15] = 0.0;
						ScoringVariables[17] = -0.5;
						RandomValue = ((77 - 1) * rand() / RAND_MAX) + 1;
						FilePathOStringStream << "Grief/" << RandomValue;
						break;
					case 4:
						ScoringVariables[3] = -0.2;
						ScoringVariables[5] = -0.5;
						ScoringVariables[7] = 0;
						ScoringVariables[9] = -0.6;
						ScoringVariables[11] = -0.2;
						ScoringVariables[13] = 0.5;
						ScoringVariables[15] = 0.1;
						ScoringVariables[17] = 0.2;
						RandomValue = ((63 - 1) * rand() / RAND_MAX) + 1;
						FilePathOStringStream << "Loathing/" << RandomValue;
						break;
					case 5:
						ScoringVariables[3] = 0.05;
						ScoringVariables[5] = 0.0;
						ScoringVariables[7] = -0.5;
						ScoringVariables[9] = -0.5;
						ScoringVariables[11] = -0.5;
						ScoringVariables[13] = -0.3;
						ScoringVariables[15] = -0.4;
						ScoringVariables[17] = 0.1;
						RandomValue = ((102 - 1) * rand() / RAND_MAX) + 1;
						FilePathOStringStream << "Neutral/" << RandomValue;
						break;
					case 6:
						ScoringVariables[3] = -0.5;
						ScoringVariables[5] = -0.5;
						ScoringVariables[7] = -0.1;
						ScoringVariables[9] = -0.2;
						ScoringVariables[11] = 0.0;
						ScoringVariables[13] = 0.2;
						ScoringVariables[15] = 0.5;
						ScoringVariables[17] = -0.5;
						RandomValue = ((83 - 1) * rand() / RAND_MAX) + 1;
						FilePathOStringStream << "Rage/" << RandomValue;
						break;
					case 7:
						ScoringVariables[3] = -0.6;
						ScoringVariables[5] = -0.6;
						ScoringVariables[7] = 0.5;
						ScoringVariables[9] = 0.2;
						ScoringVariables[11] = -0.2;
						ScoringVariables[13] = 0.2;
						ScoringVariables[15] = -0.3;
						ScoringVariables[17] = 0.4;
						RandomValue = ((50 - 1) * rand() / RAND_MAX) + 1;
						FilePathOStringStream << "Terror/" << RandomValue;
						break;
					case 8:
						ScoringVariables[3] = -0.5;
						ScoringVariables[5] = -0.3;
						ScoringVariables[7] = 0.2;
						ScoringVariables[9] = 0.1;
						ScoringVariables[11] = -0.5;
						ScoringVariables[13] = 0.0;
						ScoringVariables[15] = -0.3;
						ScoringVariables[17] = 0.5;
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
						ScoringVariables[3] = 0.1;
						ScoringVariables[5] = 0.5;
						ScoringVariables[7] = -0.7;
						ScoringVariables[9] = 0.1;
						ScoringVariables[11] = -0.7;
						ScoringVariables[13] = -0.7;
						ScoringVariables[15] = -0.6;
						ScoringVariables[17] = 0;
						RandomValue = ((31 - 1) * rand() / RAND_MAX) + 1;
						FilePathOStringStream << "Admiration/" << RandomValue;
						break;
					case 1:
						ScoringVariables[3] = 0.5;
						ScoringVariables[5] = 0.1;
						ScoringVariables[7] = -0.7;
						ScoringVariables[9] = 0;
						ScoringVariables[11] = -0.6;
						ScoringVariables[13] = -0.7;
						ScoringVariables[15] = -0.65;
						ScoringVariables[17] = -0.1;
						RandomValue = ((23 - 1) * rand() / RAND_MAX) + 1;
						FilePathOStringStream << "Ecstasy/" << RandomValue;
						break;
					case 2:
						ScoringVariables[3] = -0.1;
						ScoringVariables[5] = 0.2;
						ScoringVariables[7] = -0.1;
						ScoringVariables[9] = -0.1;
						ScoringVariables[11] = 0.5;
						ScoringVariables[13] = -0.1;
						ScoringVariables[15] = 0;
						ScoringVariables[17] = -0.5;
						RandomValue = ((20 - 1) * rand() / RAND_MAX) + 1;
						FilePathOStringStream << "Grief/" << RandomValue;
						break;
					case 3:
						ScoringVariables[3] = -0.2;
						ScoringVariables[5] = -0.5;
						ScoringVariables[7] = 0;
						ScoringVariables[9] = -0.6;
						ScoringVariables[11] = -0.2;
						ScoringVariables[13] = 0.5;
						ScoringVariables[15] = 0.1;
						ScoringVariables[17] = 0.2;
						RandomValue = ((22 - 1) * rand() / RAND_MAX) + 1;
						FilePathOStringStream << "Loathing/" << RandomValue;
						break;
					case 4:
						ScoringVariables[3] = 0.05;
						ScoringVariables[5] = 0;
						ScoringVariables[7] = -0.5;
						ScoringVariables[9] = -0.5;
						ScoringVariables[11] = -0.5;
						ScoringVariables[13] = -0.3;
						ScoringVariables[15] = -0.4;
						ScoringVariables[17] = 0.1;
						RandomValue = ((31 - 1) * rand() / RAND_MAX) + 1;
						FilePathOStringStream << "Neutral/" << RandomValue;
						break;
					case 5:
						ScoringVariables[3] = -0.5;
						ScoringVariables[5] = -0.5;
						ScoringVariables[7] = -0.1;
						ScoringVariables[9] = -0.2;
						ScoringVariables[11] = 0;
						ScoringVariables[13] = 0.2;
						ScoringVariables[15] = 0.5;
						ScoringVariables[17] = -0.5;
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
						ScoringVariables[3] = 0.1;
						ScoringVariables[5] = 0.5;
						ScoringVariables[7] = -0.7;
						ScoringVariables[9] = 0.1;
						ScoringVariables[11] = -0.7;
						ScoringVariables[13] = -0.7;
						ScoringVariables[15] = -0.6;
						ScoringVariables[17] = 0;
						RandomValue = ((58 - 1) * rand() / RAND_MAX) + 1;
						FilePathOStringStream << "Admiration/" << RandomValue;
						break;
					case 1:
						ScoringVariables[3] = 0.1;
						ScoringVariables[5] = 0.35;
						ScoringVariables[7] = -0.1;
						ScoringVariables[9] = 0.6;
						ScoringVariables[11] = -0.7;
						ScoringVariables[13] = -0.7;
						ScoringVariables[15] = -0.5;
						ScoringVariables[17] = 0.5;
						RandomValue = ((43 - 1) * rand() / RAND_MAX) + 1;
						FilePathOStringStream << "Amazement/" << RandomValue;
						break;
					case 2:
						ScoringVariables[3] = 0.5;
						ScoringVariables[5] = 0.1;
						ScoringVariables[7] = -0.7;
						ScoringVariables[9] = 0;
						ScoringVariables[11] = -0.6;
						ScoringVariables[13] = -0.7;
						ScoringVariables[15] = -0.65;
						ScoringVariables[17] = -0.1;
						RandomValue = ((108 - 1) * rand() / RAND_MAX) + 1;
						FilePathOStringStream << "Ecstasy/" << RandomValue;
						break;
					case 3:
						ScoringVariables[3] = -0.1;
						ScoringVariables[5] = 0.2;
						ScoringVariables[7] = -0.1;
						ScoringVariables[9] = -0.1;
						ScoringVariables[11] = 0.5;
						ScoringVariables[13] = -0.1;
						ScoringVariables[15] = 0;
						ScoringVariables[17] = -0.5;
						RandomValue = ((77 - 1) * rand() / RAND_MAX) + 1;
						FilePathOStringStream << "Grief/" << RandomValue;
						break;
					case 4:
						ScoringVariables[3] = -0.2;
						ScoringVariables[5] = -0.5;
						ScoringVariables[7] = 0;
						ScoringVariables[9] = -0.6;
						ScoringVariables[11] = -0.2;
						ScoringVariables[13] = 0.5;
						ScoringVariables[15] = 0.1;
						ScoringVariables[17] = 0.2;
						RandomValue = ((63 - 1) * rand() / RAND_MAX) + 1;
						FilePathOStringStream << "Loathing/" << RandomValue;
						break;
					case 5:
						ScoringVariables[3] = 0.05;
						ScoringVariables[5] = 0;
						ScoringVariables[7] = -0.5;
						ScoringVariables[9] = -0.5;
						ScoringVariables[11] = -0.5;
						ScoringVariables[13] = -0.3;
						ScoringVariables[15] = -0.4;
						ScoringVariables[17] = 0.1;
						RandomValue = ((102 - 1) * rand() / RAND_MAX) + 1;
						FilePathOStringStream << "Neutral/" << RandomValue;
						break;
					case 6:
						ScoringVariables[3] = -0.5;
						ScoringVariables[5] = -0.5;
						ScoringVariables[7] = -0.1;
						ScoringVariables[9] = -0.2;
						ScoringVariables[11] = 0;
						ScoringVariables[13] = 0.2;
						ScoringVariables[15] = 0.5;
						ScoringVariables[17] = -0.5;
						RandomValue = ((83 - 1) * rand() / RAND_MAX) + 1;
						FilePathOStringStream << "Rage/" << RandomValue;
						break;
					case 7:
						ScoringVariables[3] = -0.6;
						ScoringVariables[5] = -0.6;
						ScoringVariables[7] = 0.5;
						ScoringVariables[9] = 0.2;
						ScoringVariables[11] = -0.2;
						ScoringVariables[13] = 0.2;
						ScoringVariables[15] = -0.3;
						ScoringVariables[17] = 0.4;
						RandomValue = ((50 - 1) * rand() / RAND_MAX) + 1;
						FilePathOStringStream << "Terror/" << RandomValue;
						break;
					case 8:
						ScoringVariables[3] = -0.5;
						ScoringVariables[5] = -0.3;
						ScoringVariables[7] = 0.2;
						ScoringVariables[9] = 0.1;
						ScoringVariables[11] = -0.5;
						ScoringVariables[13] = 0;
						ScoringVariables[15] = -0.3;
						ScoringVariables[17] = 0.5;
						RandomValue = ((66 - 1) * rand() / RAND_MAX) + 1;
						FilePathOStringStream << "Vigilance/" << RandomValue;
						break;
					}
				}
			}
		}
		//Is admin
		if ((IsAdmin == true) || (OtherPersonPresent == false)) {
			ScoringVariables[19] = 1;
		}else {
			ScoringVariables[19] = -1;
		}
		//Admin busy
		if (AdminBusy == true) {
			ScoringVariables[21] = 1;
		}else {
			ScoringVariables[21] = -1;
		}
		//Person present
		if ((AdminPresent == true) || (OtherPersonPresent == true)) {
			ScoringVariables[23] = 1;
		}else {
			ScoringVariables[23] = -1;
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
	return 1;
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
		Sleep(1);
	}
	while (*RunProgram == 1) {
		if (*AudioIOToggledUpdateFlag != LastAudioIOToggledUpdateFlag) {
			AudioInputBufferToCoreThread = *AudioInputBufferFromAudioIOThread;
			if (AudioInputBufferToCoreThread.size() != FRAMES_PER_BUFFER) {
				AudioInputBufferToCoreThread.resize(FRAMES_PER_BUFFER);
			}
			AudioBufferToCoreUpdateFlag = 1;
			while ((*AudioBufferFromCoreAcknowledgementFlag == 0) && (*RunProgram == 1)) {
				//Wait for acknowledgement of data transfer
				Sleep(1);
			}
			AudioBufferToCoreUpdateFlag = 0;
			while ((*AudioBufferFromCoreAcknowledgementFlag == 1) && (*RunProgram == 1)) {
				//Wait for acknowledgement has reset
				Sleep(1);
			}

			LastAudioIOToggledUpdateFlag = (unsigned int)*AudioIOToggledUpdateFlag;
		}
		Sleep(1);
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
			while ((*AudioBufferFromCoreAcknowledgementFlag == 0) && (*RunProgram == 1)) {
				//Wait for acknowledgement of data transfer
				Sleep(1);
			}
			AudioBufferToCoreUpdateFlag = 0;
			while ((*AudioBufferFromCoreAcknowledgementFlag == 1) && (*RunProgram == 1)) {
				//Wait for acknowledgement has reset
				Sleep(1);
			}
			LastAudioIOToggledUpdateFlag = (unsigned int)*AudioIOToggledUpdateFlag;
		}
		Sleep(1);
	}
	return 1;
}


int ALICECoreThread_Genetic(
	atomic<unsigned int> *RunProgram,
	atomic<unsigned int> *ProgramStartup,
	atomic<unsigned int> *PauseNeuralNetwork,
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
		while ((*PauseNeuralNetwork == 1) && (*RunProgram == 1)) {
			//Loop while paused
		}
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
		NeuralNetworkCombinedModule = AICore.NeuralNetworkIndividual;

		NetworkScore += NetworkScoreBuffer;
		EmotinoalOutputsToUI.resize(11);

		//std::copy(NeuralNetworkCombinedModule.BufferNodeValues.begin() + 5, NeuralNetworkCombinedModule.BufferNodeValues.begin() + 16, EmotinoalOutputsToUI.begin());
		//std::copy(NeuralNetworkCombinedModule.BufferNodeValues.begin() + 22, NeuralNetworkCombinedModule.BufferNodeValues.begin() + 33, EmotinoalOutputsToUI.begin());
		//(i * 18) + 14
		for (unsigned int i = 0; i < 11; i++) {
			EmotinoalOutputsToUI[i] = NeuralNetworkCombinedModule.BufferNodeValues[(i * 18) + 14];
		}

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
	atomic<unsigned int> *PauseNeuralNetwork,
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
		Sleep(1);
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
		while ((*PauseNeuralNetwork == 1) && (*RunProgram == 1)) {
			//Loop while paused
			Sleep(100);
		}

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
							const float MinWeightChange = -5.0;
							const float MaxWeightChange = 5.0;
							const float RandomMinValue = 0.0;
							const float RandomMaxValue = 1.0;
							float RandomChance;
							for (unsigned int j = 0; j < BufferNeuralNetworkNewGeneration[(n * 5) + i].NeuronWeightsBuffer.size(); j++) {
								RandomChance = ((RandomMaxValue - RandomMinValue) * (float)rand() / (float)RAND_MAX) + RandomMinValue;
								if (RandomChance < 0.5) {//Use the weights from the other survivor
									BufferNeuralNetworkNewGeneration[(n * 5) + i].NeuronWeightsBuffer[j] = NeuralNetworkGeneration[FiveSurviorsList[i]].NeuronWeightsBuffer[j];
								}//else if (BufferNeuralNetworkNewGeneration[(n * 5) + i].NeuronWeightsBuffer[j] < 0.0f) {
								//	BufferNeuralNetworkNewGeneration[(n * 5) + i].NeuronWeightsBuffer[j] = 0.0f;
								//}
								RandomChance = ((RandomMaxValue - RandomMinValue) * (float)rand() / (float)RAND_MAX) + RandomMinValue;
								if (RandomChance < 0.1) {//Mutate the weight?
									BufferNeuralNetworkNewGeneration[(n * 5) + i].NeuronWeightsBuffer[j] = ((MaxWeightChange - MinWeightChange) * (float)rand() / (float)RAND_MAX) + MinWeightChange; //Mutate
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

		
		AudioBufferToAudioBufferAcknowledgementFromMicFlag = 0;
		while (AudioBufferFromAudioBufferUpdateFromMicFlag == 0) {
			//Wait for acknowledgement of data transfer
			Sleep(1);
		}
		AudioBufferToAudioBufferAcknowledgementFromMicFlag = 1;
		//Confirmed data transfer

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
		NeuralNetworkCombinedModule = AICore.NeuralNetworkIndividual;
		NetworkScore += NetworkScoreBuffer;
		EmotinoalOutputsToUI.resize(11);

		for (unsigned int i = 0; i < 11; i++) {
			//EmotinoalOutputsToUI[i] = NeuralNetworkCombinedModule.BufferNodeValues[(i * 18) + 14];
			EmotinoalOutputsToUI[i] = NeuralNetworkCombinedModule.BufferNodeValues[(i * 18) + 23 + 21];
		}

		while (*AudioBufferFromAudioBufferUpdateToSpeakerFlag == 0) {
			//Wait for update flag to begin data tranfer
			Sleep(1);
		}
		while (AudioOutputBufferToAudioBufferThread.size() > 0) {
			AudioOutputBufferToAudioBufferThread.clear();
		}
		AudioOutputBufferToAudioBufferThread.resize(AudioOutputBuffer.size());
		std::copy(AudioOutputBuffer.begin(), AudioOutputBuffer.end(), AudioOutputBufferToAudioBufferThread.begin());
		AudioBufferToAudioBufferAcknowledgementToSpeakerFlag = 1;
		while (*AudioBufferFromAudioBufferUpdateToSpeakerFlag == 1) {
			//Wait for update flag to reset
			Sleep(1);
		}
		AudioBufferToAudioBufferAcknowledgementToSpeakerFlag = 0;
	}
	return 1;
}

void HideConsolecursor()
{
	HANDLE consoleHandle = GetStdHandle(STD_OUTPUT_HANDLE);
	CONSOLE_CURSOR_INFO info;
	info.dwSize = 100;
	info.bVisible = FALSE;
	SetConsoleCursorInfo(consoleHandle, &info);
}
void ShowConsolecursor()
{
	HANDLE consoleHandle = GetStdHandle(STD_OUTPUT_HANDLE);
	CONSOLE_CURSOR_INFO info;
	info.dwSize = 100;
	info.bVisible = TRUE;
	SetConsoleCursorInfo(consoleHandle, &info);
}

int UserInterfaceOutputsThread(
	atomic<unsigned int> *RunProgram,
	atomic<unsigned int> *ProgramStartup,
	atomic<unsigned int> *PauseNeuralNetwork,
	vector<float> *EmotinoalOutputsToUI,
	unsigned int *NumOfIndividualInGeneration,
	unsigned int *GenerationNum,
	float *NetworkScore,
	vector<float> *ScoringVariables
) {
	vector<float> EmotinoalOutputs;
	unsigned int LocalNumOfIndividualInGeneration;
	unsigned int LocalGenerationNum;
	vector<float> LocalScoringVariables;
	float LocalNetworkScore;// unsigned int LocalNetworkScore;
	vector<float> LocalEmotinoalOutputs;

	const vector<string> VariableOutputName = { "Ecstasy: ","Admiration: ","Terror: ","Amazement: ","Grief: ", "Loathing: ","Rage: ","Vigilance: ", "ID: ","Busy: ","Present: ","Generation: ","Individual: ","Current score: " };
	HANDLE hStdout = GetStdHandle(STD_OUTPUT_HANDLE);
	COORD position = { 0, 0 };
	system("CLS");
	HideConsolecursor();
	cout.precision(3);
	cout << fixed << setfill(' ');

	for (unsigned int i = 0; i < 11; i++) {
		cout << VariableOutputName[i] << setw(20 - (VariableOutputName[i]).size()) << (float)0.0 << setw(10) << (float)0.0 << endl;
	}
	cout << VariableOutputName[11] << setw(16 - (VariableOutputName[11]).size()) << (unsigned int)0 << endl;
	cout << VariableOutputName[12] << setw(16 - (VariableOutputName[12]).size()) << (unsigned int)0 << endl;
	cout << VariableOutputName[13] << setw(20 - (VariableOutputName[13]).size()) << (float)0.0 << endl;

	while (*ProgramStartup == 1) {
		//Wait for startup to finish
		Sleep(1);
	}
	while (*RunProgram == 1) {
		while ((*PauseNeuralNetwork == 1)&&(*RunProgram == 1)) {
			//Loop while paused
			position = { 0, 14 };
			SetConsoleCursorPosition(hStdout, position);
			cout << "Paused - press P to unpause";
			Sleep(100);
		}
		if (*PauseNeuralNetwork == 0){
			position = { 0, 14 };
			SetConsoleCursorPosition(hStdout, position);
			cout << "                           ";
		}

		LocalNumOfIndividualInGeneration = *NumOfIndividualInGeneration;
		LocalNetworkScore = *NetworkScore;
		LocalGenerationNum = *GenerationNum;
		LocalScoringVariables = *ScoringVariables;
		EmotinoalOutputs.resize((*EmotinoalOutputsToUI).size());
		LocalEmotinoalOutputs.resize(LocalScoringVariables.size());
		std::copy((*EmotinoalOutputsToUI).begin(),(*EmotinoalOutputsToUI).end(), EmotinoalOutputs.begin());
		//std::copy((*ScoringVariables).begin(), (*ScoringVariables).end(), LocalEmotinoalOutputs.begin());
		//for (unsigned int i = 0; i < (*ScoringVariables).size(); i++) {

		if (LocalScoringVariables.size() != 0) {
			for (unsigned int i = 0; i < LocalScoringVariables[1]; i++) {//Same issue as below.
				LocalEmotinoalOutputs[i] = LocalScoringVariables[3 + (i * 2)];//Error occured here!!! LocalEmotinoalOutputs is size 0, when it should be at 3. ScoringVariables must have resized to 0 momentarily. -fixed, maybe
			}
		}


		if ((EmotinoalOutputs.size() == 11) && (LocalEmotinoalOutputs.size() != 0)){
					
			for (short i = 0; i < 11; i++) {
				position = { 15, i };
				SetConsoleCursorPosition(hStdout, position);
				if (EmotinoalOutputs[i] >= 0) {
					cout << " ";//Placed for positive values where '-' would be used for negative values
				}
				cout << EmotinoalOutputs[i];
				position = { 25, i };
				SetConsoleCursorPosition(hStdout, position);

				if (LocalEmotinoalOutputs[i] >= 0) {
					cout << " ";//Placed for positive values where '-' would be used for negative values
				}
				cout << LocalEmotinoalOutputs[i];
				
			}
			position = { 15, 11 };
			SetConsoleCursorPosition(hStdout, position);
			cout << LocalGenerationNum << "            ";
			position = { 15, 12 };
			SetConsoleCursorPosition(hStdout, position);
			cout << LocalNumOfIndividualInGeneration << "            ";
			position = { 15, 13 };
			SetConsoleCursorPosition(hStdout, position);
			cout << LocalNetworkScore << "            ";
		}
		Sleep(50);
		//if (GetAsyncKeyState(50) == 1) {
		//if (getch() == 50) {

	}
	return 1;
}

int NeuralNetworkRunTimeUserInputThread(
	atomic<unsigned int> *RunProgram,
	atomic<unsigned int> *ProgramStartup,
	atomic<unsigned int> *PauseNeuralNetwork
) {
	int UserInputFromKeyboard = 0;
	while (*ProgramStartup == 1) {
		//Wait for startup to finish
	}
	while (*RunProgram == 1) {
		UserInputFromKeyboard = 0;
		switch (UserInputFromKeyboard = getch()) {

		case 0x1B://esc
			*RunProgram = 0;
			break;
		case 0x50://P
		case 0x70://p
			*PauseNeuralNetwork ^= 1;
			break;
		default:
			break;
		}
	}
	return 1;
}

void main() {
	atomic<unsigned int> RunProgram = 0;
	atomic<unsigned int> ProgramStartup = 1;
	atomic<unsigned int> PauseNeuralNetwork = 0;

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

	string UserInputString = "";
	while (1) {
		cout << "Please input starting generation number:";
		getline(cin, UserInputString);
		stringstream UserInputStringStream(UserInputString);
		if ((UserInputStringStream >> GenerationNum) && (GenerationNum >= 0)) {// if a number and not negative
			break;
		}
		cout << "Invalid number, please try again" << endl;
	}


	AudioBufferFromCoreAcknowledgementToSpeakerFlag = 0; //TMP

	AudioInputBufferToSpeaker.resize(FRAMES_PER_BUFFER);
	//if (TrainingMode != 0) {
	auto NeuralNetworkRunTimeUserInputThreadID = async(
		NeuralNetworkRunTimeUserInputThread,
		&RunProgram,
		&ProgramStartup,
		&PauseNeuralNetwork
	);

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
		&PauseNeuralNetwork,
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
		&PauseNeuralNetwork,
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
		&PauseNeuralNetwork,
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
		Sleep(100);
	}
}

void CreateNeuralNode(
	NeuralNetworkModule &NeuralNetworkIndividual,
	unsigned int TypeOfNode,
	vector<unsigned int> InputNodeIDs
) {
	const float MinWeight = -10.0;
	const float MaxWeight = 10.0;
	float RandomChance;
	unsigned int StartID = NeuralNetworkIndividual.BufferNodeValues.size();
	unsigned int NumberOfInputs = InputNodeIDs.size();
	if (NumberOfInputs == 0) {
		return;
	}

	NeuralNetworkIndividual.BufferNodeValues.resize(1 + NeuralNetworkIndividual.BufferNodeValues.size());
	//TypeOfNode
	NeuralNetworkIndividual.TypeOfNode.push_back(TypeOfNode);

	//NeuronNodeStartBuffer
	unsigned int WeightLength = 0;
	unsigned int CurrentWeightLength = 0;
	unsigned int NeuronNodeStartBufferSize = NeuralNetworkIndividual.NeuronNodeStartBuffer.size();
	if (NeuronNodeStartBufferSize > 0) {
		CurrentWeightLength = NeuralNetworkIndividual.NeuronNodeStartBuffer[NeuronNodeStartBufferSize - 1] + NeuralNetworkIndividual.NeuronNodeLength[NeuronNodeStartBufferSize - 1];//InputNodeID;
		//NeuralNetworkIndividual.NeuronNodeStartBuffer.push_back(NeuralNetworkIndividual.NeuronNodeStartBuffer[NeuronNodeStartBufferSize - 1]);//InputNodeID;
	}
	NeuralNetworkIndividual.NeuronNodeStartBuffer.push_back(CurrentWeightLength);//InputNodeID;

	//NeuronNodeLength
	NeuralNetworkIndividual.NeuronNodeLength.push_back(NumberOfInputs);


	//NeuronInputNodeIDsBuffer
	for (unsigned int i = 0; i < NumberOfInputs; i++) {
		NeuralNetworkIndividual.NeuronInputNodeIDsBuffer.push_back(InputNodeIDs[i]);
	}

	//NeuronOutputNodeBuffer
	//NeuronOutputNodeIDsBuffer
	NeuralNetworkIndividual.NeuronOutputNodeIDsBuffer.push_back(StartID);

	//NeuronOutputNodeBuffer
	NeuralNetworkIndividual.NeuronOutputNodeBuffer.resize(NeuralNetworkIndividual.NeuronOutputNodeIDsBuffer.size());
	//NeuralNetworkIndividual.NeuronOutputNodeBuffer.resize(9 + NeuralNetworkIndividual.NeuronOutputNodeIDsBuffer.size());

	//NeuronWeightsBuffer / NeuronBiasBuffer
	for (unsigned int i = 0; i < NumberOfInputs; i++) {
		RandomChance = ((MaxWeight - MinWeight) * (float)rand() / (float)RAND_MAX) + MinWeight;
		NeuralNetworkIndividual.NeuronWeightsBuffer.push_back(RandomChance);

		RandomChance = ((MaxWeight - MinWeight) * (float)rand() / (float)RAND_MAX) + MinWeight;
		NeuralNetworkIndividual.NeuronBiasBuffer.push_back(RandomChance);
	}
}

/*
unsigned int InputNodeID; Node data is pulled from
Uses 9 nodes
*/
void CreateLSTMUnit(
	NeuralNetworkModule &NeuralNetworkIndividual,
	unsigned int InputNodeID
) {
	unsigned int StartID = NeuralNetworkIndividual.BufferNodeValues.size();
	const vector<unsigned int> NodeTypeVector = { 1,1,2,1,3,3,0,2,3 };
	const vector<vector<unsigned int>> InputNodeIDs = { 
		{ InputNodeID, StartID + 8 },
		{ InputNodeID, StartID + 8 },
		{ InputNodeID, StartID + 8 },
		{ InputNodeID, StartID + 8 },
		{ StartID + 0, StartID + 6 },
		{ StartID + 1, StartID + 2 },
		{ StartID + 4, StartID + 5 },
		{  StartID + 6 },
		{ StartID + 3, StartID + 7 }
	};
	for (unsigned int i = 0; i < 9; i++) {
		CreateNeuralNode(
			NeuralNetworkIndividual,
			(unsigned int)NodeTypeVector[i],
			vector<unsigned int>{ InputNodeIDs[i] }
		);
	}
	/*


	const float MinWeight = -10.0;
	const float MaxWeight = 10.0;
	float RandomChance;
	unsigned int StartID = NeuralNetworkIndividual.BufferNodeValues.size();

	NeuralNetworkIndividual.BufferNodeValues.resize(9 + NeuralNetworkIndividual.BufferNodeValues.size());
	//TypeOfNode
	NeuralNetworkIndividual.TypeOfNode.push_back(1);//Sigmoid
	NeuralNetworkIndividual.TypeOfNode.push_back(1);//Sigmoid
	NeuralNetworkIndividual.TypeOfNode.push_back(2);//Tanh
	NeuralNetworkIndividual.TypeOfNode.push_back(1);//Sigmoid
	NeuralNetworkIndividual.TypeOfNode.push_back(3);//Multiply
	NeuralNetworkIndividual.TypeOfNode.push_back(3);//Multiply
	NeuralNetworkIndividual.TypeOfNode.push_back(0);//Sum
	NeuralNetworkIndividual.TypeOfNode.push_back(2);//Tanh
	NeuralNetworkIndividual.TypeOfNode.push_back(3);//Multiply

	//NeuronNodeStartBuffer / NeuronNodeLength
	unsigned int WeightLength = 0;
	unsigned int CurrentWeightLength = 0;//InputNodeID;
	unsigned int NeuronNodeStartBufferSize = NeuralNetworkIndividual.NeuronNodeStartBuffer.size();
	if (NeuronNodeStartBufferSize > 0) {
		CurrentWeightLength = NeuralNetworkIndividual.NeuronNodeStartBuffer[NeuronNodeStartBufferSize - 1];
	} 

	for (unsigned int i = 0; i < 8; i++) {
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
	*/
}

//GenerateInitialGeneration
void main6() {
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
		NeuralNetworkGeneration[NumOfIndividualInGeneration].BufferNodeValues.resize(6);//Network input nodes
		
		CreateNeuralNode(//6
			NeuralNetworkGeneration[NumOfIndividualInGeneration],
			(unsigned int) 3,
			vector<unsigned int>{ 7 }
		);
		CreateNeuralNode(//7
			NeuralNetworkGeneration[NumOfIndividualInGeneration],
			(unsigned int) 3,
			vector<unsigned int>{ 6 }
		);
		CreateNeuralNode(//8
			NeuralNetworkGeneration[NumOfIndividualInGeneration],
			(unsigned int) 1,
			vector<unsigned int>{ 1,2,3,4,5,6 }
		);
		CreateNeuralNode(//9
			NeuralNetworkGeneration[NumOfIndividualInGeneration],
			(unsigned int)1,
			vector<unsigned int>{ 10 }
		);
		CreateNeuralNode(//10
			NeuralNetworkGeneration[NumOfIndividualInGeneration],
			(unsigned int)1,
			vector<unsigned int>{ 0,9,11 }
		);
		CreateNeuralNode(//11
			NeuralNetworkGeneration[NumOfIndividualInGeneration],
			(unsigned int)1,
			vector<unsigned int>{ 8,10,12 }
		);
		CreateNeuralNode(//12
			NeuralNetworkGeneration[NumOfIndividualInGeneration],
			(unsigned int)1,
			vector<unsigned int>{ 11 }
		);
		CreateNeuralNode(//13
			NeuralNetworkGeneration[NumOfIndividualInGeneration],
			(unsigned int)1,
			vector<unsigned int>{ 9,10,11,12 }
		);
		CreateNeuralNode(//14
			NeuralNetworkGeneration[NumOfIndividualInGeneration],
			(unsigned int)4,
			vector<unsigned int>{ 9,10,11,12 }
		);
		CreateNeuralNode(//15
			NeuralNetworkGeneration[NumOfIndividualInGeneration],
			(unsigned int)5,
			vector<unsigned int>{ 9, 10, 11, 12 }
		);
		for (unsigned int i = 0; i < 11; i++) {
			CreateNeuralNode(//16-26
				NeuralNetworkGeneration[NumOfIndividualInGeneration],
				(unsigned int)1,
				vector<unsigned int>{ 13,14,15 }
			);
		}

		for (unsigned int i = 0; i < 11; i++) {
			CreateLSTMUnit(
				NeuralNetworkGeneration[NumOfIndividualInGeneration],
				16 + i//0
			);
			CreateLSTMUnit(
				NeuralNetworkGeneration[NumOfIndividualInGeneration],
				(i * 18) + 35//5+9+21
			);
		}


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
	for (GenerationNum = 0; GenerationNum <= 17; GenerationNum++) {
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


void main2() {
	LoadScoreToConsole();
}

