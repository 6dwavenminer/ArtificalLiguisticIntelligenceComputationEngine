#pragma once




#include <iostream>
#include <atomic>
#include <portaudio.h>
#include <vector>

using namespace std;

/* #define SAMPLE_RATE  (17932) // Test failure to open with this value. */
#define SAMPLE_RATE (22050)
#define FRAMES_PER_BUFFER (2205)
#define NUM_SECONDS (5)
/* #define DITHER_FLAG     (paDitherOff)  */
#define DITHER_FLAG (0)

/* Select sample format. */
#if 1
#define PA_SAMPLE_TYPE  paFloat32
#define SAMPLE_SIZE (4)
#define SAMPLE_SILENCE (0.0f)
#define PRINTF_S_FORMAT "%.8f"
#elif 0
#define PA_SAMPLE_TYPE  paInt16
#define SAMPLE_SIZE (2)
#define SAMPLE_SILENCE (0)
#define PRINTF_S_FORMAT "%d"
#elif 0
#define PA_SAMPLE_TYPE  paInt24
#define SAMPLE_SIZE (3)
#define SAMPLE_SILENCE (0)
#define PRINTF_S_FORMAT "%d"
#elif 0
#define PA_SAMPLE_TYPE  paInt8
#define SAMPLE_SIZE (1)
#define SAMPLE_SILENCE (0)
#define PRINTF_S_FORMAT "%d"
#else
#define PA_SAMPLE_TYPE  paUInt8
#define SAMPLE_SIZE (1)
#define SAMPLE_SILENCE (128)
#define PRINTF_S_FORMAT "%d"
#endif

/*******************************************************************/


int AudioIOThread(
	atomic<unsigned int> *RunProgram,
	atomic<unsigned int> *ProgramStartup,
	unsigned int TrainingMode,
	atomic<unsigned int>& AudioIOToggledUpdateFromMicFlag,
	atomic<unsigned int>& AudioIOToggledUpdateToSpeakerFlag,
	vector<float>& AudioInputBufferFromMicThread,
	vector<float> *AudioInputBufferToSpeakerThread
);
