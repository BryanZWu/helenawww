//--------------------------------------------------------------------------
// TA_Utilities.cpp
// Allow a shared computer to run smoothly when it is being used
// by students in a CUDA GPU programming course.
//
// TA_Utilities.cpp/hpp provide functions that programmatically limit
// the execution time of the function and select the GPU with the 
// lowest temperature to use for kernel calls.
//
// Updated for Windows compatibility.
// Author: Jordan Bonilla - 4/6/16
// Windows Update: [Your Name] - [Today's Date]
//--------------------------------------------------------------------------

#include "ta_utilities.hpp"

#include <windows.h> // Windows API for process/thread management
#include <cstdio>    // printf
#include <cstdlib>   // _popen, _pclose, atoi
#include <cuda_runtime.h> // cudaGetDeviceCount, cudaSetDevice

namespace TA_Utilities
{
  /* Select the least utilized GPU on this system. Estimate
     GPU utilization using GPU temperature. */
  void select_coldest_GPU() 
  {
      // Get the number of GPUs on this machine
      int num_devices;
      cudaGetDeviceCount(&num_devices);
      if(num_devices == 0) {
          printf("select_coldest_GPU: Error - No GPU detected\n");
          return;
      }
      // Read GPU info into buffer "output"
      const unsigned int MAX_BYTES = 10000;
      char output[MAX_BYTES];
      FILE *fp = _popen("nvidia-smi", "r");
      if (!fp) {
          printf("Error - Unable to execute 'nvidia-smi'\n");
          return;
      }
      size_t bytes_read = fread(output, sizeof(char), MAX_BYTES, fp);
      _pclose(fp);
      if(bytes_read == 0) {
          printf("Error - No temperature data could be read\n");
          return;
      }
      // Array to hold GPU temperatures
      int *temperatures = new int[num_devices];
      // Parse output for temperatures using knowledge of "nvidia-smi" output format
      int i = 0;
      unsigned int num_temps_parsed = 0;
      while(output[i] != '\0') {
          if(output[i] == '%') {
              unsigned int temp_begin = i + 1;
              while(output[i] != 'C') {
                  ++i;
              }
              unsigned int temp_end = i;
              char this_temperature[32];
              // Read in the characters corresponding to this temperature
              for(unsigned int j = 0; j < temp_end - temp_begin; ++j) {
                  this_temperature[j] = output[temp_begin + j];
              }
              this_temperature[temp_end - temp_begin] = '\0';
              // Convert string representation to int
              temperatures[num_temps_parsed] = atoi(this_temperature);
              num_temps_parsed++;
          }
          ++i;
      }
      // Get GPU with lowest temperature
      int min_temp = 1e7, index_of_min = -1;
      for (int i = 0; i < num_devices; i++) 
      {
          int candidate_min = temperatures[i];
          if(candidate_min < min_temp) 
          {
              min_temp = candidate_min;
              index_of_min = i;
          }
      }
      // Tell CUDA to use the GPU with the lowest temperature
      printf("Index of the GPU with the lowest temperature: %d (%d C)\n", 
          index_of_min, min_temp);
      cudaSetDevice(index_of_min);
      // Free memory and return
      delete[] temperatures;
      return;
  }

  /* Create a thread that will terminate the current process after the
     specified time limit has been exceeded */
  DWORD WINAPI TimeLimitThread(LPVOID param) {
      int time_limit = *(int*)param;
      printf("Time limit for this program set to %d seconds\n", time_limit);
      Sleep(time_limit * 1000);
      HANDLE hProcess = GetCurrentProcess();
      TerminateProcess(hProcess, 1);
      return 0;
  }

  void enforce_time_limit(int time_limit) {
      HANDLE hThread = CreateThread(
          NULL,                // Default security attributes
          0,                   // Default stack size
          TimeLimitThread,     // Thread function
          &time_limit,         // Parameter to thread function
          0,                   // Default creation flags
          NULL                 // No thread ID
      );
      if (hThread == NULL) {
          printf("Error - Unable to create thread for time limit enforcement\n");
      } else {
          CloseHandle(hThread); // Close the thread handle
      }
  }

} // end "namespace TA_Utilities"
