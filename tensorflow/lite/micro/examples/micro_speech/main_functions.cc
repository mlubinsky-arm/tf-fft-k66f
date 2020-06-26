/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/micro/examples/micro_speech/main_functions.h"

#include "tensorflow/lite/micro/examples/micro_speech/audio_provider.h"
//#include "tensorflow/lite/micro/examples/micro_speech/command_responder.h"
//#include "tensorflow/lite/micro/examples/micro_speech/feature_provider.h"
#include "tensorflow/lite/micro/examples/micro_speech/micro_features/micro_model_settings.h"
#include "tensorflow/lite/micro/examples/micro_speech/micro_features/model.h"
//#include "tensorflow/lite/micro/examples/micro_speech/recognize_commands.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

//#include "tensorflow/lite/micro/all_ops_resolver.h"

#define ARM_MATH_CM4
#include "arm_math.h"
#include "arm_const_structs.h"

#include "mbed.h"
//#include "tensor_thread.h"
//DigitalOut led(LED1);


// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;
//FeatureProvider* feature_provider = nullptr;
//RecognizeCommands* recognizer = nullptr;
int32_t previous_time = 0;

// Create an area of memory to use for input, output, and intermediate arrays.
// The size of this will depend on the model you're using, and may need to be
// determined by experimentation.
constexpr int kTensorArenaSize = 10 * 1024;
uint8_t tensor_arena[kTensorArenaSize];


static int  fft_samples[FFT_SIZE];
static q15_t s[FFT_SIZE*2]; //has to be twice FFT size

#define INT_MODEL 1
#if INT_MODEL
    int8_t*  model_input_buffer = nullptr;
#else
    float*  model_input_buffer = nullptr;
#endif
}  // namespace

// The name of this function is important for Arduino compatibility.
void setup() {
  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  //
  //static  tflite::AllOpsResolver micro_op_resolver;
  //  static tflite::ops::micro::AllOpsResolver micro_op_resolver;
  //  static tflite::micro::AllOpsResolver micro_op_resolver;
  // NOLINTNEXTLINE(runtime-global-variables)
   static tflite::MicroMutableOpResolver<4> micro_op_resolver(error_reporter);
  /*
  printf("\n adding AddDepthwiseConv2D");
  if (micro_op_resolver.AddDepthwiseConv2D() != kTfLiteOk) {
    return;
  }

  printf("\n adding AddSoftmax");
  if (micro_op_resolver.AddSoftmax() != kTfLiteOk) {
    return;
  }

  printf("\n adding AddReshape");
  if (micro_op_resolver.AddReshape() != kTfLiteOk) {
    printf("\n ERROR adding Reshape");
    return;
  }
 
  printf("\n adding Logistic");
  if (micro_op_resolver.AddLogistic() != kTfLiteOk) {
     printf("\n ERROR adding adding Logistic");
    return;
  }
*/

  printf("\n adding Softmax");
  if (micro_op_resolver.AddSoftmax() != kTfLiteOk) {
     printf("\n ERROR adding adding Softmax");
    return;
  }

  printf("\n adding AddFullyConnected");
  if (micro_op_resolver.AddFullyConnected() != kTfLiteOk) {
    printf("\n ERROR adding AddFullyConnected");
    return;
  }

  printf("\n adding AddQuantize()");
  if (micro_op_resolver.AddQuantize() != kTfLiteOk) {
     printf("\n ERROR adding adding AddQuantize");
    return;
  }

  printf("\n adding AddDequantize()");
  if (micro_op_resolver.AddDequantize() != kTfLiteOk) {
     printf("\n ERROR adding adding AddDequantize");
    return;
  }

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

printf("\n BEFORE AllocateTesnors");
  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

printf("\n BEFORE interpreter->input(0)");

  // Get information about the memory area to use for the model's input.
  model_input = interpreter->input(0);

int k=0;
while (k<5){
  printf("\n BEFORE kFeatureElementCount =%d ", kFeatureElementCount); 
  printf("\n BEFORE model_input->dims->size =%d ", model_input->dims->size);
  printf("\n BEFORE model_input->dims->data[0] =%d ", model_input->dims->data[0]);
  printf("\n BEFORE model_input->dims->data[1] =%d ", model_input->dims->data[1]);
  k++;
}
// .lite/micro/examples/micro_speech/micro_features/micro_model_settings.h
//printf("\n BEFORE kFeatureSliceCount =%d ", kFeatureSliceCount); //49
//printf("\n BEFORE kFeatureSliceSize =%d ", kFeatureSliceSize); //40

  if ((model_input->dims->size != 2) || (model_input->dims->data[0] != 1) ||
     // (model_input->dims->data[1] != (kFeatureSliceCount * kFeatureSliceSize)) ||
      model_input->dims->data[1] != FFT_SIZE ||
      // (model_input->type != kTfLiteFloat32)    TODO make if INT_MODEL  !!!!
       (model_input->type != kTfLiteInt8)
      ) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Bad input tensor parameters in model");
    return;
  }


  printf("\n BEFORE model_input_buffer = ...");

#if INT_MODEL
  model_input_buffer = model_input->data.int8;
#else
  model_input_buffer = model_input->data.f;
#endif

  previous_time = 0;
}

void cmsis_fft(int* data, int data_size)
{
  if (data_size != FFT_SIZE){
    printf ("Error data_size=%d != FFT_SIZE=%d",data_size , FFT_SIZE);
    return;
  }

  static arm_rfft_instance_q15 fft_instance;
  //static q15_t s[FFT_SIZE*2]; //has to be twice FFT size
  arm_status status = arm_rfft_init_q15(
         &fft_instance,
         FFT_SIZE, // bin count
         0, // forward FFT
         1 // output bit order is normal
  );

  if (status != 0){
    printf ("\n cannot init CMSIS FFT");
    return;
  }

  arm_rfft_q15(&fft_instance, (q15_t*)data, s);
  arm_abs_q15(s, s, FFT_SIZE*2);
}

// The name of this function is important for Arduino compatibility.
void loop() {
  // Fetch the audio
  const int32_t current_time = LatestAudioTimestamp();

  int K_MAX=FFT_SIZE / 512;
  int16_t* audio_samples = nullptr;  //4 *512 = 2048 data points
  int audio_samples_size = 0;
  int big_index=0;
  for (int k=0; k<K_MAX; k++) {
      GetAudioSamples(  // one call GetAudioSamples() gives 30ms of audio = 512 data points @16000 Hz
                   error_reporter, 
                   k * kFeatureSliceDurationMs, // TODO
                   kFeatureSliceDurationMs,
                   &audio_samples_size,
                   &audio_samples);

      for ( int j=0; j < audio_samples_size; j++){
        if (big_index >= FFT_SIZE) break;
        fft_samples[big_index] = int (audio_samples[j]);
        big_index++;
      }
      if (big_index == FFT_SIZE) {
          //printf("\n -------- before calling cmsis_fft");
          cmsis_fft( fft_samples, FFT_SIZE);
          //printf("\n -------- after calling cmsis_fft");
      }
      previous_time = current_time;
      //return;
  }

//-------------------------------------
// Copy feature buffer to input tensor
// ------------------------------------

  for (int i = 0; i < kFeatureElementCount; i++) {
    #if INT_MODEL
        model_input_buffer[i] = s[i];
    #else
        model_input_buffer[i] = float(s[i]);
    #endif
  }

  // Run the model  
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed");
    return;
  }

  // Obtain a pointer to the output tensor
  TfLiteTensor* output = interpreter->output(0);

  #if INT_MODEL
        int y_val = output->data.int8[0];
        printf("\n model output=%d",  y_val);
  #else
        float y_val = output->data.f[0];
        printf("\n model output=%5.3f",     y_val);
        if (y_val > 0.5) {
             printf ("\n model output=%5.3%f --- failing water pump ---",     y_val  );
        }
  #endif


}
