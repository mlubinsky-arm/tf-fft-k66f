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

#define CMSIS
#ifdef CMSIS
#define ARM_MATH_CM4
#include "arm_math.h"
#include "arm_const_structs.h"
#endif


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
constexpr int kTensorArenaSize = 12 * 1024;
uint8_t tensor_arena[kTensorArenaSize];


static int  fft_samples[FFT_SIZE];

#ifdef CMSIS
static q15_t s[FFT_SIZE*2]; //has to be twice FFT size
#endif


/* tensorflow/lite/c/common.h
typedef enum {
  kTfLiteNoType = 0,
  kTfLiteFloat32 = 1,
  kTfLiteInt32 = 2,
  kTfLiteUInt8 = 3,
  kTfLiteInt64 = 4,
  kTfLiteString = 5,
  kTfLiteBool = 6,
  kTfLiteInt16 = 7,
  kTfLiteComplex64 = 8,
  kTfLiteInt8 = 9,
  kTfLiteFloat16 = 10,
  kTfLiteFloat64 = 11,
} TfLiteType;
*/
#define   M_kTfLiteNoType     0
#define   M_kTfLiteFloat32    1
#define   M_kTfLiteInt32      2
#define   M_kTfLiteUInt8      3
#define   M_kTfLiteInt64      4
#define   M_kTfLiteString     5
#define   M_kTfLiteBool       6
#define   M_kTfLiteInt16      7
#define   M_kTfLiteComplex64  8
#define   M_kTfLiteInt8       9
#define   M_kTfLiteFloat16   10
#define   M_kTfLiteFloat64   11

// MODEL  can be initialized with -D  compile flag
#if  MODEL == 1      // SoftMax Quantization float32
#define MODEL_TYPE M_kTfLiteFloat32
#elif  MODEL == 2    // Logistic   float32
#define MODEL_TYPE M_kTfLiteFloat32
#else 
#define MODEL_TYPE 0  //reserved for data collection
#endif

#if MODEL_TYPE  == M_kTfLiteInt8
    int8_t*  model_input_buffer = nullptr;
#elif MODEL_TYPE  == M_kTfLiteFloat32
    float*  model_input_buffer = nullptr;
#else
    printf("\n ERROR: model type is not defined   \n");
#endif
}  // namespace

// The name of this function is important for Arduino compatibility.
void setup() {
    int k=0;
    while(k<3){
       printf("\n setup() MODEL=%d  MODEL_TYPE =%d M_kTfLiteInt8=%d M_kTfLiteFloat32=%d", MODEL , MODEL_TYPE, M_kTfLiteInt8, M_kTfLiteFloat32);
       #if MODEL_TYPE  == M_kTfLiteInt8
           printf("\n setup() MODEL_TYPE  == kTfLiteInt8");
       #elif MODEL_TYPE  == M_kTfLiteFloat32
           printf("\n setup() MODEL_TYPE  == kTfLiteFloat32");
       #endif
       k++;
    }

  // Set up logging.

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

   static tflite::MicroMutableOpResolver<4> micro_op_resolver(error_reporter);

#if  MODEL == 2
  printf("\n adding Logistic");
  if (micro_op_resolver.AddLogistic() != kTfLiteOk) {
     printf("\n ERROR adding  Logistic");
    return;
  }

  printf("\n adding AddFullyConnected");
  if (micro_op_resolver.AddFullyConnected() != kTfLiteOk) {
    printf("\n ERROR adding AddFullyConnected");
    return;
  }
#endif

#if  MODEL == 1
  printf("\n adding Softmax");
  if (micro_op_resolver.AddSoftmax() != kTfLiteOk) {
     printf("\n ERROR adding  Softmax");
    return;
  }

  printf("\n adding AddFullyConnected");
  if (micro_op_resolver.AddFullyConnected() != kTfLiteOk) {
    printf("\n ERROR adding AddFullyConnected");
    return;
  }

  printf("\n adding AddQuantize()");
  if (micro_op_resolver.AddQuantize() != kTfLiteOk) {
     printf("\n ERROR adding  AddQuantize");
    return;
  }

  printf("\n adding AddDequantize()");
  if (micro_op_resolver.AddDequantize() != kTfLiteOk) {
     printf("\n ERROR adding  AddDequantize");
    return;
  }
#endif

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


  printf("\n BEFORE kFeatureElementCount =%d ", kFeatureElementCount); // FFT_SIZE
  printf("\n BEFORE model_input->dims->size =%d ", model_input->dims->size);
  printf("\n BEFORE model_input->dims->data[0] =%d ", model_input->dims->data[0]);
  printf("\n BEFORE model_input->dims->data[1] =%d ", model_input->dims->data[1]);
  printf("\n BEFORE model_input->type =%d ", model_input->type);
  printf("\n   MODEL_TYPE =%d ", MODEL_TYPE);

// .lite/micro/examples/micro_speech/micro_features/micro_model_settings.h

  if (model_input->type != MODEL_TYPE) {
     printf("\n  Error MODEL_TYPE =%d ...but...  model_input->type =%d", MODEL_TYPE , model_input->type);
     TF_LITE_REPORT_ERROR(error_reporter, "Bad input tensor datatype in model - expected integer");
     return;
  }


  if ( model_input->dims->data[1] != FFT_SIZE) {
    TF_LITE_REPORT_ERROR(error_reporter, "Bad input tensor size in model");
    return;
  }

  if ((model_input->dims->size != 2) || (model_input->dims->data[0] != 1)) {
    TF_LITE_REPORT_ERROR(error_reporter, "Bad input tensor dimentions in model");
    return;
  }

 // printf("\n BEFORE model_input_buffer = ...");

#if MODEL_TYPE  == M_kTfLiteInt8
   model_input_buffer = model_input->data.int8;
   printf("\n setup MODEL_TYPE  == kTfLiteInt8");
#elif MODEL_TYPE  == M_kTfLiteFloat32
  model_input_buffer = model_input->data.f;
  printf("\n setup MODEL_TYPE  == kTfLiteFloat32");
#endif

  previous_time = 0;
}

//-----------------CMSIS FFT ----------------------------
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
//--------------------------------------------------------


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
          cmsis_fft( fft_samples, FFT_SIZE);
      }
      previous_time = current_time;
  }
//-------------------------------------
// Copy feature buffer to input tensor
// ------------------------------------

  for (int i = 0; i < kFeatureElementCount; i++) {
    #if MODEL_TYPE  == M_kTfLiteInt8
       model_input_buffer[i] = s[i];
    #elif MODEL_TYPE  == M_kTfLiteFloat32
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

  #if MODEL_TYPE  == M_kTfLiteInt8
        int y_val = output->data.int8[0];
        printf("\n integer model output=%d",  y_val);
  #elif MODEL_TYPE  == M_kTfLiteFloat32
     #if MODEL == 1    //SOFTMAX
        float y_val = output->data.f[1];  //skipping element[0]
     #elif   MODEL == 2   // LOGISTIC
       float y_val = output->data.f[0];
     #endif

     printf("\n model output=%5.3f",     y_val);
     if (y_val > 0.5) {
             printf ("\n model output=%5.3f --- failing water pump ---",     y_val  );
     }
  #endif  // end of float32


}
