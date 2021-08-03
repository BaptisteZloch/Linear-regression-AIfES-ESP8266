#include <Arduino.h>
#include <aifes.h>
#include <stdlib.h> //to use abs()

aimodel_t model; // AIfES model
ailayer_t *x;    // Layer object from AIfES, contains the layers

ailoss_mse_f32_t mse_loss;  // Object for the loss
aiopti_t *optimizer; // Object for the optimizer

//2x+1
float input_data[] = {-3.0f, 0.0f, 1.0f, 3.0f, 5.0f};
float target_data[] = {-5.0f, 1.0f, 3.0f, 7.0f, 11.0f};

void *parameter_memory = NULL; // Pointer to the memory stack of the AIfES model

uint16_t input_layer_shape[] = {1, 1};
ailayer_input_t input_layer; // Definition of the AIfES input layer

ailayer_dense_t hidden_layer_1; // Definition of the dense hidden layer which is also our output layer

void build_AIfES_model()
{
  // The model is being build in this function. First of all, the layers (input, hidden layer) need to be initialized.
  // Then the storage for the parameters of the network is reserved and distributed to the layers.

  // ---------------------------------- Layer definition ---------------------------------------
  input_layer.input_dim = 2;                   // Definition of the input dimension, here a 2 dimensional input layer is selected
  input_layer.input_shape = input_layer_shape; // Definition of the input shape, here a (1,1) layout is necessary, because each sample consists of 1 value

  hidden_layer_1.neurons = 1;

  model.input_layer = ailayer_input_f32_default(&input_layer);
  x = ailayer_dense_f32_default(&hidden_layer_1, model.input_layer);
  model.output_layer = x;
  model.loss = ailoss_mse_f32_default(&mse_loss, model.output_layer);

  aialgo_compile_model(&model);

  uint32_t parameter_memory_size = aialgo_sizeof_parameter_memory(&model);

  // Output of the calculated memory size
  Serial.print(F("Required memory for parameter (Weights, Bias, ...):"));
  Serial.print(parameter_memory_size);
  Serial.println(F("Byte"));

  // Reserve the necessary memory stack
  parameter_memory = malloc(parameter_memory_size);

  // Check if malloc was successful
  if (parameter_memory == NULL)
  {
    while (true)
    {
      Serial.println(F("Not enough RAM available. Reduce the number of samples per object and flash the example again."));
      delay(1000);
    }
  }

  // Assign the memory for the trainable parameters (like weights, bias, ...) of the model.
  aialgo_distribute_parameter_memory(&model, parameter_memory, parameter_memory_size);
}

void train_AIfES_model()
{
  // In this function the model is trained with the training data

  uint32_t i; // Counting variable

  // -------------------------------- Create tensors needed for training ---------------------
  // Create the input tensor for training, contains all samples
  uint16_t input_shape[] = {5, 1};
  aitensor_t input_tensor;
  input_tensor.dtype = aif32;
  input_tensor.dim = 2;
  input_tensor.shape = input_shape;
  input_tensor.data = input_data;

  // Create the target tensor for training, contains the desired output for the corresponding sample to train the ANN
  uint16_t target_shape[] = {5, 1};
  aitensor_t target_tensor;
  target_tensor.dtype = aif32;
  target_tensor.dim = 2;
  target_tensor.shape = target_shape;
  target_tensor.data = target_data;

  // Create an output tensor for training, here the results of the ANN are saved and compared to the target tensor during training
  float output_data[5][1];            // Array for storage of the output data
  uint16_t output_shape[] = {5, 1};   // Definition of the shape of the tensor
  aitensor_t output_tensor;           // Creation of the target AIfES tensor
  output_tensor.dtype = aif32;        // Definition of the used data type, here float with 32 bits, different ones are available
  output_tensor.dim = 2;              // Dimensions of the tensor, here 2 dimensions, as specified at output_shape
  output_tensor.shape = output_shape; // Set the shape of the output tensor
  output_tensor.data = output_data;   // Assign the output_data array to the tensor. It expects a pointer to the array where the data is stored

  // -------------------------------- Initialize the weights and bias of the layers ---------------------
  // Here the weights and biases of the hidden and output layer are randomly initialized from a uniform distribution within the given range
  float from = -2.0;
  float to = 2.0;
  aimath_f32_default_tensor_init_uniform(&hidden_layer_1.weights, from, to);
  aimath_f32_default_tensor_init_uniform(&hidden_layer_1.bias, from, to);

  // -------------------------------- Define the optimizer for training ---------------------
  aiopti_t *optimizer;

  aiopti_sgd_f32_t sgd_opti;
  sgd_opti.learning_rate = 0.001f;
  sgd_opti.momentum = 0.0f;

  optimizer = aiopti_sgd_f32_default(&sgd_opti);

  // -------------------------------- Allocate and schedule the working memory for training ---------
  uint32_t memory_size = aialgo_sizeof_training_memory(&model, optimizer);

  // Output of the calculated memory size
  Serial.print(F("Required memory for the training (Intermediate results, gradients, optimization memory):"));
  Serial.print(memory_size);
  Serial.print(F("Byte"));
  Serial.println(F(""));

  // Reserve the necessary memory stack
  void *memory_ptr = malloc(memory_size);

  // Check if malloc was successful
  if (memory_ptr == NULL)
  {
    while (true)
    {
      Serial.println(F("Not enough RAM available. Reduce the number of samples per Object and flash the example again."));
      delay(1000);
    }
  }

  // Schedule the memory over the model
  aialgo_schedule_training_memory(&model, optimizer, memory_ptr, memory_size);

  // Initialize model for training
  aialgo_init_model_for_training(&model, optimizer);

  // ------------------------------------- Run the training ------------------------------------
  float loss;
  uint32_t batch_size = 5;
  uint16_t epochs = 600;
  uint16_t print_interval = 10;

  Serial.println(F("Start training"));
  for (i = 0; i < epochs; i++)
  {
    // One epoch of training. Iterates through the whole data once
    aialgo_train_model(&model, &input_tensor, &target_tensor, optimizer, batch_size);

    // Calculate and print loss every print_interval epochs
    if (i % print_interval == 0)
    {
      // Calculate loss
      aialgo_calc_loss_model_f32(&model, &input_tensor, &target_tensor, &loss);
      // Output current epoch and loss
      Serial.print(F("Epoch: "));
      Serial.print(i);
      Serial.print(F(" Loss: "));
      Serial.print(loss);
      Serial.println(F(""));
    }
  }

  Serial.println(F("Finished training"));

  // ----------------------------------------- Evaluate the trained model --------------------------
  aialgo_inference_model(&model, &input_tensor, &output_tensor);

  // Print the original labels and the predicted results
  Serial.println(F("Outputs:"));
  Serial.println(F("\tReal\tCalculated"));
  for (i = 0; i < 5; i++)
  {
    Serial.print(F("\t"));
    Serial.print(target_data[i]);
    Serial.print(F("\t"));
    Serial.println(output_data[i][0]);
  }
}

float inference_AIfES_model(float x)
{
  float input_data[1];              // Array for storage of the RGB input data
  uint16_t input_shape[] = {1, 1};  // Definition of the shape of the tensor
  aitensor_t input_tensor;          // Creation of the input AIfES tensor
  input_tensor.dtype = aif32;       // Definition of the used data type, here float with 32 bits, different ones are available
  input_tensor.dim = 2;             // Dimensions of the tensor, here 2 dimensions, as specified at input_shape
  input_tensor.shape = input_shape; // Set the shape of the input_tensor
  input_tensor.data = input_data;   // Assign the input_data array to the tensor. It expects a pointer to the array where the data is stored

  // Tensor for the output with 3 classes
  // Output values of the ANN are saved here
  float output_data[1];               // Array for storage of the output data, for each object/class one output is created
  uint16_t output_shape[] = {1, 1};   // Definition of the shape of the tensor
  aitensor_t output_tensor;           // Creation of the output AIfES tensor
  output_tensor.dtype = aif32;        // Definition of the used data type, here float with 32 bits, different ones are available
  output_tensor.dim = 2;              // Dimensions of the tensor, here 2 dimensions, as specified at output_shape
  output_tensor.shape = output_shape; // Set the shape of the input_tensor
  output_tensor.data = output_data;   // Assign the output_data array to the tensor. It expects a pointer to the array where the data is stored
  input_data[0] = x;

  aialgo_inference_model(&model, &input_tensor, &output_tensor); //make inference with our model and input data

  return output_data[0];
}

float f(float x)
{
  return 2 * x + 1;
}

float computeDifference(float theorical, float calculated)
{
  return 100 * abs((theorical - calculated) / theorical);
}

void setup()
{
  Serial.begin(115200); //115200 baud rate (If necessary, change in the serial monitor)
  while (!Serial)
    ;
  build_AIfES_model();
  train_AIfES_model();
  Serial.println(F("\tReal\tCalculated\tDifference"));
}

void loop()
{
  //generate a random float and assign it to the input of our tensor
  float _x = -100 + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (100 + 100)));
  float real_output = f(_x);
  float calculated_output = inference_AIfES_model(_x);

  Serial.print(F("\t"));
  Serial.print(real_output);
  Serial.print(F("\t"));
  Serial.print(calculated_output);
  Serial.print(F("\t\t"));
  Serial.print(computeDifference(real_output, calculated_output));
  Serial.println(" %");
  delay(700);
}
