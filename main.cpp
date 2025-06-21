#include <iostream>
#include <memory>
#include "NeuralNetwork.h"
#include <chrono>

using std::make_shared;

static int s_AllocationCount = 0;
/*
void *operator new(size_t size)
{
    // std::cout << "Allocating " << size << " bytes" << std::endl;
    ++s_AllocationCount;
    return malloc(size);
}
*/

void second_nn(int epochs)
{
    NN::Model nn({make_shared<NN::InputLayer>(2),
                  make_shared<NN::DenseLayer>(3),
                  make_shared<NN::ActivationSigmoid>(),
                  make_shared<NN::DenseLayer>(2),
                  make_shared<NN::ActivationSigmoid>()});

    std::cout << "Number of allocations init: " << s_AllocationCount-- << std::endl;
    Tensor input(
        {{0., 0.},
         {0., 1.},
         {1., 0.},
         {1., 1.}});

    Tensor target(
        {{1., 0.},
         {0., 1.},
         {0., 1.},
         {1., 0.}});

    Tensor output;

    std::cout << "Number of allocations before: " << --s_AllocationCount << std::endl;
    nn.fit(input, target, 0.4, epochs, s_AllocationCount);

    std::cout << "Number of allocations after: " << s_AllocationCount << std::endl;
    int s = input.get_dim(0);
    for (size_t i = 0; i < s; ++i)
    {
        Tensor act_input = input.get_row(i);
        nn.forward(act_input, output, s_AllocationCount);
        output.print();
    }

    std::cout << "Number of allocations after: " << s_AllocationCount << std::endl;
}

void conv_example()
{
    using namespace NN;

    auto conv = std::make_shared<Conv2DLayer>(1, 2);
    conv->connect(1);

    conv->filters(0, 0, 0, 0) = 1.0;
    conv->filters(0, 0, 0, 1) = 0.0;
    conv->filters(0, 0, 1, 0) = 0.0;
    conv->filters(0, 0, 1, 1) = -1.0;

    conv->biases(0, 0) = 0.0;

    // Entrada: [1 canal, 3 alto, 3 ancho]
    Tensor input(1, 3, 3);
    input(0, 0, 0) = 1.0;
    input(0, 0, 1) = 2.0;
    input(0, 0, 2) = 3.0;
    input(0, 1, 0) = 4.0;
    input(0, 1, 1) = 5.0;
    input(0, 1, 2) = 6.0;
    input(0, 2, 0) = 7.0;
    input(0, 2, 1) = 8.0;
    input(0, 2, 2) = 9.0;

    Tensor output;
    conv->forward(input, output);

    std::cout << "Resultado de conv_example():\n";
    output.print(); // Esperado: [-4, -4], [-4, -4]
}

void conv_example_larger()
{
    using namespace NN;

    Tensor input(1, 28, 28); // Imagen de entrada de 28x28 y 1 canal
    input.fill(0.0);         // Llenamos con ceros, puede ser con datos aleatorios

    // Capa Conv2D #1: 32 filtros de 3x3
    Conv2DLayer conv1(32, 3);
    conv1.connect(1); // Conectamos 1 canal de entrada
    Tensor output1;
    conv1.forward(input, output1);
    std::cout << "Capa Conv2D #1 (32 filtros 3x3) - Shape: (" << output1.get_dim(0) << ", " << output1.get_dim(1) << ", " << output1.get_dim(2) << ")\n";

    // Capa MaxPooling: 2x2
    MaxPooling2D pool1(2);
    Tensor output2;
    pool1.forward(output1, output2);
    std::cout << "MaxPooling #1 (2x2) - Shape: (" << output2.get_dim(0) << ", " << output2.get_dim(1) << ", " << output2.get_dim(2) << ")\n";

    // Capa Conv2D #2: 64 filtros de 3x3
    Conv2DLayer conv2(64, 3);
    conv2.connect(32); // 32 canales de entrada
    Tensor output3;
    conv2.forward(output2, output3);
    std::cout << "Capa Conv2D #2 (64 filtros 3x3) - Shape: (" << output3.get_dim(0) << ", " << output3.get_dim(1) << ", " << output3.get_dim(2) << ")\n";

    // Capa MaxPooling: 2x2
    MaxPooling2D pool2(2);
    Tensor output4;
    pool2.forward(output3, output4);
    std::cout << "MaxPooling #2 (2x2) - Shape: (" << output4.get_dim(0) << ", " << output4.get_dim(1) << ", " << output4.get_dim(2) << ")\n";

    // Capa Conv2D #3: 64 filtros de 3x3
    Conv2DLayer conv3(64, 3);
    conv3.connect(64); // 64 canales de entrada
    Tensor output5;
    conv3.forward(output4, output5);
    std::cout << "Capa Conv2D #3 (64 filtros 3x3) - Shape: (" << output5.get_dim(0) << ", " << output5.get_dim(1) << ", " << output5.get_dim(2) << ")\n";

    // Capa Flatten
    Flatten flatten;
    Tensor output6;
    flatten.forward(output5, output6);
    std::cout << "Flatten - Shape: (" << output6.get_dim(0) << ")\n";

    // Capa Densa: 64 neuronas
    DenseLayer dense(64);
    dense.connect(576);
    Tensor output7;
    dense.forward(output6, output7);
    std::cout << "Capa Densa (64 neuronas) - Shape: (" << output7.get_dim(0) << ")\n";

    // Capa de salida: 10 neuronas
    DenseLayer dense_out(10);
    dense_out.connect(64);
    Tensor output8;
    dense_out.forward(output7, output8);
    std::cout << "Capa de salida (10 neuronas) - Shape: (" << output8.get_dim(0) << ")\n";
}

void conv_example_larger2()
{
    using namespace NN;

    Tensor input(1, 7, 7);
    input.fill(1.0);

    Conv2DLayer conv1(2, 3);
    conv1.connect(1);
    Tensor output1;
    conv1.forward(input, output1);
    std::cout << "Capa Conv2D #1 (2 filtros 3x3) - Shape: ("
              << output1.get_dim(0) << ", " << output1.get_dim(1) << ", "
              << output1.get_dim(2) << ")\n";

    MaxPooling2D pool1(2);
    Tensor output2;
    pool1.forward(output1, output2);
    std::cout << "MaxPooling #1 (2x2) - Shape: ("
              << output2.get_dim(0) << ", " << output2.get_dim(1) << ", "
              << output2.get_dim(2) << ")\n";

    Flatten flatten;
    Tensor output5;
    flatten.forward(output2, output5);
    std::cout << "Flatten - Shape: (" << output5.get_dim(0) << ")\n";

    DenseLayer dense(4);
    dense.connect(8); // Tamaño de la entrada de la capa densa (2x2 = 4)
    Tensor output6;
    dense.forward(output5, output6);
    std::cout << "Capa Densa (4 neuronas) - Shape: (" << output6.get_dim(0) << ")\n";

    // 8. Capa de salida: 2 neuronas
    DenseLayer dense_out(2);
    dense_out.connect(4); // Conectamos con 4 neuronas de entrada
    Tensor output7;
    dense_out.forward(output6, output7);
    std::cout << "Capa de salida (2 neuronas) - Shape: (" << output7.get_dim(0) << ")\n";
}

int main()
{

    srand(static_cast<unsigned int>(time(0)));

    int epochs = 1000000;

    auto start = std::chrono::high_resolution_clock::now();
    std::cout << "Number of allocations: " << s_AllocationCount << std::endl;
    std::cout << "\nSecond NN" << std::endl;
    auto end = std::chrono::high_resolution_clock::now();
    second_nn(epochs);
    double time_first = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "Time: " << time_first << std::endl;
    std::cout << "Number of allocations: " << s_AllocationCount << std::endl;
}