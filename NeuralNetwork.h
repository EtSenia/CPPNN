#include "Tensor.h"
#include <vector>
#include <cmath>
#include <memory>
#include "Activation.h"

using std::shared_ptr;

namespace NN
{

    double sigmoid(double x) { return 1 / (1 + exp(-x)); }
    double sigmoid_derivative(double x) { return x * (1 - x); }
    struct Layer
    {
        int output_size;
        virtual void forward(const Tensor &input, Tensor &output) = 0;
        virtual void backward(Tensor &delta, double lr) = 0;
        virtual void update_weights(float lr) = 0;
        virtual void connect(int in) = 0;
    };

    struct InputLayer : Layer
    {
        InputLayer(int out)
        {
            output_size = out;
        }

        void connect(int in) override
        {
            return;
        }
        void forward(const Tensor &input, Tensor &output) override
        {
            output = input;
        }
        void backward(Tensor &delta, double lr) override
        {
            return;
        }
        void update_weights(float lr) override
        {
            return;
        }
    };
    struct ActivationSigmoid : Layer
    {
        Tensor output_cache;
        void forward(const Tensor &input, Tensor &output) override
        {
            const int rows = input.get_dim(0);

            for (int i = 0; i < rows; ++i)
                output(i) = sigmoid(input(i));
            output_cache = output;
        }
        void backward(Tensor &delta, double lr) override
        {
            const int rows = delta.get_dim(0);

            for (int i = 0; i < rows; ++i)
                delta(i) *= sigmoid_derivative(output_cache(i));
        }
        void update_weights(float lr) override
        {
            return;
        }
        void connect(int in) override
        {
            output_size = in;
        }
    };

    struct DenseLayer : Layer
    {
        int input_size;
        int num;
        Tensor weights;
        Tensor biases;
        Tensor input_cache;

        DenseLayer(int output_size, int num = 5)
        {
            this->output_size = output_size;
            this->num = num;
        }

        void connect(int in) override
        {
            input_size = in;
            weights = Tensor(input_size, output_size);
            biases = Tensor(output_size);
            biases.fill(0.0f);
            generate_weights();
        }

        void generate_weights()
        {
            double limit = std::sqrt(6.0 / (input_size + output_size));
            for (int i = 0; i < input_size; ++i)
                for (int j = 0; j < output_size; ++j)
                    weights(i, j) = ((double)rand() / RAND_MAX) * 2 * limit - limit;
        }

        void forward(const Tensor &input, Tensor &output) override
        {
            input_cache = input;
            output = input.dot(weights);
            output += biases;
        }

        void backward(Tensor &delta, double lr) override
        {
            Tensor temp = delta * lr;

            delta = delta.dot(weights.transpose_view(0, 1));

            biases += temp;
            weights += input_cache.outer(temp);
        }

        void update_weights(float lr) override
        {
            // biases.swap(biases + backward_cache * lr);
            // weights.swap(weights + backward_cache.outer(input_cache) * lr);
        }
    };
    struct Conv2DLayer : Layer
    {
        int in_channels;
        int out_channels;
        int kernel_size;
        int stride;
        int padding;

        Tensor filters; // [out_channels, in_channels, kernel_size, kernel_size]
        Tensor biases;  // [out_channels, 1]

        Tensor input_cache;
        Tensor output_cache;
        Tensor d_filters;
        Tensor d_biases;

        Conv2DLayer(int out_ch, int kernel_size, int stride = 1, int padding = 0)
            : out_channels(out_ch), kernel_size(kernel_size), stride(stride), padding(padding) {}

        void connect(int in) override
        {
            in_channels = in;
            filters = Tensor(out_channels, in_channels, kernel_size, kernel_size);
            biases = Tensor(out_channels, 1);
            filters.fill(0.01); // pequeña inicialización
            biases.fill(0.0);
            output_size = out_channels; // para compatibilidad
        }

        void forward(const Tensor &input, Tensor &output) override
        {
            input_cache = input;
            int C = input.get_dim(0);
            int H = input.get_dim(1);
            int W = input.get_dim(2);

            int H_out = (H + 2 * padding - kernel_size) / stride + 1;
            int W_out = (W + 2 * padding - kernel_size) / stride + 1;

            if (H_out < 1 || W_out < 1)
            {
                throw std::runtime_error("Invalid input size for Conv2D layer");
            }

            output = Tensor(out_channels, H_out, W_out);
            output_cache = output;

            for (int oc = 0; oc < out_channels; ++oc)
            {
                for (int i = 0; i < H_out; ++i)
                {
                    for (int j = 0; j < W_out; ++j)
                    {
                        double sum = 0.0;
                        for (int ic = 0; ic < in_channels; ++ic)
                        {
                            for (int ki = 0; ki < kernel_size; ++ki)
                            {
                                for (int kj = 0; kj < kernel_size; ++kj)
                                {
                                    int xi = i * stride + ki - padding;
                                    int xj = j * stride + kj - padding;
                                    if (xi >= 0 && xj >= 0 && xi < H && xj < W)
                                        sum += input(ic, xi, xj) * filters(oc, ic, ki, kj);
                                }
                            }
                        }
                        output(oc, i, j) = sum + biases(oc, 0);
                    }
                }
            }
        }

        void backward(Tensor &delta, double lr) override
        {
            int H = input_cache.get_dim(1);
            int W = input_cache.get_dim(2);
            int H_out = delta.get_dim(1);
            int W_out = delta.get_dim(2);

            d_filters = Tensor(filters.get_dim(0), filters.get_dim(1), filters.get_dim(2), filters.get_dim(3));
            d_biases = Tensor(biases.get_dim(0), biases.get_dim(1));
            Tensor new_delta(in_channels, H, W);
            d_filters.fill(0.0);
            d_biases.fill(0.0);
            new_delta.fill(0.0);

            for (int oc = 0; oc < out_channels; ++oc)
            {
                for (int i = 0; i < H_out; ++i)
                {
                    for (int j = 0; j < W_out; ++j)
                    {
                        double grad = delta(oc, i, j);
                        d_biases(oc, 0) += grad;
                        for (int ic = 0; ic < in_channels; ++ic)
                        {
                            for (int ki = 0; ki < kernel_size; ++ki)
                            {
                                for (int kj = 0; kj < kernel_size; ++kj)
                                {
                                    int xi = i * stride + ki - padding;
                                    int xj = j * stride + kj - padding;
                                    if (xi >= 0 && xj >= 0 && xi < H && xj < W)
                                    {
                                        d_filters(oc, ic, ki, kj) += input_cache(ic, xi, xj) * grad;
                                        new_delta(ic, xi, xj) += filters(oc, ic, ki, kj) * grad;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            delta.swap(std::move(new_delta));
        }

        void update_weights(float lr) override
        {
            filters = filters - d_filters * lr;
            biases = biases - d_biases * lr;
        }
    };

    struct Conv2DLayer2 : Layer
    {
        Tensor filters;
        Tensor biases;
        Tensor input_cache;
        char padding;
        int stride;
        int num_filters;
        int kernel_size;

        Conv2DLayer2(int num_filters, int kernel_size, int stride = 1, char padding = 'v')
            : stride(stride), padding(padding), num_filters(num_filters), kernel_size(kernel_size) {}

        void connect(int channels) override
        {
            filters = Tensor(kernel_size, kernel_size, channels, num_filters);
            biases = Tensor(num_filters);
        }
        void forward(const Tensor &input, Tensor &output) override
        {
            int N = 1, H, W, C = 1;

            int ndim = input.get_ndim(); // Número de dimensiones

            switch (ndim)
            {
            case 4: // (batch_size, height, width, channels)
                N = input.get_dim(0);
                H = input.get_dim(1);
                W = input.get_dim(2);
                C = input.get_dim(3);
                break;

            case 3: // (height, width, channels)
                H = input.get_dim(0);
                W = input.get_dim(1);
                C = input.get_dim(2);
                break;

            case 2: // (height, width)
                H = input.get_dim(0);
                W = input.get_dim(1);
                break;

            default:
                throw std::runtime_error("Invalid input size for Conv2D layer: Expected 2, 3, or 4 dimensions, got " + std::to_string(ndim));
            }

            int H_out;
            int W_out;

            if (padding == 'v')
            {
                H_out = (H + 2 * padding - kernel_size) / stride + 1;
                W_out = (W + 2 * padding - kernel_size) / stride + 1;
            }
            else if (padding == 's')
            {
                H_out = H;
                W_out = W;
            }
            else
            {
                throw std::runtime_error("Invalid padding type");
            }

            if (H_out < 1 || W_out < 1)
            {
                throw std::runtime_error("Invalid input size for Conv2D layer");
            }

            input_cache = input;
        }
    };

    struct MaxPooling2D
    {
        int pool_size;
        MaxPooling2D(int pool_size) : pool_size(pool_size) {}

        void forward(const Tensor &input, Tensor &output)
        {
            int H = input.get_dim(1);
            int W = input.get_dim(2);
            int H_out = H / pool_size;
            int W_out = W / pool_size;

            output = Tensor(input.get_dim(0), H_out, W_out);

            for (int c = 0; c < input.get_dim(0); ++c)
            {
                for (int i = 0; i < H_out; ++i)
                {
                    for (int j = 0; j < W_out; ++j)
                    {
                        double max_val = -INFINITY;
                        for (int pi = 0; pi < pool_size; ++pi)
                        {
                            for (int pj = 0; pj < pool_size; ++pj)
                            {
                                int xi = i * pool_size + pi;
                                int xj = j * pool_size + pj;
                                if (xi < H && xj < W)
                                    max_val = std::max(max_val, input(c, xi, xj));
                            }
                        }
                        output(c, i, j) = max_val;
                    }
                }
            }
        }
    };
    struct Flatten
    {
        void forward(const Tensor &input, Tensor &output)
        {
            int channels = input.get_dim(0);
            int height = input.get_dim(1);
            int width = input.get_dim(2);
            output = Tensor(channels * height * width);
        }
    };

    class Model
    {
        std::vector<shared_ptr<Layer>> layers;

    public:
        Model(const std::vector<shared_ptr<Layer>> &layers) : layers(layers)
        {
            for (size_t i = 1; i < layers.size(); ++i)
            {
                layers[i]->connect(layers[i - 1]->output_size);
            }
        }

        void forward(const Tensor &input, Tensor &output, int &alloc)
        {
            // layers[0]->forward(input, output);

            // std::cout << "Number of allocations after input: " << alloc << std::endl;
            layers[1]->forward(input, output);
            // std::cout << "Number of allocations after layer 1: " << alloc << std::endl;
            //   output.print();
            for (int i = 2; i < layers.size(); ++i)
            {
                layers[i]->forward(output, output);

                // std::cout << "Number of allocations after layer " << i << ": " << alloc << std::endl;
            }
        }

        double train(const Tensor &input, const Tensor &target, float lr, int &alloc)
        {
            Tensor output;
            forward(input, output, alloc);

            Tensor delta = target - output;
            // std::cout << "Number of allocations after forward: " << alloc << std::endl;

            for (int i = layers.size() - 1; i >= 0; i--)
            {
                layers[i]->backward(delta, lr);
                // std::cout << "Number of allocations after layer backward " << i << ": " << alloc << std::endl;
            }
            // std::cout << "Number of allocations after backward: " << alloc << std::endl;

            return compute_mse_loss(output, target);
        }
        double compute_mse_loss(const Tensor &output, const Tensor &target)
        {
            double sum = 0.0;
            const int rows = output.get_dim(0);
            for (int i = 0; i < rows; ++i)
                sum += std::pow(output(i) - target(i), 2);
            return sum / (rows);
        }

        void fit(const Tensor &input, const Tensor &target, double lr, int epochs, int &alloc)
        {
            for (int e = 0; e < epochs; ++e)
            {
                double loss = 0.0;
                int s = input.get_dim(0);
                for (size_t i = 0; i < s; ++i)
                {
                    Tensor act_input = input.get_row(i);
                    Tensor act_target = target.get_row(i);
                    loss += train(act_input, act_target, lr, alloc);
                }
                if (e % 100000 == 0)
                {
                    std::cout << "Epoch " << e << ", Loss: " << loss << std::endl;
                }
            }
        }
    };

};