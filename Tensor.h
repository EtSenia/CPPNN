#include <iostream>
#include <vector>

class Tensor
{
    int shape[4];
    int strides[4];
    int ndim;
    bool own_data = true;
    double *data = nullptr;
    int size = 0;

    void compute_strides();

public:
    Tensor() = default;
    Tensor(int i);
    Tensor(int i, int j);
    Tensor(int i, int j, int k);
    Tensor(int i, int j, int k, int l);
    Tensor(const Tensor &other);
    Tensor(const int *shape,
           const int *strides,
           int dims,
           double *data_ptr);
    // Tensor(std::initializer_list<double> _data);
    Tensor(std::initializer_list<std::initializer_list<double>> _data);

    ~Tensor();

    void fill(double value);

    Tensor &swap(Tensor &&other) noexcept;
    Tensor transpose_view(int axis1, int axis2) const;
    Tensor dot(const Tensor &other) const;
    Tensor &dot(const Tensor &other, Tensor &result) const;
    Tensor outer(const Tensor &other) const;
    Tensor get_row(int i) const;
    Tensor get_col(int i) const;

    int get_dim(int dim) const { return shape[dim]; }
    int get_ndim() const { return ndim; }

    Tensor operator+(const Tensor &other) const;
    Tensor &operator+=(const Tensor &other);
    Tensor operator-(const Tensor &other) const;
    Tensor &operator-=(const Tensor &other);
    Tensor operator*(const Tensor &other) const;
    Tensor &operator*=(const Tensor &other);
    Tensor operator*(double scalar) const;
    Tensor &operator=(const Tensor &other);
    Tensor &operator=(Tensor &&other) noexcept;

    double &operator()(int i);
    const double &operator()(int i) const;

    double &operator()(int i, int j);
    const double &operator()(int i, int j) const;

    double &operator()(int i, int j, int k);
    const double &operator()(int i, int j, int k) const;

    double &operator()(int i, int j, int k, int l);
    const double &operator()(int i, int j, int k, int l) const;

    void recursive_print(int depth, int offset)
    {
        if (depth == ndim)
        {
            return; // No hacer nada si alcanzamos más allá de las dimensiones permitidas.
        }

        int dim = shape[depth];
        std::cout << "[";

        for (int i = 0; i < dim; ++i)
        {
            int index = offset + i * strides[depth];

            if (depth == ndim - 1) // En la última dimensión, imprimir el dato.
            {
                std::cout << data[index];
            }
            else
            {
                recursive_print(depth + 1, index); // Llamada recursiva a la siguiente dimensión
            }

            // Si no es el último elemento, agregar una coma.
            if (i != dim - 1)
                std::cout << ", ";
        }

        std::cout << "]";
    }

    void recursive_print(int depth, int offset) const
    {
        if (depth == ndim)
        {
            return; // No hacer nada si alcanzamos más allá de las dimensiones permitidas.
        }

        int dim = shape[depth];
        std::cout << "[";

        for (int i = 0; i < dim; ++i)
        {
            int index = offset + i * strides[depth];

            if (depth == ndim - 1) // En la última dimensión, imprimir el dato.
            {
                std::cout << data[index];
            }
            else
            {
                recursive_print(depth + 1, index); // Llamada recursiva a la siguiente dimensión
            }

            // Si no es el último elemento, agregar una coma.
            if (i != dim - 1)
                std::cout << ", ";
        }

        std::cout << "]";
    }

    void print()
    {
        std::cout << "shape: ";
        for (int i = 0; i < ndim; ++i)
        {
            std::cout << shape[i] << " ";
        }
        recursive_print(0, 0);
        std::cout << std::endl;
    }
    void print() const
    {
        std::cout << "shape: ";
        for (int i = 0; i < ndim; ++i)
        {
            std::cout << shape[i] << " ";
        }
        recursive_print(0, 0);
        std::cout << std::endl;
    }
};