#include "Tensor.h"
#include <cstring>
#include <cassert>

void Tensor::compute_strides()
{
    strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; --i)
    {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
}

Tensor::Tensor(int i)
{
    ndim = 1;
    shape[0] = i;
    size = i;
    strides[0] = 1;
    data = new double[size];
    own_data = true;
}

Tensor::Tensor(int i, int j)
{
    ndim = 2;
    shape[0] = i;
    shape[1] = j;
    size = i * j;
    strides[1] = 1;
    strides[0] = j;
    data = new double[size];
    own_data = true;
}

Tensor::Tensor(int i, int j, int k)
{
    ndim = 3;
    shape[0] = i;
    shape[1] = j;
    shape[2] = k;
    size = i * j * k;
    strides[2] = 1;
    strides[1] = k;
    strides[0] = j * k;
    data = new double[size];
    own_data = true;
}

Tensor::Tensor(int i, int j, int k, int l)
{
    ndim = 4;
    shape[0] = i;
    shape[1] = j;
    shape[2] = k;
    shape[3] = l;
    size = i * j * k * l;
    strides[3] = 1;
    strides[2] = l;
    strides[1] = k * l;
    strides[0] = j * k * l;
    data = new double[size];
    own_data = true;
}

Tensor::Tensor(const Tensor &other) : ndim(other.ndim), size(other.size)
{
    for (int i = 0; i < ndim; ++i)
    {
        shape[i] = other.shape[i];
        strides[i] = other.strides[i];
    }
    data = new double[size];
    std::memcpy(data, other.data, size * sizeof(double));
}

Tensor::Tensor(const int *shp, const int *strde, int dims, double *data_ptr) : ndim(dims), data(data_ptr), own_data(false)
{
    size = 1;
    for (int i = 0; i < ndim; ++i)
    {
        shape[i] = shp[i];
        strides[i] = strde[i];
        size *= shape[i];
    }
}

Tensor::Tensor(std::initializer_list<std::initializer_list<double>> values)
{
    ndim = 2;
    int rows = static_cast<int>(values.size());
    int cols = values.begin()->size();

    // Verificar que todas las filas tengan el mismo tamaño
    for (const auto &row : values)
    {
        if (row.size() != static_cast<size_t>(cols))
        {
            throw std::invalid_argument("All rows must have the same size.");
        }
    }

    shape[0] = rows;
    shape[1] = cols;
    size = rows * cols;

    data = new double[size];

    int idx = 0;
    for (const auto &row : values)
    {
        for (double val : row)
        {
            data[idx++] = val;
        }
    }

    compute_strides();
}

Tensor::~Tensor()
{
    if (own_data && data)
        delete[] data;
}

void Tensor::fill(double value)
{
    std::fill(data, data + size, value);
}

Tensor &Tensor::swap(Tensor &&other) noexcept
{
    if (this != &other)
    {
        std::swap(shape, other.shape);
        std::swap(strides, other.strides);
        std::swap(ndim, other.ndim);
        std::swap(data, other.data);
        std::swap(size, other.size);
        std::swap(own_data, other.own_data);
    }
    return *this;
}

Tensor Tensor::transpose_view(int axis1, int axis2) const
{
    int new_shape[4];
    int new_strides[4];

    for (int i = 0; i < ndim; ++i)
    {
        new_shape[i] = shape[i];
        new_strides[i] = strides[i];
    }

    std::swap(new_shape[axis1], new_shape[axis2]);
    std::swap(new_strides[axis1], new_strides[axis2]);

    return Tensor(new_shape, new_strides, ndim, data);
}

Tensor Tensor::dot(const Tensor &other) const
{
    if (shape[0] != other.shape[0])
        throw std::runtime_error("Dimension mismatch for dot product");
    Tensor result(other.shape[1]);
    for (int j = 0; j < other.shape[1]; ++j)
    {
        double sum = 0.0;
        for (int k = 0; k < shape[0]; ++k)
        {
            double a = data[k];
            double b = other.data[k * other.strides[0] + j * other.strides[1]];
            sum += a * b;
        }
        result.data[j] = sum;
    }
    return result;
}

Tensor &Tensor::dot(const Tensor &other, Tensor &result) const
{
    if (shape[0] != other.shape[0])
        throw std::runtime_error("Dimension mismatch for dot product");
    Tensor temp(other.shape[1]);
    for (int j = 0; j < other.shape[1]; ++j)
    {
        double sum = 0.0;
        for (int k = 0; k < shape[0]; ++k)
        {
            double a = data[k];
            double b = other.data[k * other.strides[0] + j * other.strides[1]];
            sum += a * b;
        }
        temp.data[j] = sum;
    }
    result.swap(std::move(temp));
    return result;
}

Tensor Tensor::outer(const Tensor &v2) const
{
    // Obtener longitudes
    int len1 = shape[0];
    int len2 = v2.shape[0];

    Tensor result(len1, len2);

    for (int i = 0; i < len1; ++i)
    {
        double val1 = data[i];

        for (int j = 0; j < len2; ++j)
        {
            double val2 = v2.data[j];

            result.data[i * result.strides[0] + j * result.strides[1]] = val1 * val2;
        }
    }

    return result;
}

Tensor Tensor::get_row(int row) const
{
    if (ndim != 2)
        throw std::runtime_error("get_row: tensor must be 2D");
    if (row < 0 || row >= shape[0])
        throw std::out_of_range("Row index out of bounds");

    int new_shape[4] = {shape[1], 1, 1, 1};
    int new_strides[4];
    for (int i = 0; i < 4; ++i)
    {
        new_strides[i] = strides[i];
    }

    double *row_data = &data[row * strides[0]];
    return Tensor(new_shape, new_strides, 1, row_data);
}

Tensor Tensor::get_col(int col) const
{
    if (ndim != 2)
        throw std::runtime_error("get_col: tensor must be 2D");
    if (col < 0 || col >= shape[1])
        throw std::out_of_range("Column index out of bounds");

    Tensor result(shape[0], 1);
    for (int i = 0; i < shape[0]; ++i)
        result(i, 0) = data[i * strides[0] + col * strides[1]];
    return result;
}

Tensor Tensor::operator+(const Tensor &other) const
{
    Tensor result(*this);
    for (int i = 0; i < size; ++i)
        result.data[i] = data[i] + other.data[i];
    return result;
}

Tensor &Tensor::operator+=(const Tensor &other)
{
    for (int i = 0; i < size; ++i)
        data[i] += other.data[i];
    return *this;
}

Tensor Tensor::operator-(const Tensor &other) const
{
    Tensor result(*this);
    for (int i = 0; i < size; ++i)
        result.data[i] = data[i] - other.data[i];
    return result;
}

Tensor &Tensor::operator-=(const Tensor &other)
{
    for (int i = 0; i < size; ++i)
        data[i] -= other.data[i];
    return *this;
}

Tensor Tensor::operator*(const Tensor &other) const
{
    Tensor result(*this);
    for (int i = 0; i < size; ++i)
        result.data[i] = data[i] * other.data[i];
    return result;
}

Tensor &Tensor::operator*=(const Tensor &other)
{
    for (int i = 0; i < size; ++i)
        data[i] *= other.data[i];
    return *this;
}

Tensor Tensor::operator*(double scalar) const
{
    Tensor result(*this);
    for (int i = 0; i < size; ++i)
        result.data[i] = data[i] * scalar;
    return result;
}

Tensor &Tensor::operator=(const Tensor &other)
{
    if (this != &other)
    {
        if (size != other.size)
        {
            if (own_data && data)
                delete[] data;
            size = other.size;
            data = new double[size];
        }

        ndim = other.ndim;
        own_data = true;

        for (int i = 0; i < ndim; ++i)
        {
            shape[i] = other.shape[i];
            strides[i] = other.strides[i];
        }

        std::memcpy(data, other.data, size * sizeof(double));
    }
    return *this;
}

Tensor &Tensor::operator=(Tensor &&other) noexcept
{
    // dont copy, transfer
    if (this != &other)
    {
        if (size != other.size)
        {
            if (own_data && data)
                delete[] data;
            size = other.size;
            data = other.data;
        }

        ndim = other.ndim;
        own_data = other.own_data;

        for (int i = 0; i < ndim; ++i)
        {
            shape[i] = other.shape[i];
            strides[i] = other.strides[i];
        }

        other.ndim = 0;
        other.own_data = false;
        other.data = nullptr;
        other.size = 0;
    }
    return *this;
}

double &Tensor::operator()(int i)
{
    return data[i];
}

const double &Tensor::operator()(int i) const
{
    return data[i];
}

double &Tensor::operator()(int i, int j)
{
    return data[i * strides[0] + j * strides[1]];
}

const double &Tensor::operator()(int i, int j) const
{
    return data[i * strides[0] + j * strides[1]];
}

double &Tensor::operator()(int i, int j, int k)
{
    return data[i * strides[0] + j * strides[1] + k * strides[2]];
}

const double &Tensor::operator()(int i, int j, int k) const
{
    return data[i * strides[0] + j * strides[1] + k * strides[2]];
}

double &Tensor::operator()(int i, int j, int k, int l)
{
    return data[i * strides[0] + j * strides[1] + k * strides[2] + l * strides[3]];
}

const double &Tensor::operator()(int i, int j, int k, int l) const
{
    return data[i * strides[0] + j * strides[1] + k * strides[2] + l * strides[3]];
}
