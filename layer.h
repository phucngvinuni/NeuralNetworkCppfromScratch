#ifndef LAYER_H
#define LAYER_H

#include <iostream>
#include <Eigen/Dense>
#include <functional>

using namespace Eigen;

class Layer {
private:
    MatrixXd weights; // Ma trận trọng số
    VectorXd biases; // Vector bias
    VectorXd neurons_values; // Giá trị trước khi áp dụng hàm kích hoạt
    VectorXd neurons_values_activate; // Giá trị sau khi áp dụng hàm kích hoạt

    std::function<VectorXd(const VectorXd&)> activation_function; // Hàm kích hoạt
    std::function<VectorXd(const VectorXd&)> activation_function_derivative; // Đạo hàm của hàm kích hoạt

    VectorXd delta; // Gradient lỗi cho lớp

public:
    // Constructor
    Layer(int input_size, int neurons,
          std::function<VectorXd(const VectorXd&)> activation_function,
          std::function<VectorXd(const VectorXd&)> activation_function_derivative)
        : activation_function(activation_function),
          activation_function_derivative(activation_function_derivative) {
        weights = MatrixXd::Random(input_size, neurons) * 0.001; // Khởi tạo ngẫu nhiên nhỏ để tránh vanishing gradient
        biases = VectorXd::Zero(neurons); // Khởi tạo bias bằng 0
        std::cout << "Layer created" << std::endl;
    }

    // Lan truyền xuôi
    void forward(const VectorXd &input) {
        std::cout << "Forward in Layer" << std::endl;
        VectorXd Z = weights.transpose() * input + biases;
        if (Z.hasNaN()) {
        std::cerr << "Z has NaN in layer" << std::endl;}
        neurons_values = Z;
        neurons_values_activate = activation_function(Z); // Áp dụng hàm kích hoạt
    }

    // Lấy giá trị đầu ra sau kích hoạt
    VectorXd get_neurons_values_activate() const {
        return neurons_values_activate;
    }

    // Lấy giá trị trước kích hoạt
    VectorXd get_neurons_values() const {
        return neurons_values;
    }

    // Gán giá trị delta (gradient lỗi)
    void set_delta(const VectorXd &delta) {
        this->delta = delta;
    }

    // Lấy giá trị delta
    VectorXd get_delta() const {
        return delta;
    }

    // Tính delta trong lan truyền ngược
    void calculate_delta(const VectorXd &next_layer_delta, const MatrixXd &next_layer_weights) {
        VectorXd weighted_delta = next_layer_weights * next_layer_delta;
        delta = weighted_delta.array() * activation_function_derivative(neurons_values).array();
    }

    // Cập nhật trọng số và bias
void update_weights(const VectorXd &input_data, double learning_rate) {
    VectorXd gradient = input_data * delta.transpose();
    if (gradient.norm() > 1.0) {
        gradient = gradient.normalized() * 1.0;
    }
    weights -= learning_rate * gradient;
    biases -= learning_rate * delta;

    if (weights.hasNaN()) {
        std::cerr << "NaN detected in weights" << std::endl;
    }
    if (biases.hasNaN()) {
        std::cerr << "NaN detected in biases" << std::endl;
    }
}

    // Lấy trọng số
    MatrixXd get_weights() const {
        return weights;
    }

    // Lấy bias
    VectorXd get_biases() const {
        return biases;
    }

    // Cập nhật trọng số và bias với batch gradient descent
    void update_weights_batch(const MatrixXd &inputs, const MatrixXd &deltas, double learning_rate) {
        MatrixXd gradient_weights = inputs * deltas.transpose();
        VectorXd gradient_biases = deltas.rowwise().sum();

        weights -= learning_rate * gradient_weights;
        biases -= learning_rate * gradient_biases;
    }
    
    VectorXd derivative_of_activation_function() {
        std::cout << "derivative_of_activation_function" << std::endl;
        return activation_function_derivative(neurons_values);

    }
};

#endif // LAYER_H