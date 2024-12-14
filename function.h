#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <Eigen/Dense>
#include <algorithm> // std::max
#include <cmath> // std::exp
using namespace Eigen;

namespace function {

    // ReLU Activation Function
    VectorXd relu(const VectorXd &v) {
        return v.array().max(0.0);
    }

    // Derivative of ReLU
    VectorXd relu_derivative(const VectorXd &v) {
        return (v.array() > 0.0).cast<double>();
    }

    // Softmax Activation Function
    VectorXd softmax(const VectorXd &v) {
        VectorXd shifted_v = v.array() - v.maxCoeff(); // Đảm bảo ổn định số học
        VectorXd exp_values = shifted_v.array().    exp();
        return exp_values / exp_values.sum();
    }

    // Softmax Derivative (Jacobian)
    MatrixXd softmax_derivative(const VectorXd &v) {
        VectorXd s = softmax(v); // Tính softmax
        MatrixXd jacobian = MatrixXd::Zero(s.size(), s.size());

        for (int i = 0; i < s.size(); ++i) {
            for (int j = 0; j < s.size(); ++j) {
                if (i == j) {
                    jacobian(i, j) = s(i) * (1 - s(i));
                } else {
                    jacobian(i, j) = -s(i) * s(j);
                }
            }
        }
        return jacobian;
    }

    //  Loss Function
double cross_entropy_loss(const Eigen::VectorXd &output, const Eigen::VectorXd &target) {
    const double epsilon = 1e-15;

    // Chuyển đổi vector sang mảng
    Eigen::ArrayXd target_array = target.array();
    
    // Thêm epsilon vào output và tính log
    Eigen::ArrayXd log_output_array = (output.array() + epsilon).log().eval();

    // Tính toán tổn thất cross-entropy
    double loss = -(target_array * log_output_array).sum();

    return loss;
}


// Derivative of Cross-Entropy Loss
VectorXd cross_entropy_loss_derivative(const VectorXd &output, const VectorXd &target) {
    return output - target;
}

}

#endif // FUNCTIONS_H