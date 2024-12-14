#include <iostream>
#include "function.h"

int main() {
    VectorXd output(3), target(3);
    output << 0.3, 0.5, 0.2; // Example prediction (softmax output)
    target << 0.0, 1.0, 0.0; // One-hot encoded target

    // Loss
    double loss = function::cross_entropy_loss(output, target);
    std::cout << "Cross-Entropy Loss: " << loss << "\n";

    // Derivative
    VectorXd grad = function::cross_entropy_derivative(output, target);
    std::cout << "Cross-Entropy Gradient:\n" << grad << "\n";

    return 0;
}
