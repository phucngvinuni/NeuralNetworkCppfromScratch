#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include "layer.h"
#include "neuralnetwork.h"
#include "function.h"

int main() {
    std::vector<Eigen::VectorXd> train_dataset;
    train_dataset.push_back((Eigen::VectorXd(2) << 1, 0).finished());
    train_dataset.push_back((Eigen::VectorXd(2) << 1, 1).finished());

    std::vector<Eigen::VectorXd> label_train_dataset;
    label_train_dataset.push_back((Eigen::VectorXd(1) << 0).finished());
    label_train_dataset.push_back((Eigen::VectorXd(1) << 1).finished());
    label_train_dataset.push_back((Eigen::VectorXd(1) << 1).finished());
    label_train_dataset.push_back((Eigen::VectorXd(1) << 0).finished());

    Layer hidden_layer(2, 2, function::sigmoid, function::sigmoid_derivative);
    Layer output_layer(2, 1, function::sigmoid, function::sigmoid_derivative);
    NeuralNetwork nn({hidden_layer, output_layer});

    nn.train(train_dataset, label_train_dataset, 0.01, 10000);

    double accuracy = nn.test(train_dataset, label_train_dataset);
    std::cout << "Accuracy: " << accuracy * 100 << "%" << std::endl;

    return 0;
}