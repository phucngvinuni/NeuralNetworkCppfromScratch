#include <iostream>
#include <Eigen/Dense>
#include <ctime>

#include "layer.h"
#include "function.h"
#include "neuralnetwork.h"
#include "utils.h"

using namespace Eigen;
using namespace function;

const std::string mnist_train_data_path = "data/MNIST/train-images-idx3-ubyte";
const std::string mnist_train_label_path = "data/MNIST/train-labels-idx1-ubyte";
const std::string mnist_test_data_path = "data/MNIST/t10k-images-idx3-ubyte";
const std::string mnist_test_label_path = "data/MNIST/t10k-labels-idx1-ubyte";

int main() {
    srand(time(0));

    std::vector<VectorXd> train_dataset;
    std::vector<VectorXd> label_train_dataset;

    std::vector<VectorXd> test_dataset;
    std::vector<VectorXd> label_test_dataset;

    utils::read_mnist_train_data(mnist_train_data_path, train_dataset);
    utils::read_mnist_train_label(mnist_train_label_path, label_train_dataset);

    utils::read_mnist_test_data(mnist_test_data_path, test_dataset);
    utils::read_mnist_test_label(mnist_test_label_path, label_test_dataset);
    std::cout << "Data loaded." << std::endl;
    Layer hidden_layer(784, 128, relu, relu_derivative);
    Layer output_layer(128, 10, softmax, softmax_derivative);
    std::cout << "2 layers created." << std::endl;
    NeuralNetwork nn({hidden_layer, output_layer});
    std::cout << "Training..." << std::endl;
    nn.train(train_dataset, label_train_dataset, 0.01, 5);
    nn.test(test_dataset, label_test_dataset);

    return 0;
}