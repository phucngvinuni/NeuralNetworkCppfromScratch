#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "layer.h"
#include <vector>
#include <Eigen/Dense>
class NeuralNetwork {
private:
    std::vector<Layer> layers;

public:
    // Constructor
    NeuralNetwork(const std::vector<Layer>& layers) : layers(layers) {}

    // Forward propagation
    void forward(const Eigen::VectorXd& input) {
        for (size_t i = 0; i < layers.size(); ++i) {
            if (i == 0) {
                layers[i].forward(input);
            } else {
                layers[i].forward(layers[i - 1].get_neurons_values_activate());
            }
        }
    }

    // Backpropagation
    void backprop(const Eigen::VectorXd& target, double learning_rate) {
    // Output layer error
    Layer& output_layer = layers.back();
    Eigen::VectorXd output = output_layer.get_neurons_values_activate();
    Eigen::VectorXd error = function::cross_entropy_loss_derivative(output, target);
    output_layer.set_delta(error);

    // Backpropagate errors to hidden layers
    for (int i = layers.size() - 2; i >= 0; --i) {
        Layer& current_layer = layers[i];
        Layer& next_layer = layers[i + 1];
        current_layer.calculate_delta(
            next_layer.get_delta(),
            next_layer.get_weights()
        );
    }

    // Update weights and biases
    for (size_t i = 0; i < layers.size(); ++i) {
        if (i == 0) {
            layers[i].update_weights(target, learning_rate);
        } else {
            layers[i].update_weights(
                layers[i - 1].get_neurons_values_activate(),
                learning_rate
            );
        }
    }
}

    // Training method
    void train(
        const std::vector<Eigen::VectorXd>& train_data,
        const std::vector<Eigen::VectorXd>& labels,
        double learning_rate,
        int epochs
    ) {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            for (size_t i = 0; i < train_data.size(); ++i) {
                forward(train_data[i]);
                backprop(labels[i], learning_rate);

            }
            std::cout << "Epoch " << epoch << " done." << std::endl;
            std::cout << "Accuracy: " << test(train_data, labels) << std::endl;
            // Add epoch-wise metrics or logging here
        }
    }

    // Testing method
    double test(
        const std::vector<Eigen::VectorXd>& test_data,
        const std::vector<Eigen::VectorXd>& labels
    ) {
        int correct = 0;
        for (size_t i = 0; i < test_data.size(); ++i) {
            forward(test_data[i]);
            const Eigen::VectorXd& output = layers.back().get_neurons_values_activate();
            Eigen::Index predicted;
            output.maxCoeff(&predicted);
            Eigen::Index actual;
            labels[i].maxCoeff(&actual);
            if (predicted == actual) {
                correct++;
            }
        }
        return (double)correct / test_data.size();
    }
};

#endif // NEURALNETWORK_H