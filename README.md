
# 🌟 Building Your First Neural Network in TensorFlow

Welcome to the repository for implementing a basic neural network using TensorFlow! This project includes the essential components of building a neural network, including forward propagation, cost computation, and model training. 🚀

## 📚 Table of Contents

1. [Overview](#overview)
2. [Forward Propagation](#forward-propagation)
3. [Compute Cost](#compute-cost)
4. [Train the Model](#train-the-model)
5. [Plotting Functions](#plotting-functions)
6. [Credits](#credits)
7. [Contact](#contact)

## 📖 Overview

This repository demonstrates how to build a simple neural network using TensorFlow. The network architecture consists of three layers with ReLU activation functions in between, designed to perform a classification task.

## 🔄 Forward Propagation

The `forward_propagation` function implements forward propagation through the neural network:

```python
def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model: 
    LINEAR -> RELU -> LINEAR -> RELU -> LINEAR
    
    Arguments:
    X -- input dataset, of shape (input size, number of examples)
    parameters -- dictionary containing the model's weights and biases:
                  "W1", "b1", "W2", "b2", "W3", "b3"
    
    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    
    # Retrieve parameters from the dictionary for each layer
    W1, b1 = parameters['W1'], parameters['b1']
    W2, b2 = parameters['W2'], parameters['b2']
    W3, b3 = parameters['W3'], parameters['b3']

    # Layer 1: Linear -> ReLU
    Z1 = tf.add(tf.matmul(W1, X), b1)  # Compute Z1 = W1*X + b1
    A1 = tf.keras.activations.relu(Z1)  # Apply ReLU activation

    # Layer 2: Linear -> ReLU
    Z2 = tf.add(tf.matmul(W2, A1), b2)  # Compute Z2 = W2*A1 + b2
    A2 = tf.keras.activations.relu(Z2)  # Apply ReLU activation

    # Layer 3: Linear (no activation for output layer)
    Z3 = tf.add(tf.matmul(W3, A2), b3)  # Compute Z3 = W3*A2 + b3

    return Z3  # Return the final output (Z3)
```

### Forward Propagation Testing

To ensure the correctness of the forward propagation implementation, we conduct tests using the `forward_propagation_test` function:

```python
def forward_propagation_test(target, examples):
    """
    Tests the forward_propagation function by performing forward passes 
    on batches of examples and checking the shape and values of the output.
    
    Arguments:
    target -- the forward_propagation function to be tested
    examples -- the dataset to test, expected to be a TensorFlow dataset
    
    Returns:
    None
    """
    
    # Create minibatches from the examples dataset, with 2 examples per batch
    minibatches = examples.batch(2)
    
    for minibatch in minibatches:
        # Perform the forward propagation on the current minibatch
        forward_pass = target(tf.transpose(minibatch), parameters)
        print(forward_pass)
        
        # Check if the output is a tensor and has the expected shape
        assert isinstance(forward_pass, tf.Tensor), "Output is not a tensor"
        assert forward_pass.shape == (6, 2), "The shape of the output must be (6, 2)"
        
        # Check if the output values are approximately as expected
        expected_output = np.array([[-0.13430887,  0.14086473],
                                    [ 0.21588647, -0.02582335],
                                    [ 0.7059658,   0.6484556 ],
                                    [-1.1260961,  -0.9329492 ],
                                    [-0.20181894, -0.3382722 ],
                                    [ 0.9558965,   0.94167566]])
        assert np.allclose(forward_pass, expected_output), "Output does not match the expected values"
        
        break  # Only test the first minibatch

    # Indicate that all tests have passed successfully
    print("✅ All forward propagation tests passed successfully.")
```

---

## 📊 Compute Cost

The `compute_cost` function calculates the categorical cross-entropy cost:

```python
def compute_cost(logits, labels):
    """
    Computes the categorical cross-entropy cost.
    
    Args:
    logits (tf.Tensor): Output of forward propagation (output of the last LINEAR unit), 
                        of shape (num_classes, num_examples).
    labels (tf.Tensor): True labels, same shape as logits, one-hot encoded.
    
    Returns:
    tf.Tensor: Scalar tensor representing the cost.
    """
    
    # Ensure the logits and labels are transposed only if necessary
    logits = tf.transpose(logits)
    labels = tf.transpose(labels)

    # Compute the categorical cross-entropy loss and sum it across all examples
    cost = tf.reduce_sum(
        tf.keras.metrics.categorical_crossentropy(labels, logits, from_logits=True)
    )
    
    return cost
```

### Cost Testing

Testing the `compute_cost` function ensures it performs as expected:

```python
def compute_cost_test(target, Y):
    pred = tf.constant([[ 2.4048107,   5.0334096 ],
             [-0.7921977,  -4.1523376 ],
             [ 0.9447198,  -0.46802214],
             [ 1.158121,    3.9810789 ],
             [ 4.768706,    2.3220146 ],
             [ 6.1481323,   3.909829  ]])
    minibatches = Y.batch(2)
    for minibatch in minibatches:
        result = target(pred, tf.transpose(minibatch))
        break
        
    print(result)
    assert(type(result) == tf.Tensor), "Use the TensorFlow API"
    
    # Indicate successful test completion
    print("✅ All tests passed successfully.")
```

---

## 🛠️ Train the Model

The `model` function implements the neural network and manages the training process:

```python
def model(X_train, Y_train, X_test, Y_test, learning_rate=0.0001,
          num_epochs=1500, minibatch_size=32, print_cost=True):
    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.
    
    Arguments:
    X_train -- training set, of shape (input size = 12288, number of training examples = 1080)
    Y_train -- test set, of shape (output size = 6, number of training examples = 1080)
    X_test -- test set, of shape (input size = 12288, number of test examples = 120)
    Y_test -- test set, of shape (output size = 6, number of test examples = 120)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs for optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 10 epochs
    
    Returns:
    parameters -- parameters learnt by the model. They can be used to make predictions.
    """
    
    costs = []   # To keep track of the cost
    train_acc = []
    test_acc = []
    
    # Initialize parameters
    parameters = initialize_parameters()

    # Adam optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    # Categorical accuracy for tracking accuracy
    test_accuracy = tf.keras.metrics.CategoricalAccuracy()
    train_accuracy = tf.keras.metrics.CategoricalAccuracy()

    dataset = tf.data.Dataset.zip((X_train, Y_train)).batch(minibatch_size).prefetch(8)
    test_dataset = tf.data.Dataset.zip((X_test, Y_test)).batch(minibatch_size).prefetch(8)

    m = dataset.cardinality().numpy()  # Get number of training samples

    # Training loop
    for epoch in range(num_epochs):
        epoch_cost = 0.0
        train_accuracy.reset_state()  # Reset train accuracy for the new epoch

        # Iterate through minibatches
        for minibatch_X, minibatch_Y in dataset:
            with tf.GradientTape() as tape:
                # Forward propagation
                Z3 = forward_propagation(tf.transpose(minibatch_X), parameters)
                # Compute cost
                minibatch_cost = compute_cost(Z3, tf.transpose(minibatch_Y))

            # Update train accuracy
            train_accuracy.update_state(tf.transpose(Z3), minibatch_Y)

            # Compute gradients and update parameters
            grads = tape.gradient(minibatch_cost, parameters.values())
            optimizer.apply_gradients(zip(grads, parameters.values()))

            epoch_cost += minibatch_cost

        # Average cost for the epoch
        epoch_cost /= m

        # Print cost every 10 epochs
        if print_cost and epoch % 10 == 0:
            print(f"Cost after epoch {epoch}: {epoch_cost:.6f}")
            print(f"Train accuracy: {

train_accuracy.result().numpy() * 100:.2f}%")
            costs.append(epoch_cost)

        # Update test accuracy
        test_accuracy.reset_state()  # Reset test accuracy for the new epoch
        for minibatch_X_test, minibatch_Y_test in test_dataset:
            Z3_test = forward_propagation(tf.transpose(minibatch_X_test), parameters)
            test_accuracy.update_state(tf.transpose(Z3_test), minibatch_Y_test)

    # Plot the cost
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters
```

---

## 📈 Plotting Functions

To help visualize the training process, we've included functions to plot costs and accuracies:

### Cost Plot

```python
def plot_cost(costs, learning_rate):
    """
    Plots the cost over epochs.

    Arguments:
    costs -- list of costs over epochs
    learning_rate -- the learning rate used in training
    """
    
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
```

### Accuracy Plot

```python
def plot_accuracy(train_acc, test_acc, learning_rate):
    """
    Plots training and test accuracy over epochs.

    Arguments:
    train_acc -- list of training accuracy values
    test_acc -- list of test accuracy values
    learning_rate -- the learning rate used in training
    """
    
    plt.plot(train_acc, label='Train Accuracy')
    plt.plot(test_acc, label='Test Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.title("Learning rate =" + str(learning_rate))
    plt.legend()
    plt.show()
```

---

## 🎓 Credits

This project is inspired by the [Deep Learning Specialization](https://www.deeplearning.ai/courses/deep-learning-specialization/) from DeepLearning.AI. A big thank you to Andrew Ng and the team for providing such an excellent resource! 🙌

---

## 📬 Contact

Feel free to reach out if you have any questions or feedback!

- **Email**: [swe.saqib@gmail.com](mailto:swe.saqib@gmail.com)
- **LinkedIn**: [LinkedIn Profile](https://www.linkedin.com/in/muhammad-saqib-b77aa41b6/)
