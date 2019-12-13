---
title: "Multiple Optimal Learning Factors for Feed-forward Neural Nets""
date: 2019-12-12T21:40:32-06:00
draft: true
---

A batch training algorithm is developed for a fully connected multi-layer perceptron, with a single hidden layer, which uses two-stages per iteration. In the first stage, Newton's method is used to find a vector of optimal learning factors (OLFs), one for each hidden unit, which is used to update the input weights. Linear equations are solved for output weights in the second stage. Elements of the new method's Hessian matrix are shown to be weighted sums of elements from the Hessian of the whole network. The effects of linearly dependent inputs and hidden units on training are analyzed and an improved version of the batch training algorithm is developed. In several examples, the improved method performs better than first order training methods like backpropagation and scaled conjugate gradient, with minimal computational overhead and performs almost as well as Levenberg–Marquardt, a second order training method, with several orders of magnitude fewer multiplications.
