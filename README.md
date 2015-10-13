# Classification

It includes two experiments on parsing; RNN_SLU and LR_Classify

# RNN_SLU
Semantic parsing using Recurrent Neural Network

In this experiment we intended to use Recurrent Neural Networks for the purpose of semantic parsing. We used the Elman architecture RNN implemented using Theano in (Mesnil et al., 2013) with Free917 data set for entity and function classification.

The questions of data set Free917 similar to LR-Classify repository, are already manually aligned in 'partial' directory. pkl_Gen3.py generates a pickle file out of the directory. elman-forward.py automatically generates the features and classifies the chunks to their corresponding entity or function.

# LR-Classify

In this experiment a Theano implemented logistics regression is used to classify the chunks in the questions of Free917 data set to their corresponding entities or functions. Feature.py generates two groups of features; lexical and compositional (functional).

Two optimization methods is used; one based on stochastic gradient and the other based on conjugate gradient. For smaller data sets conjugate gradient method usually perfumes better.
