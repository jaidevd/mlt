# Generative vs discriminative classifiers

## Discriminative classifiers: 

Discriminative classifiers divide the feature space into regions that separate
the data belonging to different classes such that every separate region contains samples belonging
to a single class. The decision boundary is determined by constructing and solving equations of the form

$$ \mathbf{y} = \mathbf{w}^T\mathbf{\phi(X)}+b $$

In practice equations of this kind may not always be solvable, so discriminative classifiers
minimize some loss to solve an approximation of this equation. Algorithms of this type are not
necessarily concerned with the probability distributions of features or labels.

Additionally, determining
probabilities using the conditional distribution of y given x, can give us an idea of how strong the
possibility of a data instance belonging to a specific class is.

## Generative classifiers:
Generative classifiers are concerned with finding the joint probability of features and labels, i.e.
they try to estimate the probabilities of all pairs of features and labels found in the training
data. Generative models assume that there is an underlying probability distribution from which the
data has been sampled, and try to estimate the parameters of this distribution. Thus, they see the
data as being "generated" through an estimated distribution (discriminative models, in contrast, are
need not make this assumption).

# What is the Naïve Bayes method?

The Naïve Bayes algorithm is a set of generative classifiers. The
fundamental assumption in a Naïve Bayes algorithm is that

conditional on the class, features are independent, i.e. one feature value appearing in a given
class is independent of the value of any other feature in the same class.

E.g. Given that a sample is drawn from the "setosa" class of the iris dataset, knowing it's sepal
length should tell us nothing about its petal width.


> The algorithm assumes that a change in value of one feature does not impact the other
(within the ambit of the subset as above).

This is impractical in real life, since features are likely to be interrelated (e.g. in the Titanic
dataset, given that a passenger survived and paid a low fare, we might be able to conclude that they
are a woman. In text data, we know that occurences of words are not independent of each other).
Hence it is a ‘naïve’ (oversimplifying) assumption.

Thus, the algorithm:

* assumes that the features in each class in the dataset is sampled from a distribution
* estimates the parameters for each such distribution

Ultimately, the model ends up with one unique distribution for each class.
Note that these distributions are all from from the same family (i.e. Normal, Binomial, Gamma, etc).

Then, from the set of features in the test dataset, the algorithm estimates the probabilities that the new sample belongs to each of the class-wise districutions, and the class that has the highest probability is the predicted class.

> Add an example - perferably from the HWG dataset.

> Explain briefly the example.

# Mathematical model

The goal of the prediction is to find the probability that a feature vector belongs to a given
class. In other words, for a set of $k$ classes ${c_1, c_2, \dots, c_k}$, the prediction involves
computing the following probabilities:

$$
P(c_i\|\mathbf{x}) \forall i \in [1, k]
$$

where $\mathbf{x}$ denotes the feature vector and $c_i$ is the $i^th$ class. This expression denotes
the posterior probability that a feature vector belongs to a given class. 

However, in the training dataset, we only have the prior probabilities, i.e. we can only compute:

$$
P(\mathbf{x}|c_i)
$$

which denotes the probability of finding a feature vector in a given class. 

So, the algorithm uses the Bayes’ theorem to convert the prior probabilities into the posterior
probabilities.

## Estimating class-wise distributions

As in typical machine learning problems, the parameters of the class-wise distributions are unknown.
These are typically computed with maximum likelihood estimation.

## Prediction
For a sample $\mathbf{x}$ with features, say, $[x_1, x_2, x_3, x_4]$ and label $y$,

we want to find the probability of the  sample belonging to class $c_{i}$ i.e
required evaluation is

$$
P(y=c_i|\mathbf{x}; \mathbf{\theta})
$$

- the  probability of the class being $c_i$ given the sample $\mathbf{x}$, parameterized by $\mathbf{\theta}$.

(Note: $\mathbf{\theta}$ is a vector containing the parameters of the corresponding probability
distributions. It could denote a vector of $[\mu, \sigma]$ for a Gaussian distribution, or a vector
of $k, n, p$ for a Binomial distribution, etc.)

This expression, then can be expanded using Bayes theorem as follows:

$$
P(y=c_i|\mathbf{x}) = \frac{P(\mathbf{x}|y=c_i)}{P(\mathbf{x})} \times P(y=c_{i})
$$

We know the following quantities from the training data:

1. the prior probability $P(\mathbf{x}|y=c_i)$ from the training data, and
2. the class priors $P(y=c_i) \forall i \in [1, k]$

The denominator in the RHS can be ignored, since it is only a normalizing factor, and will not
affect the relative probabilities of a sample belonging to different classes.

The expression for the prior probability, $P(\mathbf{x}|y=c_i)$ can be expanded as

$$
P(x_1, x_2, x_3, x_4 | y=c_i)
$$

(note that $\mathbf{x}$ is a vector containing 4 elements). Using the naive assumption of
class-conditional independence, this expression can be expanded as follows:

$$
P(x_1|y=c_i)P(x_2|y=c_i)P(x_3|y=c_i)P(x_4|y=c_i)
$$

Now, this is a representation which can be computed from the training data.

With all these quantities in place for each class, we can compute the LHS for all $k$ classes.
Whichever class has the highest probability then becomes our prediction for the sample $\mathbf{x}$.


# Examples


> Insert Titanic example


# Types of Naive Bayes Classifiers

When to use a naïve bayes algorithm?  Small datasets (n<=m) Features are uncorrelated Categorical
features (Bernoulli/Categorical NB) Gaussian Naïve bayes can be used when data is non-linear
Advantages Naïve Bayes classifiers generally have high bias and low variance. They generalise the
training dataset properties. For example in the Titanic dataset considered above, variables like
Name, Age and Fare take a large range of values and hence are highly prone to overfitting if we use
classifiers like Linear regression, logistic regression or Decision trees.  Naïve Bayes is a good
baseline model owing to it’s simple assumption and generalisation property.  Naïve Bayes models
train quickly.  No requirement of transformation of non-linear datasets Disadvantages The features
need not necessarily be conditionally independent and the naive assumption fails in most cases.
Could produce lower accuracy.  Cannot be used with data where features are numerical and not
distributed normally.


# Further Reading
Sources: Generative vs discriminative models https://cs229.stanford.edu/summer2019/cs229-notes2.pdf
https://stats.stackexchange.com/questions/12421/generative-vs-discriminative Naïve Bayes vs Logistic
regression: https://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf Naïve bayes as a
baseline classifier: https://www.cl.cam.ac.uk/teaching/1617/MLRD/handbook/nb.pdf Also refer:
https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote05.html
