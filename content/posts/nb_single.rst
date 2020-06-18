---
title: "The Naive Bayes Classifier"
summary: "...not so naive after all"
---

.. contents::
   :depth: 3
..

*Updated*:

**Summary**: An indepth look at the classic Naive Bayes classifier with
code examples, applications and more!

.. admonition:: TODO

   Figure out how to add Read More... link here so Tumblr only shows the
   summary for each post

.. container:: contents

   Sections

--------------

Introduction
============

Naive Bayes, a classical model for decision making with roots in
probability theory has been around for a long time and is both revered
and used as a punching bag by more modern methods. While it is
prominently featured in machine learning texts and libraries, there is
not so much *learning* happening with the Naive Bayes, if one takes the
view that learning happens via iteration and adjustment of some internal
parameters. Nope. As we will see, the Naive Bayes, with simple (yet
questionable) assumptions about the data looks at the data and makes a
decision based on the *likelihood* and the *prior* distributions.
Nothing to iterate, nothing to tune (well, not really, but more on that
later). In fact, for the most part, it operates from a *generative* or
*sampling* paradigm. Although it is interesting to note that it can also
act as a typical classifier (say, like the logistic regression model)
and operate out of a *discriminative* paradigm (more below).

So how does it all fit? Well, machine learning problems, specifically
the supervised learning problem, can be posed as a general problem of
learning a function :math:`f: X -> Y` that maps the inupt X to the
outputs Y as best as possible. Supervised classification problems
involve Y being discrete valued, taking on one of C classes or
categories. Now, from probability theory, finding the learning function
is equivalent to finding the posterior probability :math:`P(Y|X)`.

   Q: "Ok, so how to estimate :math:`P(Y|X)`?"

   A: (in a deep voice) "**The Bayes Theorem**"

.. admonition:: TODO

   Find the LateX equivalent for "->" symbol and replace it in the
   sentence above.

--------------

Theory
======

Broadly the idea is to view the inputs and outputs as a joint
distribution described by :math:`P(\mathbf{x}, i_c)`. The joint
distribution can be expressed in two ways.

.. math:: P(\mathbf{x}, i_c) = p(\mathbf{x}|i_c) \cdot P(i_c) = p(i_c | \mathbf{x}) \cdot P(\mathbf{x})

.. admonition:: TODO

   Explain the notation used

Rearranging the terms, we get the Bayes rule as

.. math:: p(i_c|\mathbf{x}) = \frac{p(\mathbf{x}|i_c) \cdot P(i_c)}{P(\mathbf{x})}

This is a way of expressing the posterior probability using the
likelihood, priors and evidence, i.e.,

.. math:: posterior = \frac{likelihood \times prior}{evidence}

Note that the denominator *evidence* is merely a scaling factor to
ensure that the *posterior* is within the range :math:`(0, 1)`, like all
probabilities should. The likelihood and priors form the most impact on
the estimated posterior probability.

So far so good, but...

Applying the Bayes rule, in its present form, to classification problems
is impractical. Why? Consider the case of boolean variables where both Y
and X take on either 0 or 1 for values. If each X in the data is an
n-dimensional vector, then we will need to estimate about :math:`2^n`
parameters, which can explode for even modest values of n. Also, what
about multi-valued or continuous valued variables, with
interactions....aaargh!

(Deep breaths...)

   *Keep Calm and Naive Bayes on!*

It must be noted that there is an entire field called *Bayesian
Learning* dedicated to addressing this problem.

The Naive Bayes (NB) glosses over some of the issues by simply assuming
that the :math:`\mathbf{x}` are *independent*, for (or within) each
class. Under this assumption, the conditional probability (likelihood)
can be expressed as:

.. math:: p(\mathbf{x}|i_c) = \prod_{j=1}^{P} p(x_j|i_c)

.. admonition:: TODO

   Explain how under independence, the joint probability becomes a
   product form of same terms and hence results in the form above.

Depending on the assumption on the underlying likelihood distribution
(ex. Gaussian, Binomial, Poisson), different NB classifiers can be
generated for different tasks.

Classifier Modes
----------------

A typical classifier operates in one of two *modes* or *paradigm*.

-  sampling (generative) paradigm, which focuses on the individual
   distributions of the classes, comparing these to indirectly produce a
   comparison between the classes.
-  diagnostic (discriminative) paradigm, which focuses on the
   differences between the classes, i.e., on discriminating them

The nice thing about NB is that it can be viewed from both perspectives.

A quick example, using the `Scikit-Learn <https://scikit-learn.org>`__
Python library, illustrates the typical workflow involved intraining a
Naive Bayes classifier and testing its performance on test data (i.e.,
data not used or seen by the classifier during training).

.. code:: python

   """Example of using Gaussing Naive Bayes for Classification

   Ref: https://scikit-learn.org/stable/modules/naive_bayes.html#gaussian-naive-bayes

   """

   from sklearn.datasets import load_iris
   from sklearn.model_selection import train_test_split
   from sklearn.naive_bayes import GaussianNB

   X, y = load_iris(return_X_y=True)
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

   gnb_clf = GaussianNB()
   gnb_clf.fit(X_train, y_train)
   y_pred = gnb_clf.predict(X_test)
   print(f"Total number of samples in testing set: {X_test.shape[0]}")
   print(f"Number of mislabeled points in the test set: {(y_test != y_pred).sum()}")

.. code:: shell

   Total number of samples in testing set: 75
   Number of mislabeled points in the test set: 4

While this *off the shelf* approach is fine for quick comparison or
obtaining a baseilne, implementing from scratch will lead to a deeper
understanding on the inner workings of the Naive Bayes classifier.

--------------

Types of Naive Bayes
====================

Applications in text analytics - document categorization, sentiment
analysis, spam identification

Apply NB to datasets like Titanic or MNIST or other popular ones.

Can multiple distributions be used for different subsets of features and
be combined to form a joint discriminating function?

Continuous Random Variables
---------------------------

Gaussian Naive Bayes from Scratch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Below is an implementation of a Gaussian Naive Bayes classifier from
scratch.

.. code:: python

   """Module for implementing Naive Bayes algorithm for classification.

   These algorithms assume strong independence of the features within a class and
   the type of the likelihood function (ex. Gaussian, Bernoulli, Multinomial,
   etc.)
   """

   from collections import Counter

   import numpy as np

   class NaiveGaussian():
       """Naive Bayes classifier assuming Gaussian likelihood function

       For more details see :ref:`Naive Bayes <../docs/nb.html>`.

       Attributes
       ----------

       class_count_: array, shape (nb_classes,)
           Number of samples per class
       class_prior_: array, shape (nb_classes,)
           Probability of occurrance of each class
       class_labels_: array, shape (nb_classes,)
           Class labels or IDs present in the data
       feature_mean_: array, shape (nb_classes, nb_features)
           Input feature mean per class
       feature_variance_: array, shape (nb_classes, nb_features)
           Input feature variance per class

       Example Usage
       -------------

       >>> import numpy as np
       >>> np.random.seed(0)
       >>> X = np.random.randn(100,2)
       >>> y = X.sum(axis=1) > 0
       >>> clf = NaiveGaussian().fit(X, y)
       >>> Xtest = np.random.randn(20, 2)
       >>> ytest = Xtest.sum(axis=1) > 0
       >>> ypred = clf.predict(Xtest)
       >>> err = np.sum(ytest != ypred)
       >>> print(f'Misclassified {err} out of {ytest.shape[0]} samples')

       """

       def fit(self, X, y):
           """Fit a Naive Bayes classifier assuming Gaussian likelihood

           Parameters
           ----------

           X: array-like, shape (nb_samples, nb_features)
              Training data
           y: array-like, shape (nb_samples)
              Target output labels or classes

           Returns
           -------

           self: object
           """

           self.nb_features_ = X.shape[1]
           # compute class labels, counts and priors
           self.class_counts_ = Counter(y)
           # TODO does sorting the keys matter?
           self.class_labels_ = np.asarray(sorted(self.class_counts_.keys()))
           self.nb_classes_ = len(self.class_labels_)
           self.class_prior_ = np.asarray([self.class_counts_[k]/y.shape[0] 
               for k in self.class_labels_])
           self.feature_mean_ = np.zeros((self.nb_classes_, self.nb_features_))
           self.feature_variance_ = np.zeros((self.nb_classes_,
               self.nb_features_))
           for c in range(self.nb_classes_):
               Xc = X[y == self.class_labels_[c]]
               self.feature_mean_[c, :] = np.mean(Xc, axis=0)
               self.feature_variance_[c, :] = np.var(Xc, axis=0)

           return self

       def _joint_log_likelihood(self, X):
           """Return the Joint Loglikelihood value"""
           joint_log_likelihood = []
           for i in range(np.size(self.class_labels_)):
               jointi = np.log(self.class_prior_[i])
               n_ij = - 0.5 * np.sum(np.log(2. * np.pi * 
                   self.feature_variance_[i, :]))
               n_ij -= 0.5 * np.sum(((X - self.feature_mean_[i, :]) ** 2) /
                                   (self.feature_variance_[i, :]), 1)
               joint_log_likelihood.append(jointi + n_ij)

           joint_log_likelihood = np.array(joint_log_likelihood).T

           return joint_log_likelihood

       def predict(self, X):
           """
           Perform classification on an array of test vectors X.

           Parameters
           ----------

           X : array-like of shape (n_samples, n_features)

           Returns
           -------

           C : ndarray of shape (n_samples,)
               Predicted target values for X

           """
           jll = self._joint_log_likelihood(X)

           return self.class_labels_[np.argmax(jll, axis=1)]

Naive Bayes Decision Boundary
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Assumes the likelihood to be a Gaussian distribution.

.. code:: python

   import numpy
   import matplotlib.pyplot as plt
   import seaborn as sns
   sns.set()
   from sklearn.naive_bayes import GaussianNB

   # generate bivariate random variables for 2 classes
   seed = 0
   numpy.random.RandomState(seed)
   mean = [0, 0]
   cov = numpy.identity(2)
   x1 = numpy.random.multivariate_normal(mean, cov, size=(2, 100))
   y1 = numpy.zeros(x1.shape[1])
   x2 = numpy.random.multivariate_normal([0, 2], numpy.diag([1, 2]), size=(2, 100))
   y2 = numpy.ones(x2.shape[1])

   # concat data into a single set
   X = numpy.vstack((x1[0], x2[0]))
   y = numpy.hstack((y1, y2))

   # train a Gaussian NB classifier
   clf = GaussianNB().fit(X, y)

   # test data
   test_samples = 5000
   Xtest = numpy.random.uniform([-3, -2], [4, 6], size=(test_samples, 2)) 
   ypred = clf.predict(Xtest) # predictions

   # plots
   fig, ax = plt.subplots()
   ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdBu', s=50)
   ax.scatter(Xtest[:, 0], Xtest[:, 1], c=ypred, alpha=0.1, s=10, cmap='RdBu')
   plt.show()
   plt.savefig('bivariate_gauss_nb_boundary.png', dpi=200)

.. figure:: ../tutorials/bivariate_gauss_nb_boundary.png
   :alt: Decision boundary for a bivariate Gaussian Naive Bayes classifier
   :figclass: align-center
   :width: 50.0%

   Decision boundary for a bivariate Gaussian Naive Bayes classifier

Under the hood

fit() computes and stores the mean and deviations of all input features,
per class along with the class priors, number of classes, class IDs,
etc.

predict() computes the joing log-likelihood function and assigns the
class to the one with the maximum value. Include mathematical
formulation for Gaussian case.

Flexible Naive Bayes
~~~~~~~~~~~~~~~~~~~~

Discrete Random Variables
-------------------------

-  Bernoulli
-  Binomial
-  Multinomial
-  Multinomial with Binary features

Multinomial Naive Bayes
~~~~~~~~~~~~~~~~~~~~~~~

Assumes the likelihood to be a multinomial distribution.

Describe the binomial and multinomial distribution and derivation of the
discriminating function.

--------------

Naive Insights
==============

Power Despite Naivete
---------------------

So why does the NB perform so well? A few reasons.

-  In most applications, only the decision surface matters
-  NB can produce complex, nonlinear decision boundaries and can hence
   generate elaborate fits
-  Feature engineering and related variable selection methods applied to
   the data beforehand can make the independence assumption not too
   detrimental
-  Complexity of n-univariate likelihood distributions is far lower than
   a single n-variate multivariate distribution

Pros and Cons
-------------

================================================================================================= ========================================
Pros                                                                                              Cons
================================================================================================= ========================================
Fast, intuitive, easy to build, Non-iterative                                                     Independence assumption is not practical
Does surprisingly well despite assumptions                                                        See what I did there? ;-)
Useful in higher dimensions where the independence assumption is more likely to hold             
Interpretable - the weights of evidence reveals individual feature contribution to the prediction
Can create nonlinear decision boundaries & complex models                                        
Very few tunable parameters                                                                       Very few tunable parameters :-/
================================================================================================= ========================================

Despite the cons, NB is a quick way to get a baseline for comparison
with and improving other models.

**Note on Bias-Variance Trade-off for NB**

--------------

References
==========

Book Chapters
-------------

-  Chapter 2 from Richard O. Duda, Peter E. Hart, and David G. Stork.
   2000. *Pattern Classification* (2nd Edition). Wiley-Interscience,
   USA.
-  Chapter 1 from Christopher M. Bishop. 2006. *Pattern Recognition and
   Machine Learning* (Information Science and Statistics).
   Springer-Verlag, Berlin, Heidelberg.
-  Chapter 9 from Xindong Wu and Vipin Kumar. 2009. *The Top Ten
   Algorithms in Data Mining (1st. ed.)*. Chapman & Hall/CRC.
-  Ch. 3 of Tom Mitchell's book on ML - Generative vs. Discriminant
   Classifiers: NB and Logistic Regression
-  Introduction to Information Retrieval - Ch. 13
-  NLTK With Python `online <http://www.nltk.org/book/>`__.
-  Ch. 4 NB and Sentiment Analysis from Speech and Language Processing
   text

Implementations (Python specific):
----------------------------------

-  Scikit-Learn `Naive
   Bayes <https://scikit-learn.org/stable/modules/naive_bayes.html>`__

Online Tutorial & Posts
-----------------------

-  DONE 2020-04-16 `In Depth: Naive Bayes
   Classification <https://jakevdp.github.io/PythonDataScienceHandbook/05.05-naive-bayes.html>`__,
   Python Data Science Handbook, Jake VanderPlas
-  DONE 2020-04-16 Scikit-Learn Tutorial on `Working with Text
   Data <https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html>`__
   (contains skeleton code for exercises)
-  Sebastian Raschka on `Naive Bayes and Text
   Classification <https://sebastianraschka.com/Articles/2014_naive_bayes_1.html>`__
-  Will Kurt on `Logistic Regression and Bayes
   Theorem <https://www.countbayesie.com/blog/2019/6/12/logistic-regression-from-bayes-theorem>`__.
   This site also contains other interesting posts on probability theory
   and related concepts
-  `Naive Bayes
   Classifier <https://www.python-course.eu/naive_bayes_classifier_introduction.php>`__
   on Python-Course.eu site, implementation from scratch

Additional Resources
--------------------

-  Tutorial `Deep Learning for NLP (without
   magic) <https://www.socher.org/index.php?n=DeepLearningTutorial.DeepLearningTutorial>`__
-  Fast.ai Course on `Natural Language
   Processing <https://github.com/fastai/course-nlp>`__
-  Stanford Course `NLP with Deep
   Learning <https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/index.html>`__
-  Stanford Course (undergrad level) - `From Language to
   Information <https://web.stanford.edu/class/cs124/>`__
-  Christopher D. Manning's Courses on Natural Language Processing
   `listed here <https://nlp.stanford.edu/manning/>`__
-  Google `Ngram Viewer <https://books.google.com/ngrams>`__

Spam Filtering
--------------

Paul Graham `A Plan for Spam <http://www.paulgraham.com/spam.html>`__
