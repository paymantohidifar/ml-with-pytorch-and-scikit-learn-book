# Bonus materials for Chapter 3

## **Decision Trees**

### **How do bias/variance trade-off play out in Random Forests?**

In Random Forests, when we average the predictions of multiple highly complex, unconstrained decision trees, we are exploiting statistical aggregation to systematically damp down variance.

Suppose we have an ensemble of $B$ identical, identically distributed (i.d.) trees, each with a variance of $\sigma^2$. If these trees were completely independent, the variance of their average would simply be $\frac{\sigma^2}{B}$, which approaches zero as $B \to \infty$.

* **Effect on variance:** Because the trees are trained on the same underlying dataset, they are correlated. If the positive pairwise correlation between any two trees is $\rho$, the exact analytical variance of the ensemble average is:

    $$\text{Var}\left(\frac{1}{B}\sum_{i=1}^{B} X_i\right) = \rho \sigma^2 + \frac{1-\rho}{B}\sigma^2$$

    So, as $B$ (the number of trees) increases, the second term $\frac{1-\rho}{B}\sigma^2$ shrinks toward zero and the variance of the ensemble plateaus at $\rho \sigma^2$. The takeaway is that the final variance is strictly bound by $\rho$ (how correlated the trees are) and averaging drastically reduces variance compared to an individual tree, provided the trees aren't perfectly correlated ($\rho < 1$).

* **Effect on bias:** Because the final prediction is a simple linear combination (an average) of the individual expected values, the bias of the ensemble is identical to the bias of a single individual tree. In practice, because bootstrap sampling (in RF) slightly limits the data choices available to any single tree, individual trees cannot fit the training data *quite* as perfectly as a single standard decision tree. Therefore, individual tree bias increases slightly, but this minor sacrifice is heavily outweighed by the massive reduction in variance.

### **Extremely Randomized Trees (or Extra Trees)?**

Extremely Randomized Trees (commonly referred to as Extra Trees or ERF) is a tree-based ensemble learning algorithm designed by Pierre Geurts, Damien Ernst, and Louis Wehenkel in 2006. From a first-principles perspective, Extra Trees is a variant of Random Forests built to push the Bias-Variance Trade-off to an extreme. It purposefully sacrifices individual tree accuracy (increasing individual bias) to achieve a massive reduction in the overall ensemble variance.

To understand Extra Trees, it is best to compare its core mechanics directly against a standard Random Forest. Both algorithms build an ensemble of independent decision trees and average their predictions, but Extra Trees introduces a different randomization strategy.

* **Level 1 randomization: Data sampling**
    * **Random Forest:** Uses Bootstrap Aggregation (Bagging). Each tree is trained on a unique, randomly sampled subset of rows (roughly 63.2% of the original dataset), which naturally de-correlates the trees.
    * **Extra Trees:** By default, each tree uses the entire, raw training dataset (`bootstrap=False`). It does not rely on row-level sampling to create diversity.

* **Level 2 randomization: Node splitting**
    * **Random Forest:** For each of those $m$ features, the algorithm mathematically scans through *every single possible split point* in the data to find the exact threshold that maximizes information gain (e.g., minimizing Gini Impurity or Mean Squared Error). This is a greedy, locally optimized split.
    * **Extra Trees:** Instead of hunting for the best possible threshold, the algorithm draws a completely random split point (uniformly distributed between the feature's minimum and maximum values) for each of the $m$ features. It then evaluates those $m$ random thresholds and picks the best one among them.

#### **Bias/variance trade-off in ERF models**

As we describe in the previous bonus topic, the variance of an ensemble average of $B$ correlated trees can be calculated using the following equation:

$$\text{Var}(\text{Ensemble}) = \rho \sigma^2 + \frac{1-\rho}{B}\sigma^2$$

Where $\sigma^2$ is the variance of an individual tree and $\rho$ is the correlation between any two trees.

Because Random Forests look for the absolute optimal mathematical boundaries, trees will naturally choose similar split thresholds if they share features. By choosing thresholds completely at random, Extra Trees forces the structures of the trees to be radically different from one another, driving $\rho$ close to zero.

Additionally, random split thresholds act as a heavy structural regularizer. Individual trees are prevented from perfectly micro-carving the feature space to fit highly localized noise.

On the bias side, however, the outcome is a little different. Because the split points are randomized rather than optimized, individual trees are structurally weaker and have higher bias than Random Forest trees. However, when we average hundreds of these heavily de-correlated trees together, the reduction in variance completely overrides the slight bias increase, leading to excellent generalization performance.

#### **Practical engineering advantages**

* **Massive speed up in training latency:** The most computationally expensive step in tree-based modeling is sorting the data at every single node to find the optimal mathematical split point. Because Extra Trees bypasses this search entirely by picking a random number, Extra Trees trains significantly faster than Random Forests or Gradient Boosted Trees, especially on datasets with high numbers of continuous features.

* **Robustness to overfitting on continuous noise:** If our dataset contains features with smooth, noisy continuous variables, a standard Random Forest will meticulously fit splits directly to those tiny variations. Extra Trees' random thresholds smoothly slice through the noise, making the model highly robust to variance in continuous data.

#### **When to use ERF in production**

* **Use Extra Trees if:** You are working with high-dimensional tabular data, your compute/time budget for training is highly constrained, and your baseline Random Forest is severely overfitting (exhibiting high variance).
* **Avoid Extra Trees if:** Your features are mostly sparse, low-cardinality categorical encodings (where drawing a random threshold uniformly doesn't provide meaningful architectural diversity) or if your dataset is so small that the algorithm's high bias tax prevents it from learning the true underlying signal.

In scikit-learn, we can easily deploy this architecture using `ExtraTreesClassifier` or `ExtraTreesRegressor`.

---