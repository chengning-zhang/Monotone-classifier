# Monotone Classifier(MC)

A custom implementation of `Monotone Classifier` in Python 3 that safely interact with scikit-learn Pipelines and model selection tools.
We develop a statistical framework for training and evaluating binary classification rules that leverage suitable monotonic relationship between the
predictors and response. `Monotone Classifier` is motivated by problems in diagnostic medicine,
where multiple input variables, whether it be quantitative biomarkers, composite clinical scores,
or ordinal assessments by radiologists, are often monotonically associated with the probability of
the underlying disease. 

##

<em class="property">class </em></code><code class="sig-name descname">MonotoneClassifier</code>
<span class="sig-paren">
  (
  </span>
  <em class="sig-param"><span class="n">tol</span><span class="o">=</span><span class="default_value">1e-7</span></em>, 
  <em class="sig-param"><span class="n">max_iter</span><span class="o">=</span><span class="default_value">300</span></em>, 
  <em class="sig-param"><span class="n">verbose</span><span class="o">=</span><span class="default_value">False</span></em>, 
  <em class="sig-param"><span class="n">handle</span><span class="o">=</span><span class="default_value">'missing'</span></em>, 
  <em class="sig-param"><span class="n">random_init</span><span class="o">=</span><span class="default_value">True</span></em>, 
  <em class="sig-param"><span class="n">init_low</span><span class="o">=</span><span class="default_value">1</span></em>, 
  <em class="sig-param"><span class="n">init_high</span><span class="o">=</span><span class="default_value">4</span></em>
  <span class="sig-paren">
  )
</span>
¶</a></dt>


<dl class="field-list">
<dt class="field-odd">Parameters</dt>
<dd class="field-odd"><dl>
<dt><strong>tol: </strong><span class="classifier">float, default=1e-7</span></dt><dd><p>Tolerance for stopping criteria.</p>
</dd>
<dt><strong>max_iter: </strong><span class="classifier">int, default=300</span></dt><dd><p>Maximum number of iterations taken for the solvers to converge.</p>
</dd>
<dt><strong>handle: </strong><span class="classifier">{'missing', 'complete', 'mean', 'median'}, default=’missing’</span></dt><dd><p>Used to specify the method to deal with missing covariates. 'missing' use EM to handle missing X, which is consistent under MAR; 'complete' delete incomplete examples, which causes bias in MAR;
'mean' impute missing values by mean of that column; 'median' impute missing values by median of that column.</p>
</dd>
<dt><strong>verbose: </strong><span class="classifier">bool, default=False</span></dt><dd><p>whether to plot the self.Error_X_, self.Error_U_ and self.loglikelihood_ when training.</p>
</dd>
<dt><strong>random_init: </strong><span class="classifier">bool, default=True</span></dt><dd><p>Whether randomly intialize point mass of U and X.</p>
</dd>
<dt><strong>init_low: </strong><span class="classifier">float, default=1</span></dt><dd><p>Controls the range of randomly intialized point mass of U and X. Only valid when random_init is True</p>
</dd>
<dt><strong>init_high: </strong><span class="classifier">float, default=4</span></dt><dd><p>Controls the range of randomly intialized point mass of U and X. Only valid when random_init is True</p>
</dd>
</dl>
</dd>

<dl class="field-list">
<dt class="field-odd">Attributes</dt>
<dd class="field-odd"><dl>
<dt><strong>dict_U_: </strong><span class="classifier">dict</span></dt><dd><p>dictionary where keys are {x_1,...x_m} and values are point mass of U at x_i.</p>
</dd>
<dt><strong>dict_X_: </strong><span class="classifier">dict</span></dt><dd><p>dictionary where keys are {x_1,...x_m} and values are point mass of X at x_i.</p>
</dd>
<dt><strong>n_iter_: </strong><span class="classifier">int</span></dt><dd><p>Actual number of iterations.</p>
</dd>
<dt><strong>Error_U_: </strong><span class="classifier">list of size self.n_iter_</span></dt><dd><p>Mean sum of square of difference of dict_U_ between current iteration and previous iteration.</p>
</dd>
<dt><strong>Error_X_: </strong><span class="classifier">list of size self.n_iter_</span></dt><dd><p>Mean sum of square of difference of dict_X_ between current iteration and previous iteration.</p>
</dd>
<dt><strong>m_: </strong><span class="classifier">int</span></dt><dd><p>number of x_1....x_m, number of unique observed values of X.</p>
</dd>
<dt><strong>AIC_: </strong><span class="classifier">int</span></dt><dd><p> AIC = 2k - 2ln(L), k be the number of estimated parameters in the model.</p>
</dd>
<dt><strong>BIC_: </strong><span class="classifier">int</span></dt><dd><p> kln(n) - 2ln(L), k be the number of estimated parameters in the model.</p>
</div>
</dd>
</dl>
</dd>
</dl>


Examples
```
>>> mc = MonotoneClassifier().fit(X, y)
>>> mc.predict(X[:2, :])
array([0, 0])
>>> mc.predict_proba(X[:2, :])
array([[9.8...e-01, 1.8...e-02, 1.4...e-08],
      [9.7...e-01, 2.8...e-02, ...e-08]])
```


</pre></div>
</div>
<p class="rubric">Methods</p>
<table class="longtable docutils align-default">
<colgroup>
<col style="width: 10%" />
<col style="width: 90%" />
</colgroup>
<tbody>
<tr class="row-odd"><td><p><a class="reference internal" href="#sklearn.linear_model.LogisticRegression.fit" title="sklearn.linear_model.LogisticRegression.fit"><code class="xref py py-obj docutils literal notranslate"><span class="pre">fit</span></code></a>(X, y)</p></td>
<td><p>Fit the model according to the given training data.</p></td>
</tr>
<tr class="row-even"><td><p><a class="reference internal" href="#sklearn.linear_model.LogisticRegression.get_params" title="sklearn.linear_model.LogisticRegression.get_params"><code class="xref py py-obj docutils literal notranslate"><span class="pre">get_params</span></code></a>([deep])</p></td>
<td><p>Get parameters for this estimator.</p></td>
</tr>
<tr class="row-odd"><td><p><a class="reference internal" href="#sklearn.linear_model.LogisticRegression.predict" title="sklearn.linear_model.LogisticRegression.predict"><code class="xref py py-obj docutils literal notranslate"><span class="pre">predict</span></code></a>(X)</p></td>
<td><p>Predict class labels for samples in X.</p></td>
</tr>
<tr class="row-even"><td><p><a class="reference internal" href="#sklearn.linear_model.LogisticRegression.predict_log_proba" title="sklearn.linear_model.LogisticRegression.predict_log_proba"><code class="xref py py-obj docutils literal notranslate"><span class="pre">predict_log_proba</span></code></a>(X)</p></td>
<td><p>Predict logarithm of probability estimates.</p></td>
</tr>
<tr class="row-odd"><td><p><a class="reference internal" href="#sklearn.linear_model.LogisticRegression.predict_proba" title="sklearn.linear_model.LogisticRegression.predict_proba"><code class="xref py py-obj docutils literal notranslate"><span class="pre">predict_proba</span></code></a>(X)</p></td>
<td><p>Probability estimates.</p></td>
</tr>
<tr class="row-even"><td><p><a class="reference internal" href="#sklearn.linear_model.LogisticRegression.score" title="sklearn.linear_model.LogisticRegression.score"><code class="xref py py-obj docutils literal notranslate"><span class="pre">score</span></code></a>(X, y)</p></td>
<td><p>Return the mean accuracy on the given test data and labels.</p></td>
</tr>
<tr class="row-odd"><td><p><a class="reference internal" href="#sklearn.linear_model.LogisticRegression.set_params" title="sklearn.linear_model.LogisticRegression.set_params"><code class="xref py py-obj docutils literal notranslate"><span class="pre">set_params</span></code></a>(**params)</p></td>
<td><p>Set the parameters of this estimator.</p></td>
</tr>
</tbody>
</table>

## Built With

* [Dropwizard](https://scikit-learn.org/stable/modules/classes.html) - scikit-learn API
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds


## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Lu Mao**, **Chengning Zhang** - *Initial work* - [Monotone Classifier](https://github.com/chengning-zhang/)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
