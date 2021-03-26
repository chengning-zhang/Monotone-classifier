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
  <em class="sig-param"><span class="n">tol</span><span class="o">=</span><span class="default_value">1e-6</span></em>, 
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


<dt class="field-even">Attributes</dt>
<dd class="field-even"><dl>
<dt><strong>classes_</strong><span class="classifier">ndarray of shape (n_classes, )</span></dt><dd><p>A list of class labels known to the classifier.</p>
</dd>
<dt><strong>coef_</strong><span class="classifier">ndarray of shape (1, n_features) or (n_classes, n_features)</span></dt><dd><p>Coefficient of the features in the decision function.</p>
<p><code class="docutils literal notranslate"><span class="pre">coef_</span></code> is of shape (1, n_features) when the given problem is binary.
In particular, when <code class="docutils literal notranslate"><span class="pre">multi_class='multinomial'</span></code>, <code class="docutils literal notranslate"><span class="pre">coef_</span></code> corresponds
to outcome 1 (True) and <code class="docutils literal notranslate"><span class="pre">-coef_</span></code> corresponds to outcome 0 (False).</p>
</dd>
<dt><strong>intercept_</strong><span class="classifier">ndarray of shape (1,) or (n_classes,)</span></dt><dd><p>Intercept (a.k.a. bias) added to the decision function.</p>
<p>If <code class="docutils literal notranslate"><span class="pre">fit_intercept</span></code> is set to False, the intercept is set to zero.
<code class="docutils literal notranslate"><span class="pre">intercept_</span></code> is of shape (1,) when the given problem is binary.
In particular, when <code class="docutils literal notranslate"><span class="pre">multi_class='multinomial'</span></code>, <code class="docutils literal notranslate"><span class="pre">intercept_</span></code>
corresponds to outcome 1 (True) and <code class="docutils literal notranslate"><span class="pre">-intercept_</span></code> corresponds to
outcome 0 (False).</p>
</dd>
<dt><strong>n_iter_</strong><span class="classifier">ndarray of shape (n_classes,) or (1, )</span></dt><dd><p>Actual number of iterations for all classes. If binary or multinomial,
it returns only 1 element. For liblinear solver, only the maximum
number of iteration across all classes is given.</p>
<div class="versionchanged">
<p><span class="versionmodified changed">Changed in version 0.20: </span>In SciPy &lt;= 1.0.0 the number of lbfgs iterations may exceed
<code class="docutils literal notranslate"><span class="pre">max_iter</span></code>. <code class="docutils literal notranslate"><span class="pre">n_iter_</span></code> will now report at most <code class="docutils literal notranslate"><span class="pre">max_iter</span></code>.</p>
</div>
</dd>
</dl>
</dd>
</dl>
