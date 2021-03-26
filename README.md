# Monotone Classifier(MC)

A custom implementation of `Monotone Classifier` in Python 3 that safely interact with scikit-learn Pipelines and model selection tools.
We develop a statistical framework for training and evaluating binary classification rules that leverage suitable monotonic relationship between the
predictors and response. `Monotone Classifier` is motivated by problems in diagnostic medicine,
where multiple input variables, whether it be quantitative biomarkers, composite clinical scores,
or ordinal assessments by radiologists, are often monotonically associated with the probability of
the underlying disease. 



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
<dt><strong>handle</strong><span class="classifier">{'missing', 'complete', 'mean', 'median'}, default=’missing’</span></dt><dd><p>Used to specify the method to deal with missing covariates. 'missing' use EM to handle missing X, which is consistent under MAR; 'complete' delete incomplete examples, which causes bias in MAR;
'mean' impute missing values by mean of that column; 'median' impute missing values by median of that column</p>
<div class="versionadded">
<p><span class="versionmodified added">New in version 0.19: </span>l1 penalty with SAGA solver (allowing ‘multinomial’ + L1)</p>
</div>
</dd>
<dt><strong>dual</strong><span class="classifier">bool, default=False</span></dt><dd><p>Dual or primal formulation. Dual formulation is only implemented for
l2 penalty with liblinear solver. Prefer dual=False when
n_samples &gt; n_features.</p>
</dd>
<dt><strong>tol</strong><span class="classifier">float, default=1e-4</span></dt><dd><p>Tolerance for stopping criteria.</p>
</dd>
<dt><strong>C</strong><span class="classifier">float, default=1.0</span></dt><dd><p>Inverse of regularization strength; must be a positive float.
Like in support vector machines, smaller values specify stronger
regularization.</p>
</dd>
<dt><strong>fit_intercept</strong><span class="classifier">bool, default=True</span></dt><dd><p>Specifies if a constant (a.k.a. bias or intercept) should be
added to the decision function.</p>
</dd>
<dt><strong>intercept_scaling</strong><span class="classifier">float, default=1</span></dt><dd><p>Useful only when the solver ‘liblinear’ is used
and self.fit_intercept is set to True. In this case, x becomes
[x, self.intercept_scaling],
i.e. a “synthetic” feature with constant value equal to
intercept_scaling is appended to the instance vector.
The intercept becomes <code class="docutils literal notranslate"><span class="pre">intercept_scaling</span> <span class="pre">*</span> <span class="pre">synthetic_feature_weight</span></code>.</p>
<p>Note! the synthetic feature weight is subject to l1/l2 regularization
as all other features.
To lessen the effect of regularization on synthetic feature weight
(and therefore on the intercept) intercept_scaling has to be increased.</p>
</dd>
<dt><strong>class_weight</strong><span class="classifier">dict or ‘balanced’, default=None</span></dt><dd><p>Weights associated with classes in the form <code class="docutils literal notranslate"><span class="pre">{class_label:</span> <span class="pre">weight}</span></code>.
If not given, all classes are supposed to have weight one.</p>
<p>The “balanced” mode uses the values of y to automatically adjust
weights inversely proportional to class frequencies in the input data
as <code class="docutils literal notranslate"><span class="pre">n_samples</span> <span class="pre">/</span> <span class="pre">(n_classes</span> <span class="pre">*</span> <span class="pre">np.bincount(y))</span></code>.</p>
<p>Note that these weights will be multiplied with sample_weight (passed
through the fit method) if sample_weight is specified.</p>
<div class="versionadded">
<p><span class="versionmodified added">New in version 0.17: </span><em>class_weight=’balanced’</em></p>
</div>
</dd>
<dt><strong>random_state</strong><span class="classifier">int, RandomState instance, default=None</span></dt><dd><p>Used when <code class="docutils literal notranslate"><span class="pre">solver</span></code> == ‘sag’, ‘saga’ or ‘liblinear’ to shuffle the
data. See <a class="reference internal" href="../../glossary.html#term-random_state"><span class="xref std std-term">Glossary</span></a> for details.</p>
</dd>
<dt><strong>solver</strong><span class="classifier">{‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’},             default=’lbfgs’</span></dt><dd><p>Algorithm to use in the optimization problem.</p>
<ul class="simple">
<li><p>For small datasets, ‘liblinear’ is a good choice, whereas ‘sag’ and
‘saga’ are faster for large ones.</p></li>
<li><p>For multiclass problems, only ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’
handle multinomial loss; ‘liblinear’ is limited to one-versus-rest
schemes.</p></li>
<li><p>‘newton-cg’, ‘lbfgs’, ‘sag’ and ‘saga’ handle L2 or no penalty</p></li>
<li><p>‘liblinear’ and ‘saga’ also handle L1 penalty</p></li>
<li><p>‘saga’ also supports ‘elasticnet’ penalty</p></li>
<li><p>‘liblinear’ does not support setting <code class="docutils literal notranslate"><span class="pre">penalty='none'</span></code></p></li>
</ul>
<p>Note that ‘sag’ and ‘saga’ fast convergence is only guaranteed on
features with approximately the same scale. You can
preprocess the data with a scaler from sklearn.preprocessing.</p>
<div class="versionadded">
<p><span class="versionmodified added">New in version 0.17: </span>Stochastic Average Gradient descent solver.</p>
</div>
<div class="versionadded">
<p><span class="versionmodified added">New in version 0.19: </span>SAGA solver.</p>
</div>
<div class="versionchanged">
<p><span class="versionmodified changed">Changed in version 0.22: </span>The default solver changed from ‘liblinear’ to ‘lbfgs’ in 0.22.</p>
</div>
</dd>
<dt><strong>max_iter</strong><span class="classifier">int, default=100</span></dt><dd><p>Maximum number of iterations taken for the solvers to converge.</p>
</dd>
<dt><strong>multi_class</strong><span class="classifier">{‘auto’, ‘ovr’, ‘multinomial’}, default=’auto’</span></dt><dd><p>If the option chosen is ‘ovr’, then a binary problem is fit for each
label. For ‘multinomial’ the loss minimised is the multinomial loss fit
across the entire probability distribution, <em>even when the data is
binary</em>. ‘multinomial’ is unavailable when solver=’liblinear’.
‘auto’ selects ‘ovr’ if the data is binary, or if solver=’liblinear’,
and otherwise selects ‘multinomial’.</p>
<div class="versionadded">
<p><span class="versionmodified added">New in version 0.18: </span>Stochastic Average Gradient descent solver for ‘multinomial’ case.</p>
</div>
<div class="versionchanged">
<p><span class="versionmodified changed">Changed in version 0.22: </span>Default changed from ‘ovr’ to ‘auto’ in 0.22.</p>
</div>
</dd>
<dt><strong>verbose</strong><span class="classifier">int, default=0</span></dt><dd><p>For the liblinear and lbfgs solvers set verbose to any positive
number for verbosity.</p>
</dd>
<dt><strong>warm_start</strong><span class="classifier">bool, default=False</span></dt><dd><p>When set to True, reuse the solution of the previous call to fit as
initialization, otherwise, just erase the previous solution.
Useless for liblinear solver. See <a class="reference internal" href="../../glossary.html#term-warm_start"><span class="xref std std-term">the Glossary</span></a>.</p>
<div class="versionadded">
<p><span class="versionmodified added">New in version 0.17: </span><em>warm_start</em> to support <em>lbfgs</em>, <em>newton-cg</em>, <em>sag</em>, <em>saga</em> solvers.</p>
</div>
</dd>
<dt><strong>n_jobs</strong><span class="classifier">int, default=None</span></dt><dd><p>Number of CPU cores used when parallelizing over classes if
multi_class=’ovr’”. This parameter is ignored when the <code class="docutils literal notranslate"><span class="pre">solver</span></code> is
set to ‘liblinear’ regardless of whether ‘multi_class’ is specified or
not. <code class="docutils literal notranslate"><span class="pre">None</span></code> means 1 unless in a <a class="reference external" href="https://joblib.readthedocs.io/en/latest/parallel.html#joblib.parallel_backend" title="(in joblib v1.1.0.dev0)"><code class="xref py py-obj docutils literal notranslate"><span class="pre">joblib.parallel_backend</span></code></a>
context. <code class="docutils literal notranslate"><span class="pre">-1</span></code> means using all processors.
See <a class="reference internal" href="../../glossary.html#term-n_jobs"><span class="xref std std-term">Glossary</span></a> for more details.</p>
</dd>
<dt><strong>l1_ratio</strong><span class="classifier">float, default=None</span></dt><dd><p>The Elastic-Net mixing parameter, with <code class="docutils literal notranslate"><span class="pre">0</span> <span class="pre">&lt;=</span> <span class="pre">l1_ratio</span> <span class="pre">&lt;=</span> <span class="pre">1</span></code>. Only
used if <code class="docutils literal notranslate"><span class="pre">penalty='elasticnet'</span></code>. Setting <code class="docutils literal notranslate"><span class="pre">l1_ratio=0</span></code> is equivalent
to using <code class="docutils literal notranslate"><span class="pre">penalty='l2'</span></code>, while setting <code class="docutils literal notranslate"><span class="pre">l1_ratio=1</span></code> is equivalent
to using <code class="docutils literal notranslate"><span class="pre">penalty='l1'</span></code>. For <code class="docutils literal notranslate"><span class="pre">0</span> <span class="pre">&lt;</span> <span class="pre">l1_ratio</span> <span class="pre">&lt;1</span></code>, the penalty is a
combination of L1 and L2.</p>
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
