class MonotoneClassifier():
  """
  Monotonic Classifier, which can handle missing covariates.
  
  Parameters
  ----------
  
  tol : float, default=1e-4
    Tolerance for stopping criteria.

  max_iter : int, default=100
    Maximum number of iterations taken for the solvers to converge.

  verbose : bool, default = False
    whether to plot the self.Error_X_, self.Error_U_  and self.loglikelihood_ when training 

  init_point_mass : 
    initial point mass of EM algorithm, defaults is 1/m

  Attributes
  ----------
  dict_U_: dictionary 
    dictionary where keys are {x_1,...x_m} and values are point mass of U at x_i
  
  dict_X_: dictionary 
    dictionary where keys are {x_1,...x_m} and values are point mass of X at x_i

  n_iter_ : ndarray of shape (1, )
    Actual number of iterations
  
  Error_U_: list of size n_iter_
    the sum of square of difference of dict_U_ between current iteration and previous iteration.

  Error_X_: list of size n_iter_
    the sum of square of difference of dict_X_ between current iteration and previous iteration.

  m_ : int
    number of x_1....x_m, number of unique observed values of X

  AIC_: int
    AIC = 2k - 2ln(L) ## ??? k be the number of estimated parameters in the model, is it the number of point mass or the number of predictors?

  BIC_: int
    BIC = kln(n) - 2ln(L)
  
  note: When fitting models, it is possible to increase the likelihood by adding parameters, but doing so may result in overfitting. 
        Both BIC and AIC attempt to resolve this problem by introducing a penalty term for the number of parameters in the model;
        the penalty term is larger in BIC than in AIC.

  References
  ----------

  Examples
  --------
  >>> from sklearn.datasets import load_iris
  >>> X, y = load_iris(return_X_y=True)
  >>> mc = MonotoneClassifier().fit(X, y)
  >>> mc.predict(X[:2, :])
  array([0, 0])
  >>> clf.predict_proba(X[:2, :])
  array([[9.8...e-01, 1.8...e-02, 1.4...e-08],
        [9.7...e-01, 2.8...e-02, ...e-08]])
  """  

  def __init__(self, tol=0.00001, max_iter = 100, verbose = False):
      self.tol = tol
      self.max_iter = max_iter
      self.verbose = verbose
      self._fitted = False

  def fit(self, X, Y):
      """
      Fit the model according to the given training data.
      
      Step 1: get augmented support set [x1...xm] by self._support(), which returns self.C_, self.m_, self.uniqueObservedX_
      Step 2: NPEM by self._NPEM(), which returns self.n_iter_, self.Error_U_ ,self.Error_X_ , self.dict_X_, self.dict_U_, self.log_likelihood_, self.AIC_, self.BIC_ 
      
      Parameters
      ----------
      X : {array-like, sparse matrix} of shape (n_samples, n_features)
          Training vector, where n_samples is the number of samples and
          n_features is the number of features. 
          pd.df or np.array
      
      y : array-like of shape (n_samples,)
          Target vector relative to X. pd.series or list[]

      Returns
      -------
      self
          Fitted estimator self.C_, self.m_, self.uniqueObservedX_, self.n_iter_, self.Error_U_ ,self.Error_X_ , self.dict_X_, self.dict_U_, self.log_likelihood_, self.AIC_, self.BIC_ 
      """

      try:
        X = X.to_numpy() # np.array is more efficient than list
        Y = Y.to_numpy()
      except:
          pass
      
      # get augmented support 
      self._support(X,Y) 

      # NPEM training
      hashmap_U = dict(zip(self.uniqueObservedX_, [1/self.m_]*(self.m_) )) # initial of EM
      hashmap_X = dict(zip(self.uniqueObservedX_, [1/self.m_]*(self.m_) ))
      ## NPEM to train point mass on m points
      self._NPEM(hashmap_U, hashmap_X, X, Y)

      #
      # after training.
      if self.n_iter_ >= self.max_iter and (self.Error_U_[-1] > self.tol or self.Error_X_[-1] > self.tol):
        warnings.warn('training iterations exceeds Max iteration and SSE greater than tolerance!') 


      # plot the training process
      if self.verbose:
          self._plot()

      return self # mc = class().fit, otherwise mc is none


  def _NPEM(self, hashmap_U, hashmap_X, X, Y):
      """
      helper function of self.fit for NPMLE
      Parameters
      ----------
      hashmap_U : dictionary of size self.m_
        initial probability point mass of U on x_1,...x_m

      hashmap_X : dictionary of size self.m_
        initial probability point mass of X on x_1,...x_m
      
      X : 2D np.array([[],[],...[]])
        Training X where X has been converted to np.array in self.fit
      
      Y : 1D np.array([])
      
      Returns
      -------
      self
        Fitted estimator self.n_iter_, self.Error_U_ ,self.Error_X_ , self.dict_X_, self.dict_U_, self.log_likelihood_, self.AIC_, self.BIC_ 
      """
      n = len(X)
      prev_U = hashmap_U 
      prev_X = hashmap_X
      iter = 0
      Error_U = []
      Error_X = []
      Likelihood= []
      log_likelihood = []
      while True:
          # R_j is the cdf(xi), uniqueObservedX_ includes all fully observed X and augmented X. 
          R_j = {ele: self._CDF(ele, prev_U) for ele in self.uniqueObservedX_}; #print(R_j); print(prev_U);print(prev_X)
          #log_likelihood
          log_likelihood.append(self._likelihood(R_j, prev_U, prev_X, X, Y, iter))
          # 
          cur_U = prev_U.copy()
          cur_X = prev_X.copy()
          for xk in cur_U:
              item1 = 0 # for cumulative sum, R(dxk)
              item2 = 0 # for qk
              for i,xi in enumerate(X):
                  # U Point mass
                  # if xi is complete
                  if not any(np.isnan(xi)):
                      if Y[i] == 1 and self._compare(xk,xi):
                          item1 += 1/ R_j[tuple(xi)]

                      if Y[i] == 0 and not self._compare(xk,xi):
                          item1 += 1/ (1-R_j[tuple(xi)])
                  # else xi is missing
                  else:
                      if Y[i] == 1:
                        numerator = sum([prev_X[xl] if self._compare(xk, xl) else 0 for xl in self.C_[self._hashable_x(xi)]])
                        denominator = sum([prev_X[xl]* R_j[xl] for xl in self.C_[self._hashable_x(xi)] ])
                        item1 += numerator/ denominator
                  
                      if Y[i] == 0:
                        numerator = sum([prev_X[xl] if not self._compare(xk, xl) else 0 for xl in self.C_[self._hashable_x(xi)]])
                        denominator = sum([prev_X[xl]* (1-R_j[xl]) for xl in self.C_[self._hashable_x(xi)] ])
                        item1 += numerator/ denominator
              
                  # X Point mass
                  # if xi is complete
                  if not any(np.isnan(xi)):
                      if all(xk == xi): # xk tuple, xi array
                          item2 += 1

                  # else xi is missing
                  else:
                      if xk in set(self.C_[self._hashable_x(xi)]) and Y[i] == 1:
                          numerator = prev_X[xk] * R_j[xk]
                          denominator = sum([prev_X[xl]* R_j[xl] for xl in self.C_[self._hashable_x(xi)] ]  )
                          item2 += numerator/ denominator


                      if xk in set(self.C_[self._hashable_x(xi)]) and Y[i] == 0:
                          numerator = prev_X[xk] * (1- R_j[xk])
                          denominator = sum([prev_X[xl]* (1-R_j[xl]) for xl in self.C_[self._hashable_x(xi)] ]  )
                          item2 += numerator/ denominator
              #
              cur_U[xk] = prev_U[xk]* item1 /n 
              cur_X[xk] = item2 /n
      
          iter +=1
          error1 = sum(( np.array(list(cur_U.values())) - np.array(list(prev_U.values())) )**2)
          error2 = sum(( np.array(list(cur_X.values())) - np.array(list(prev_X.values())) )**2)
          Error_U.append(error1); Error_X.append(error2)
          if (error1 < self.tol and error2 < self.tol) or iter > self.max_iter:
              break

          prev_U = cur_U; prev_X = cur_X

      self.n_iter_ = iter
      self.Error_U_ = Error_U
      self.Error_X_ = Error_X
      self.dict_X_ = cur_X
      self.dict_U_ = cur_U   
      self.log_likelihood_ = log_likelihood
      self.AIC_ = 2* self.m_ - 2*self.log_likelihood_[-1]
      self.BIC_ = self.m_*np.log(n) - 2*self.log_likelihood_[-1]
      self._fitted = True

  def _likelihood(self, R_j, prev_U, prev_X, X, Y, iter):
      """
      helper function of fit to get the log likelihood at current iteration. Should be strictly increasing.
      
      Parameters
      ----------
      R_j : dict 
        The cdf of at current iteration j

      prev_U : dict
        The point mass of U at current iteration j

      prev_X : dict
        The point mass of X at current iteration j

      X : 2D np.array

      Y : 1D np.array
      
      iter : int
        The jth iteration

      Returns
      -------
      ans : float
        The log likelihood at current iteration. 

      """
      n = len(X)
      ans = 0
      for i,xi in enumerate(X):
          if Y[i] == 1:
              if not any(np.isnan(xi)): # if xi is complete
                add_on = prev_X[tuple(xi)]* R_j[tuple(xi)]; 
              else:
                add_on =  sum([prev_X[xl]* R_j[xl] for xl in self.C_[self._hashable_x(xi)]  ] ) 
          
          elif Y[i] == 0:
              if not any(np.isnan(xi)): # if xi is complete
                add_on = prev_X[tuple(xi)]* (1-R_j[tuple(xi)]) 
              else:
                add_on = sum([prev_X[xl]* (1-R_j[xl]) for xl in self.C_[self._hashable_x(xi)]  ] ) 
          
          if add_on <= 0:
            print('At %s iteration, the %s example result in nan, Y[i] is %s, xi is %s missing' % (iter, i, Y[i], any(np.isnan(xi))))
            return np.nan

          else:
            ans += np.log(add_on)

      return ans/n



  def _support(self, X, Y):
      """
      helper function of self.fit to get support of X and U.
      1. Use all fully observed Xi. 2. Augment partially observed Xi and add to the support set.

      Parameters
      ----------
      X : np.array
      Y : np.array

      Returns
      -------
      self
        Fitted estimator self.C_, self.m_, self.uniqueObservedX_

      """
      # idx1 = [i for i,ele in enumerate(Y) if ele == 1]
      uniqueObservedX_arr = None # useful when augment the support set. need unique value of certain col
      uniqueObservedX = set()
      # get fully observed x
      for i,x in enumerate(X):
          if not any( np.isnan(x) ): #and i in idx1: # fully observed, and Y ==1, to make sure same support when there is no missing covariates
            uniqueObservedX.add(tuple(x))
            if uniqueObservedX_arr is None:
              uniqueObservedX_arr = x
            else:
              uniqueObservedX_arr = np.vstack((uniqueObservedX_arr, x))
    
      # now uniqueObservedX save unique values of X which is fully observed.
      # augment the support set
      C = defaultdict(list) # key: x with missing values,  value: potential xl
      for x in X:
        if any(np.isnan(x)) and self._hashable_x(x) not in C: # x with missing values and this missing pattern not seen so far.
          missing_col = np.where(np.isnan(x) == True )[0] # array([...])
          augment_x = x.copy().reshape(1, -1) # copy!!! otherwise change X itself!!# initialize. after loop, augment_x[l] is xl. For example, x = [1,2,nan, nan]. augment_x = [[1,2,1,1],[1,2,1,2], [1,2,2,1], [1,2,2,2] ]
          #
          for j in range(len(missing_col)):
              col_idx = missing_col[j]
              augment_x_new = None # partial result of augment_x, for example after loops augment_x_new = [[1,2,1, nan], [1,2,2,nan]]
              #
              for x_new in augment_x:
                for col_unique_values in np.unique( uniqueObservedX_arr[:, col_idx] ):
                  x_new[col_idx] = col_unique_values
                  if augment_x_new is None:
                    augment_x_new = x_new.copy()
                  else:
                    augment_x_new = np.vstack((augment_x_new, x_new.copy() )) # is copy needed?
              #
              augment_x = augment_x_new
          
          # augment_x should be added to support
          for xl in augment_x:
            uniqueObservedX.add(tuple(xl))
            C[self._hashable_x(x)].append(tuple(xl))

      m = len(uniqueObservedX) ## U is supported on those m points
      self.m_ = m
      self.uniqueObservedX_ = uniqueObservedX
      self.C_ = C 

  def _hashable_x(self, x):
      """
      tuple(np.array([1, np.nan])) is hashable but nan cause duplicate keys, replace np.nan with 'nan'
      
      Parameters
      ----------
      x : np.array() with np.nan 
      
      # missing_col : index of columns which has missing values

      Returns
      -------

      x_tuple : tuple with np.nan replaced with 'nan'
      """   
      return tuple(x.astype('|S5'))
       

  def predict_proba(self, X):
      """
      Probability estimates.

      Parameters
      ----------
      X : array-like of shape (n_samples, n_features)
          Vector to be scored, where `n_samples` is the number of samples and
          `n_features` is the number of features. pd.df or list[list]
      Return
      -------
      T : array-like of shape (n_samples,) np.array(list())
      """
      if not self._fitted:
        raise ValueError('The object has not been fitted yet.')

      try:
          X = X.to_numpy()
      except:
          pass
      
      return np.array([self._CDF(x, self.dict_U_) for x in X])


  def predict(self, X):
      """
      label estimates.

      Parameters
      ----------
      X : array-like of shape (n_samples, n_features)
          Vector to be scored, where `n_samples` is the number of samples and
          `n_features` is the number of features.
      Return
      -------
      T : array-like of shape (n_samples,)
      """
      prob = self.predict_proba(X)
      return np.array([1 if p > 0.5 else 0 for p in prob]) 
  
  
  def _CDF(self, x, hashmap):
      """
      helper function to calculate the CDF at x given point mass hashmap, i.e sum_{x_k <= x} P(U = xk)

      note : it is possible that ans > 1 due to computation issues, (5,5): 1.00000002, force it to 1.
      Parameters
      ----------
      x : list or tuple

      hashmap : dictionary of size m
        Given point mass for calculating CDF

      Returns
      -------
      ans : float
        CDF at x, sum_{x_k <= x} P(U = xk)
      """

      ans = 0
      for k,v in hashmap.items():
          if self._compare(k,x):
            ans+= v

      return ans if ans<=1 else 1 # it is possible that ans > 1 due to computation issues, (5,5): 1.00000002


  def _compare(self, x1, x2):
      """
      helper function to compare whether x1 <= x2 for all elements.
      Parameters
      ----------
      x1 : list or tuple
      
      x2 : list or tuple
      
      Returns
      -------
      ans : bool
      """
      if len(x1) != len(x2):
        raise ValueError('x1 and x2 different dimentions')

      ans = []
      for i in range(len(x1)):
          ans.append(x1[i] <= x2[i])
      
      return all(ans)

  def _plot(self):
      """
      helper function plot the training process 

      """
      try:
        plt.plot(list(range(1,self.n_iter_+1)), self.Error_U_, '-o')
        plt.title("")
        plt.xlabel("num of iteration")
        plt.ylabel("SSE of latent variable U")
        xint = range( 2, self.n_iter_ + 1, int(self.n_iter_/10) )
        plt.xticks(xint)
        plt.show()
      except:
        plt.plot(list(range(1,self.n_iter_+1)), self.Error_U_, '-o')
        plt.title("")
        plt.xlabel("num of iteration")
        plt.ylabel("SSE of latent variable U")
        plt.show()

      try:
        plt.plot(list(range(1,self.n_iter_+1)), self.Error_X_, '-o')
        plt.title("")
        plt.xlabel("num of iteration")
        plt.ylabel("SSE of X")
        xint = range( 2, self.n_iter_ + 1, int(self.n_iter_/10) )
        plt.xticks(xint)
        plt.show()
      except:
        plt.plot(list(range(1,self.n_iter_+1)), self.Error_X_, '-o')
        plt.title("")
        plt.xlabel("num of iteration")
        plt.ylabel("SSE of X")
        plt.show()

      try:
        plt.plot(list(range(1,self.n_iter_+1)), self.log_likelihood_, '-o')
        plt.title("")
        plt.xlabel("num of iteration")
        plt.ylabel("Observed log likelihood")
        xint = range( 2, self.n_iter_ + 1, int(self.n_iter_/10) )
        plt.xticks(xint)
        plt.show()
      except:
        plt.plot(list(range(1,self.n_iter_+1)), self.log_likelihood_, '-o')
        plt.title("")
        plt.xlabel("num of iteration")
        plt.ylabel("Observed log likelihood")
        plt.show()

