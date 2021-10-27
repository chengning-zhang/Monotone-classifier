class MonotoneClassifier():
    def __init__(self, tol = 1e-04, max_iter = 300, verbose = False):
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        self._fitted = False

    def fit(self, X, Y):
        try:
          X = X.to_numpy() 
          Y = Y.to_numpy()
        except:
          pass
        
        # step 0: get augmented support 
        M_n_m, unique_X_arr, unique_X_dic, idx_lst = self._support(X)
        
        # step 1: underset matrix, A_m_m = I(xk <= xl )
        A_m_m = self._underset(unique_X_arr);

        # step 2: useful matrix from A_m_m and M_n_m
        designM_left, designM_right = self._designMatrix(M_n_m, A_m_m)

        # 
        m = unique_X_arr.shape[0]
        R_d_U = np.array([1/m]*m); R_d_X = np.array([1/m]*m)  # 1d np.array ,initial of EM
        # step 3 iteration using M_n_m, A_m_m, designM_left, designM_right, Y
        self._NPEM(R_d_U, R_d_X, M_n_m, A_m_m, designM_left, designM_right, Y, unique_X_arr)

        # after training.
        if self.n_iter_ >= self.max_iter and (self.log_likelihood_[-1] - self.log_likelihood_[-2] > self.tol):   
            warnings.warn('training iterations exceeds Max iteration and difference of log likelihood still greater than tolerance!')

        #
        self.m_ = m
        self.M_n_m_, self.unique_X_arr_, = M_n_m, unique_X_arr  # self.unique_X_dic_, self.idx_lst_ unique_X_dic, idx_lst
        self.A_m_m_ = A_m_m
        self.designM_left_, self.designM_right_ = designM_left, designM_right
        
        # plot the training process
        if self.verbose:
            self._plot()

        return self
    
    
    # step 0, augmented support: unique_X_arr_, Missing pattern: M_n_m_
    def _support(self, X):
        n,p = X.shape[0], X.shape[1]
        X_complete_arr = []
        # get fully observed x
        for x in X:
          if not any( np.isnan(x) ): # x fully observed 
            X_complete_arr.append(x.copy()) # for safety copy it. 
        X_complete_arr = np.vstack(X_complete_arr);  # may have duplicate completely observed x
        unique_val_byCol = {};  # save unique values of each column from completely observed x
        for j in range(p):
          unique_val_byCol[j] = np.unique(X_complete_arr[:,j])
        # print('unique_val_byCol %s' % unique_val_byCol)
        # augment support
        unique_X_dic = {} # key is xk fully observed, value is the idx in {x1,...xm}
        unique_X_arr = [] # support x1, ...xm, fully observed
        count = 0 # keep track of augmented xj, up to m 
        idx_lst = [] # n elements, the ith element recording: k \in Ci, k = 1...m 
        for x in X:
          if not any( np.isnan(x) ): # x fully observed 
              if tuple(x) in unique_X_dic: 
                idx = unique_X_dic[tuple(x)]
                idx_lst.append([idx])
              else:
                unique_X_arr.append(x)
                unique_X_dic[tuple(x)] = count
                idx_lst.append([count])
                count+=1 # a new xk

          else: # x coarsened, augmented it using unique_val_byCol
              missing_col = np.where(np.isnan(x))[0]; # print('missing_col %s' % missing_col) # 1d array
              # use augment_x_lst to save all xk \in Ci , # use MC_fasterVersion_9_11_2021
              augment_x_lst = [x.copy()] # initialize.  use copy!!! otherwise change X itself!! 
              for col_idx in missing_col:
                  augment_x_new_lst = [] # used to save partial result after filling all possible values of column j
                  for x_partial_augmented in augment_x_lst: # augment_x save partial result before column j
                      for col_unique_values in unique_val_byCol[col_idx]:
                        x_partial_augmented[col_idx] = col_unique_values
                        augment_x_new_lst.append(x_partial_augmented.copy()) # copy! 
                  # update 
                  augment_x_lst = augment_x_new_lst
              # now augment_x_lst  save all xk \in Ci
              x_idx = []
              for xl in augment_x_lst:
                if tuple(xl) in unique_X_dic:
                  idx = unique_X_dic[tuple(xl)]
                  x_idx.append(idx)
                else:
                  unique_X_arr.append(xl)
                  unique_X_dic[tuple(xl)] = count
                  x_idx.append(count)
                  count+=1 # a new xk

              idx_lst.append(x_idx)
        
        unique_X_arr = np.vstack(unique_X_arr)      
        # M_n_m matrix:   
        m = unique_X_arr.shape[0]
        M_n_m = np.zeros([n,m])
        for i in range(len(idx_lst)):
          for k in idx_lst[i]:
            M_n_m[i,k] = 1
        return M_n_m, unique_X_arr, unique_X_dic, idx_lst

    # step 1: get underset matrix A_m_m
    def _underset(self, unique_X_arr):
        m = unique_X_arr.shape[0]
        A_m_m = np.zeros([m,m])
        for l in range(m):
          for k in range(m):
            if all(unique_X_arr[k] <= unique_X_arr[l]): 
              A_m_m[l, k] = 1
        #
        return A_m_m

    # step 2: get designmatrix from M_n_m, A_m_m
    def _designMatrix(self, M_n_m, A_m_m):
        # left part will dot product with D, # right part will dot part with 1 - D
        # [A1_M, .... Am_M]
        m = A_m_m.shape[0]
        designM_left = []; designM_right = []
        for k in range(m):
            A_k = A_m_m[:, k]; # I(xk <= xl)
            A_k_bar = (A_k == 0).astype(int) #  I(xk !<= xl)
            A_k_M_left = ( M_n_m + A_k == 2).astype(int); 
            A_k_M_right = ( M_n_m + A_k_bar == 2).astype(int)
            designM_left.append(A_k_M_left); 
            designM_right.append(A_k_M_right)
        # 
        designM_left = np.dstack(designM_left) ; # previoulsy used hstack n*m^2 then times m^2*m of ql (sparse matrix), now change to dstack and tensordot ql directly, avoid using sparse matrix ql
        designM_right = np.dstack(designM_right)
        return designM_left, designM_right

    # step 3: iteration 
    def _NPEM(self, R_d_U, R_d_X, M_n_m, A_m_m, designM_left, designM_right, Y, unique_X_arr):
        prev_U = R_d_U; 
        prev_X = R_d_X
        iter = 0
        log_likelihood = []
        Error_U = []
        Error_X = []
        n = M_n_m.shape[0]
        m = M_n_m.shape[1]
        Y_bar = (Y == 0).astype(int) # Y == 0 
        Y_Y_bar = np.hstack([Y, Y_bar]); #print('Y %s' % Y); print('Y_bar %s' % Y_bar)
        while True:
            # print('**'*10)
            # print('R dx %s' % prev_U)
            R_j = np.matmul(A_m_m, prev_U); #print('R_j %s' % R_j) # 1*m 
            # log_likelihood
            log_likelihood.append( self._likelihood(R_j, prev_X, M_n_m, Y_Y_bar, Y) )
            #
            # ## update U 
            # left part will dot product with D
            denom_l = np.matmul(M_n_m, R_j * prev_X); #print('denom_l %s' % denom_l)  # n*1.
            numerator_l = (np.tensordot(designM_left, prev_X.reshape([-1, 1]), axes=(1, 0))[:,:, 0]).T  # .T to make sure numerator is m*n  #numerator_l = np.matmul(designM_left, np.kron(np.eye(m),prev_X).T ).T; #print('numerator_l %s' %numerator_l) 
            left = np.divide(numerator_l, denom_l, out = np.zeros_like(numerator_l, dtype = float), where = numerator_l!=0) ; #print('left_U %s' % left )# m*n dot n. 
            # right part will dot part with 1 - D
            denom_r = np.matmul(M_n_m, (1 - R_j) * prev_X) ; #print('denom_r %s' % denom_r)
            numerator_r = (np.tensordot(designM_right, prev_X.reshape([-1, 1]), axes=(1, 0))[:,:,0]).T  #numerator_r = np.matmul(designM_right, np.kron(np.eye(m),prev_X).T ).T; #print('numerator_r %s' %numerator_r)
            right = np.divide(numerator_r, denom_r, out = np.zeros_like(numerator_r, dtype = float), where = numerator_r!=0);#print('right_U %s' % right) # if denom_r_i ==0, means x_i is complete and its largest. I(xk !<= x_i) ==0 for all k =1,..m. meaning numerator_r[:, i]=0.  0/0 = 0
            # left and right combine into one
            left_right = np.hstack([left, right]) # m * 2n
            # update cur_U
            cur_U = 1/n * prev_U * np.matmul(left_right, Y_Y_bar); #print('cur_U %s' % cur_U)
            #
            # ## update X
            # left part will dot product with D
            M_denom_l = np.divide(M_n_m.T,  denom_l, out = np.zeros_like(M_n_m.T, dtype=float), where= M_n_m.T!=0);  #print('M_denom_l %s' %M_denom_l)# m*n 
            left_X = (M_denom_l.T * R_j  * prev_X).T ; #print('left_X %s' % left_X) # m*n 
            # right part will dot part with 1 - D
            # M dot R_bar / denom_r = M / denom_r * R_bar.  M dot R_bar has two cases of 0 : case 1: from M, which means that item won't be used case 2 : from R_bar =0. 
            M_denom_r = np.divide(M_n_m.T,  denom_r, out = np.zeros_like(M_n_m.T, dtype=float), where= M_n_m.T!=0); #print('M_denom_r %s' %M_denom_r)# m*n # May have 1/0 = inf get warning, # M has the dominating 0, 0 means won't use that item
            right_X = (M_denom_r.T * (1-R_j ) * prev_X).T ; #print('right_X with nan %s' % right_X) # m*n another warning for .inf*0 
            right_X = np.where(np.isnan(right_X),1, right_X ) ;  #print('right_X %s' % right_X) # May have inf*0 = nan,force it to 1
            # left and right combine into one
            left_right_X = np.hstack([left_X, right_X]);
            # update cur_X
            cur_X = 1/n * np.matmul(left_right_X, Y_Y_bar); #print('cur_X %s' % cur_X)
            # 
            iter +=1
            error_U = sum(( cur_U -  prev_U)**2 )/m
            error_X = sum(( cur_X -  prev_X)**2 )/m
            Error_U.append(error_U); Error_X.append(error_X)
            if (iter >= 2 and log_likelihood[-1] - log_likelihood[-2] < self.tol) or iter > self.max_iter:
               break
            #      
            prev_U = cur_U; prev_X = cur_X
        
        R_j = np.matmul(A_m_m, cur_U)
        self.risk_score_ = dict(zip([tuple(xk) for xk in unique_X_arr], list(R_j) ) )
        self.n_iter_ = iter
        self.Error_U_ = Error_U; self.Error_X_ = Error_X
        self.dict_U_ = dict(zip([tuple(xk) for xk in unique_X_arr], list(cur_U) ) )
        self.dict_X_ = dict(zip([tuple(xk) for xk in unique_X_arr], list(cur_X) ) )
        self.R_dx_ = cur_U # used for predict
        self.q_dx_ = cur_X
        self.log_likelihood_ = log_likelihood
        self.AIC_ = 2* m - 2*self.log_likelihood_[-1]
        self.BIC_ = m * np.log(n) - 2*self.log_likelihood_[-1]
        self._fitted = True

        
    def _likelihood(self, R_j, prev_X, M_n_m, Y_Y_bar, Y):
        n = M_n_m.shape[0]
        # left D
        left = np.matmul(M_n_m, R_j* prev_X) 
        left = np.log(left, out = np.zeros_like(left), where= Y!=0)
        # right 1-D
        right = np.matmul(M_n_m, (1-R_j) * prev_X) 
        right = np.log(right, out = np.zeros_like(right), where= Y==0) 
        left_right = np.hstack([left, right]); #print('log left_right likelihood %s' % left_right)
        return np.dot(left_right, Y_Y_bar) / n



    def predict_proba(self, X):
        if not self._fitted:
          raise ValueError('The object has not been fitted yet.')
        #
        try:
          X = X.to_numpy()
        except:
          pass
        # 
        ret = []
        for x in X:
            if not any(np.isnan(x)): # fully observed
                ret.append(self._CDF(x))

            # elif : # missing but with seen pattern

            else:
                ret.append(np.nan) 

        return np.array(ret)

    def predict(self, X):
        prob = self.predict_proba(X)
        return (prob>=0.5).astype(int)

    
    # helper function for predict_prob
    def _CDF(self, x):
        vec = np.zeros([self.m_])
        for k in range(self.m_):
          xk = self.unique_X_arr_[k]
          if all(xk <= x):
            vec[k] = 1
        return np.dot(self.R_dx_, vec)


    def _plot(self):
        """
        helper function plot the training process 

        """
        print('tol: %s' % self.tol);
        print('max_iter: %s' % self.max_iter)
        print('number of iterations: %s' % self.n_iter_)
        print('number of unique observed values: %s' % self.m_)
        print('estimated likelihood %s' % self.log_likelihood_[-1])
        print('AIC: %s' % self.AIC_)
        print('BIC: %s' % self.BIC_)

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
