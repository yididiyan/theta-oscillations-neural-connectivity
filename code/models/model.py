

from pyexpat import model
import numpy as np
import statsmodels.api as sm

from scipy.stats import multivariate_normal 
import matplotlib.pyplot as plt



def prepare_Y_X(Ydict, fitting_times, Xmat=None, nobs=None):
    

    Yvec = Ydict['spike_vec']
    Past = Ydict['past_filt'][:,:2] #first variable is for past two ms, second is for previous 8 ms

    if Xmat is not None:
        Past = np.concatenate((Past, Xmat), axis=1)




    Yvec= Yvec[fitting_times[:,0]==1]
    Past = Past[fitting_times[:,0]==1,:]


    #trim down
    if nobs:
        Yvec = Yvec[:nobs,:]
        Past = Past[:nobs,:]


    #First add a constant intercept variable (all ones)
    T = Past.shape[0] #the length, number observations
    const = np.ones(( Past.shape[0] ,1))


    return Yvec, np.concatenate((const, Past),1)



class MdlModel(object):
    def __init__(self, Ydict, Xmat, fitting_times, nobs=None) -> None:
        self.Ydict = Ydict
        self.Xmat = Xmat
        self.fitting_times = fitting_times
        self.nobs = nobs
        self._result = None 


        

    def fit(self):
        if not self.Ydict: 
            return 
        Yvec, X = prepare_Y_X(self.Ydict, self.fitting_times, self.Xmat, self.nobs)

        if Yvec.sum()<5.0:
            # too few spikes 
            return None 

        else:
        
            ## fit with all data 
            model = sm.GLM(Yvec, X, family=sm.families.Binomial())                

            return {
                'type': 'MDL',
                'result': model.fit(),
                'endog': Yvec,
                'exog':  X
            }
    





class TrainTestModel(object):
    def __init__(self, Ydict, Xmat, fitting_times, nobs=None, training_prop=.8) -> None:
        self.Ydict = Ydict
        self.Xmat = Xmat
        self.fitting_times = fitting_times
        self.nobs = nobs

        ## index of train/test split 
        self.training_prop = training_prop
        assert self.training_prop < 1. , "Training proportion should be less than 1." 

        

    def fit(self):
        if not self.Ydict: 
            return 
        Yvec, X = prepare_Y_X(self.Ydict, self.fitting_times, self.Xmat, self.nobs)


        if Yvec.sum()<5.0:
            # too few spikes 
            return None 

        else:
        
            split_index = int(len(Yvec) * self.training_prop)
            ## fit on train data, and evaluate entropy on test data 
            model = sm.GLM(Yvec[:split_index], X[:split_index], family=sm.families.Binomial())
            return {
                'type': 'TTS',
                'result': model.fit(), ## model result
                'endog': Yvec[split_index:], # Y's for entropy calculation
                'exog':  X[split_index:] # X;s for entropy calculation
            }
        
        


 

class ModelComparator(object):
    def __init__(self, model_Y, model_Y_X) -> None:
        self.model_Y = model_Y
        self.model_Y_X = model_Y_X

    def compare(self):
        assert self.model_Y['type'] == self.model_Y_X['type'], "Models should be of the same type."

        if self.model_Y['type'] == 'MDL':
            return self.compare_fit_mdl()
        elif self.model_Y['type'] == 'TTS':
            return self.compare_fit_tts()
        else: 
            raise ValueError('Unsupported model comparision')



        
    @staticmethod
    def get_entropy_and_mdl(model_result):
        T = model_result.nobs
        
        H = - model_result.llf/T
        n_params = len(model_result.params) ## Including the intercept term
        mdl = (n_params)*np.log2(T)/2/T
        
        
        return H, mdl
    
    @staticmethod
    def get_entropy_and_mdl_2(params, endog, exog):
        '''
        Looks like internal statsmodel uses log and not log2 
        '''
        exparg = np.dot(exog, params)
        prob1 = 1./(1. + np.exp(-exparg))
        prob0 = 1. - prob1 
        endog = np.squeeze(endog)
        
        T = len(endog)
        
        n_params = len(params)
        mdl = 0. # zero MDL penalty - since we already address the problem using train-test split method 
        H = - np.sum( np.log(prob1)* endog  + np.log(prob0) * (1. - endog))   / T
        
        return H, mdl
    
    
    def compare_fit_tts(self, params_Y=None, params_Y_X=None):
        res_Y, endog_Y, exog_Y = self.model_Y['result'], self.model_Y['endog'], self.model_Y['exog']
        res_Y_X, endog_Y_X, exog_Y_X = self.model_Y_X['result'], self.model_Y_X['endog'], self.model_Y_X['exog']
        
        
        assert np.all(endog_Y  == endog_Y_X)
        params_Y = params_Y if params_Y is not None else res_Y.params
        params_Y_X =  params_Y_X if params_Y_X is not None else res_Y_X.params 
        
        
        HY, mdlY = ModelComparator.get_entropy_and_mdl_2(params_Y, endog_Y, exog_Y)
        HY_X, mdlY_X = ModelComparator.get_entropy_and_mdl_2(params_Y_X, endog_Y_X, exog_Y_X)
        
        
        results = {}
        
        #I(X-->Y) = H(Y) - H(Y||X)
        DI_X_Y = HY - HY_X
        
        mdl_DI = mdlY-mdlY_X 
        
        results = {}
        results['HY'] = HY
        results['mdlY'] = mdlY

        results['HY_X'] = HY_X
        results['mdlY_X'] = mdlY_X
        
        results['DI_X_Y'] = DI_X_Y
        results['mdl_DI'] = mdl_DI
        results['DI_discounted_MDL'] = DI_X_Y+mdl_DI
        
        
        results['nobs'] = res_Y_X.nobs
        results['paramsY'] = params_Y
        results['paramsY_X'] = params_Y_X
     
        return results
    
    
    def compare_fit_mdl(self):
        '''
        :param fitY - model fit results using only Y's past 
        :param fitY_X - model fit results using both Y's past and bunch of parents (X's) 
        '''
        
        res_Y, res_Y_X = self.model_Y['result'], self.model_Y_X['result']

        if res_Y is None or res_Y_X is None: return None # nothing to compare
        assert res_Y.nobs == res_Y_X.nobs, "Number of observations for the models does not match"
        
        T = res_Y.nobs ## number of observations to which the fit was done on 

        
        
        HY, mdlY = ModelComparator.get_entropy_and_mdl(res_Y)
        HY_X, mdlY_X = ModelComparator.get_entropy_and_mdl(res_Y_X)
        
        results = {}
        
        #I(X-->Y) = H(Y) - H(Y||X)
        DI_X_Y = HY - HY_X
        
        mdl_DI = mdlY-mdlY_X 
        
        results = {}
        results['HY'] = HY
        results['mdlY'] = mdlY

        results['HY_X'] = HY_X
        results['mdlY_X'] = mdlY_X
        
        results['DI_X_Y'] = DI_X_Y
        results['mdl_DI'] = mdl_DI
        results['DI_discounted_MDL'] = DI_X_Y+mdl_DI
        
        
        results['nobs'] = res_Y_X.nobs
        results['paramsY'] = res_Y.params
        results['paramsY_X'] = res_Y_X.params
     
        return results

