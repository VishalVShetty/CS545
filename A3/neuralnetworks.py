import numpy as np
import optimizers as opt
import sys  # for sys.float_info.epsilon
import qdalda

######################################################################
## class NeuralNetwork()
######################################################################
##ParentClass
class NeuralNetwork():

    def __init__(self, n_inputs, n_hidden_units_by_layers, n_outputs):
        '''
        n_inputs: int
        n_hidden_units_by_layers: list of ints, or empty
        n_outputs: int
        '''

        self.n_inputs = n_inputs
        self.n_hidden_units_by_layers = n_hidden_units_by_layers
        self.n_outputs = n_outputs

        # Build list of shapes for weight matrices in each layera
        shapes = []
        n_in = n_inputs
        for nu in self.n_hidden_units_by_layers + [n_outputs]:
            shapes.append((n_in + 1, nu))
            n_in = nu

        self.all_weights, self.Ws = self._make_weights_and_views(shapes)
        self.all_gradients, self.Grads = self._make_weights_and_views(shapes)

        self.total_epochs = 0
        self.error_trace = []
        self.X_means = None
        self.X_stds = None
        self.T_means = None
        self.T_stds = None

    def _make_weights_and_views(self, shapes):
        '''
        shapes: list of pairs of ints for number of rows and columns
                in each layer
        Returns vector of all weights, and views into this vector
                for each layer
        '''
        all_weights = np.hstack([np.random.uniform(size=shape).flat
                                 / np.sqrt(shape[0])
                                 for shape in shapes])
        # Build list of views by reshaping corresponding elements
        # from vector of all weights into correct shape for each layer.
        views = []
        first_element = 0
        for shape in shapes:
            n_elements = shape[0] * shape[1]
            last_element = first_element + n_elements
            views.append(all_weights[first_element:last_element]
                         .reshape(shape))
            first_element = last_element

        return all_weights, views

    def __repr__(self):
        return f'NeuralNetwork({self.n_inputs}, ' + \
            f'{self.n_hidden_units_by_layers}, {self.n_outputs})'

    def train(self, X, T, n_epochs, method='sgd', learning_rate=None, verbose=True):
        '''
        X: n_samples x n_inputs matrix of input samples, one per row
        T: n_samples x n_outputs matrix of target output values,
            one sample per row
        n_epochs: number of passes to take through all samples
            updating weights each pass
        method: 'sgd', 'adam', or 'scg'
        learning_rate: factor controlling the step size of each update
        '''

        # Setup standardization parameters
        # Setup standardization parameters
        if self.X_means is None:
            self.X_means = X.mean(axis=0)
            self.X_stds = X.std(axis=0)
            self.X_stds[self.X_stds == 0] = 1
            self.T_means = T.mean(axis=0)
            self.T_stds = T.std(axis=0)

        # Standardize X and T
        X = (X - self.X_means) / self.X_stds
        T = (T - self.T_means) / self.T_stds

        # Instantiate Optimizers object by giving it vector of all weights
        optimizer = opt.Optimizers(self.all_weights)

        _error_convert_f = lambda err: (np.sqrt(err) * self.T_stds)[0]

        if method == 'sgd':

            error_trace = optimizer.sgd(self._error_f, self._gradient_f,
                                        fargs=[X, T], n_epochs=n_epochs,
                                        learning_rate=learning_rate,
                                        error_convert_f=_error_convert_f,
                                        verbose=verbose)

        elif method == 'adam':

            error_trace = optimizer.adam(self._error_f, self._gradient_f,
                                         fargs=[X, T], n_epochs=n_epochs,
                                         learning_rate=learning_rate,
                                         error_convert_f=_error_convert_f,
                                         verbose=verbose)

        elif method == 'scg':

            error_trace = optimizer.scg(self._error_f, self._gradient_f,
                                        fargs=[X, T], n_epochs=n_epochs,
                                        error_convert_f=_error_convert_f,
                                        verbose=verbose)

        else:
            raise Exception("method must be 'sgd', 'adam', or 'scg'")

        self.total_epochs += len(error_trace)
        self.error_trace += error_trace

        # Return neural network object to allow applying other methods
        # after training, such as:    Y = nnet.train(X, T, 100, 0.01).use(X)

        return self
    
    def _forward(self, X):
        '''
        X assumed to be standardized and with first column of 1's
        '''
        self.Ys = [X]
        for W in self.Ws[:-1]:  # forward through all but last layer
            #print('at forward',self.Ys[-1].shape, W[1:, :].shape)
            self.Ys.append(np.tanh(self.Ys[-1] @ W[1:, :] + W[0:1, :]))
        last_W = self.Ws[-1]
        self.Ys.append(self.Ys[-1] @ last_W[1:, :] + last_W[0:1, :])
        return self.Ys

    # Function to be minimized by optimizer method, mean squared error
    def _error_f(self, X, T):
        Ys = self._forward(X)
        mean_sq_error = np.mean((T - Ys[-1]) ** 2)
        return mean_sq_error

    # Gradient of function to be minimized for use by optimizer method
    def _gradient_f(self, X, T):
        # Assumes forward_pass just called with layer outputs saved in self.Ys.
        n_samples = X.shape[0]
        n_outputs = T.shape[1]

        # D is delta matrix to be back propagated
        D = -(T - self.Ys[-1]) / (n_samples * n_outputs)
        self._backpropagate(D)

        return self.all_gradients

    def _backpropagate(self, D):
        # Step backwards through the layers to back-propagate the error (D)
        n_layers = len(self.n_hidden_units_by_layers) + 1
        for layeri in range(n_layers - 1, -1, -1):
            # gradient of all but bias weights
            self.Grads[layeri][1:, :] = self.Ys[layeri].T @ D
            # gradient of just the bias weights
            self.Grads[layeri][0:1, :] = np.sum(D, axis=0)
            # Back-propagate this layer's delta to previous layer
            if layeri > 0:
                D = D @ self.Ws[layeri][1:, :].T * (1 - self.Ys[layeri] ** 2)

    def use(self, X):
        '''X assumed to not be standardized'''
        # Standardize X
        X = (X - self.X_means) / self.X_stds
        Ys = self._forward(X)
        # Unstandardize output Y before returning it
        return Ys[-1] * self.T_stds + self.T_means

    def get_error_trace(self):
        return self.error_trace


    
## Parent Class ends here Child Class Starts here ##
    

    
    
    
##ChildClass
class NeuralNetworkClassifier(NeuralNetwork):
    def __init__(self,n_inputs, n_hidden_units_by_layers, n_classes):
        super().__init__(n_inputs, n_hidden_units_by_layers, n_classes)
    
    def __str__(self):
        s = self.__repr__()
        if self.total_epochs > 0:
            s += f'\n Trained for {self.total_epochs} epochs.'
            s += f'\n Final standardized training error {self.error_trace[-1]:.4g}.'
        return s

    ##modified trina fucntion for classfier
    def train(self, X, T, n_epochs, method='sgd', learning_rate=None, verbose=True):
        '''
        X: n_samples x n_inputs matrix of input samples, one per row
        T: n_samples x n_outputs matrix of target output values,
            one sample per row
        n_epochs: number of passes to take through all samples
            updating weights each pass
        method: 'sgd', 'adam', or 'scg'
        learning_rate: factor controlling the step size of each update
        '''

        # Setup standardization parameters
        # Setup standardization parameters
        if self.X_means is None:
            #self.X_means = X.mean(axis=0)
            #self.X_stds = X.std(axis=0)
            #self.X_stds[self.X_stds == 0] = 1
            #written a function for this no need for thses lines of code
            #self.T_means = T.mean(axis=0)
            #self.T_stds = T.std(axis=0)
            X = self.standardize(X)

        # Standardize X and T
        #X = (X - self.X_means) / self.X_stds

        # Instantiate Optimizers object by giving it vector of all weights
        optimizer = opt.Optimizers(self.all_weights)
        TI = self.makeIndicatorVars(T)

        _error_convert_f = lambda nll: np.exp(-nll)

        if method == 'sgd':

            error_trace = optimizer.sgd(self.neg_log_likelihood, self._gradient_f,
                                        fargs=[X, TI], n_epochs=n_epochs,
                                        learning_rate=learning_rate,
                                        error_convert_f=_error_convert_f,
                                        verbose=verbose)

        elif method == 'adam':

            error_trace = optimizer.adam(self.neg_log_likelihood, self._gradient_f,
                                         fargs=[X, TI], n_epochs=n_epochs,
                                         learning_rate=learning_rate,
                                         error_convert_f=_error_convert_f,
                                         verbose=verbose)

        elif method == 'scg':

            error_trace = optimizer.scg(self.neg_log_likelihood, self._gradient_f,
                                        fargs=[X, TI], n_epochs=n_epochs,
                                        error_convert_f=_error_convert_f,
                                        verbose=verbose)

        else:
            raise Exception("method must be 'sgd', 'adam', or 'scg'")

        self.total_epochs += len(error_trace)
        self.error_trace += error_trace

        # Return neural network object to allow applying other methods
        # after training, such as:    Y = nnet.train(X, T, 100, 0.01).use(X)

        return self
    

    def forward(self, X):
        '''
        X assumed to be standardized and with first column of 1's
        '''
        self.Ys = [X]
        for W in self.Ws[:-1]:  # forward through all but last layer
            #print('at forward',self.Ys[-1].shape, W[1:, :].shape)
            self.Ys.append(np.tanh(self.Ys[-1] @ W[1:, :] + W[0:1, :]))
        last_W = self.Ws[-1]
        self.Ys.append(self._softmax(self.Ys[-1] @ last_W[1:, :] + last_W[0:1, :])) ## softmax called here to avoid calling it in Other functions that follow forward
        return self.Ys

    def _softmax(self, Y):
        '''Apply to final layer weighted sum outputs'''
        # Trick to avoid overflow
        #print(Y)
        #maxY = Y.max()#changed since list operation was not possible NOT WORKINg
        Z = np.max(Y)
        maxY = max(0,Z)
        expY = np.exp(Y - maxY)
        denom = expY.sum(1).reshape((-1, 1))
        Y = expY / (denom + sys.float_info.epsilon)
        return Y
    
    def neg_log_likelihood(self,X,TI):
        # w = warg.reshape((-1,K))
        Y = self.forward(X) #forward function called here
        #print('Output from forward',Y)
        #G = self._softmax(Y[-1]) ## softmax called within forward now for ease
        #print('Output from softmax',Y[-1])
        #print('Output from negLL',-np.mean(TI * np.log(Y[-1] + sys.float_info.epsilon)))
        return -np.mean(TI * np.log(Y[-1] + sys.float_info.epsilon))

    def standardize(self,X):
        X_means = X.mean(axis=0)
        X_stds = X.std(axis=0)
        X_stds[X_stds == 0] = 1
        return (X - X_means) / X_stds

    def makeIndicatorVars(self,T):
    # Make sure T is two-dimensional. Should be nSamples x 1.
        if T.ndim == 1:
            T = T.reshape((-1, 1))    
        self.Tunique = np.unique(T) # needed for extracting unique values from T i.e the different class types
        return (T == np.unique(T)).astype(int)
   
    def use(self, X,return_hidden_layer_outputs=False):
        '''X assumed to not be standardized'''
        # Standardize X
        X = self.standardize(X)
        Ys = self.forward(X)
        classes= np.argmax(Ys[-1],axis=1).reshape(-1,1)
        
        classes_T = classes
        
        i = 0
        while i <(classes.shape[0]):
            for value in classes[i]:
                classes_T[i] = self.Tunique[value]
            i += 1 
        #classes_T = [self.Tunique[value] for i in range(classes.shape[0]) for value in classes[i]] 
        
        # Unstandardize output Y before returning it
        #return Ys[-1] * self.T_stds + self.T_means
        Zs = Ys[:-1]
        return (classes_T,Ys[-1],Zs) if return_hidden_layer_outputs else (classes_T, Ys[-1]) 
