
import numpy as np
import optimizers as opt


class NeuralNetwork():
    """
    A class that represents a neural network for nonlinear regression

    Attributes
    ----------
    n_inputs : int
        The number of values in each sample
    n_hidden_units_by_layers: list of ints, or empty
        The number of units in each hidden layer.
        Its length specifies the number of hidden layers.
    n_outputs: int
        The number of units in output layer
    all_weights : one-dimensional numpy array
        Contains all weights of the network as a vector
    Ws : list of two-dimensional numpy arrays
        Contains matrices of weights in each layer,
        as views into all_weights
    all_gradients : one-dimensional numpy array
        Contains all gradients of mean square error with
        respect to each weight in the network as a vector
    Grads : list of two-dimensional numpy arrays
        Contains matrices of gradients weights in each layer,
        as views into all_gradients
    total_epochs : int
        Total number of epochs trained so far
    error_trace : list
        Mean square error (standardized) after each epoch
    X_means : one-dimensional numpy array
        Means of the components, or features, across samples
    X_stds : one-dimensional numpy array
        Standard deviations of the components, or features, across samples
    T_means : one-dimensional numpy array
        Means of the components of the targets, across samples
    T_stds : one-dimensional numpy array
        Standard deviations of the components of the targets, across samples
        
    Methods
    -------
    make_weights_and_views(shapes)
        Creates all initial weights and views for each layer

    train(X, T, n_epochs, method='sgd', learning_rate=None, verbose=True)
        Trains the network using samples by rows in X and T

    use(X)
        Applies network to inputs X and returns network's output
    """

    def __init__(self, nip, nhubl, nop):
        """Creates a neural network with the given structure

        Parameters
        ----------
        n_inputs : int
            The number of values in each sample
        n_hidden_units_by_layers : list of ints, or empty
            The number of units in each hidden layer.
            Its length specifies the number of hidden layers.
        n_outputs : int
            The number of units in output layer

        Returns
        -------
        NeuralNetwork object
        """

        # Assign attribute values. Set self.X_means to None to indicate
        # that standardization parameters have not been calculated.
        # ....
        self.nip=nip
        if not isinstance(nhubl, list):
            raise Exception('number of hidden units by layers must be a list.')
        self.nhubl=nhubl
        self.nop=nop
        self.X_means=None
        self.X_stds=None
        self.T_means=None
        self.T_stds=None
        self.total_epoch=None
        self.error_trace=None
          

        # Build list of shapes for weight matrices in each layer
        # ...
        Sow = []
        if(len(self.nhubl)):
            Sow.append((self.nip+1,nhubl[0]))
            for i in range(len(self.nhubl)):
                Sow.append((self.nhubl[i]+1,self.nhubl[i+1] if i+1<len(self.nhubl) else self.nop))
        else:
            Sow.append((self.nip+1,self.nop))
            
        
        # Call make_weights_and_views to create all_weights and Ws
        # ...
        self.all_weights,self.Wm=self.make_weights_and_views(Sow)
        
        
        # Call make_weights_and_views to create all_gradients and Grads
        # ...
        self.Agrad,self.grad=self.make_weights_and_views(Sow)

    def make_weights_and_views(self,Sow):
        """Creates vector of all weights and views for each layer

        Parameters
        ----------
        shapes : list of pairs of ints
            Each pair is number of rows and columns of weights in each layer

        Returns
        -------
        Vector of all weights, and list of views into this vector for each layer
        """

        # Create one-dimensional numpy array of all weights with random initial values
        #  ...
        Law=0
        for i in range(len(Sow)):
            Law+=Sow[i][0]*Sow[i][1]
        all_weights=(np.random.uniform(-1,1,Law)).flat/np.sqrt(self.nip+self.nop)
        
        # Build list of views by reshaping corresponding elements
        # from vector of all weights into correct shape for each layer.        
        # ...
        Wm=[]
        j=0
        for i in range(len(Sow)):
            Mlc=Sow[i][0]*Sow[i][1]
            Wm.append(all_weights[j:j+Mlc].reshape(Sow[i][0],Sow[i][1]))
            j=j+Mlc
            
        return all_weights,Wm

    def __repr__(self):
        return f'NeuralNetwork({self.n_inputs}, ' + \
            f'{self.n_hidden_units_by_layers}, {self.n_outputs})'

    def __str__(self):
        s = self.__repr__()
        if self.total_epochs > 0:
            s += f'\n Trained for {self.total_epochs} epochs.'
            s += f'\n Final standardized training error {self.error_trace[-1]:.4g}.'
        return s
 
    def train(self, X, T, n_epochs, method='sgd', learning_rate=None, verbose=True):
        """Updates the weights 

        Parameters
        ----------
        X : two-dimensional numpy array
            number of samples  x  number of input components
        T : two-dimensional numpy array
            number of samples  x  number of output components
        n_epochs : int
            Number of passes to take through all samples
        method : str
            'sgd', 'adam', or 'scg'
        learning_rate : float
            Controls the step size of each update, only for sgd and adam
        verbose: boolean
            If True, progress is shown with print statements
        """

        # Calculate and assign standardization parameters
        # ...
        self.X_means=np.mean(X,axis=0)
        self.X_stds=np.std(X,axis=0)
        self.T_means=np.mean(T,axis=0)
        self.T_stds=np.std(T,axis=0)

        # Standardize X and T
        # ...
        XS=(X-self.X_means)/self.X_stds
        TS=(T-self.T_means)/self.T_stds

        # Instantiate Optimizers object by giving it vector of all weights
        optimizer = opt.Optimizers(self.all_weights)

        error_convert_f = lambda err: (np.sqrt(err) * self.T_stds)[0]
        
        # Call the requested optimizer method to train the weights.
        self.total_epochs=n_epochs
        self.error_trace=[]


        if method == 'sgd':

            error_trace=optimizer.sgd(self.error_f,self.gradient_f,[XS,TS],n_epochs,0.01,True,error_convert_f,False,None)

        elif method == 'adam':

            error_trace=optimizer.adam(self.error_f,self.gradient_f,[XS,TS],n_epochs,learning_rate,verbose,error_convert_f,None)

        elif method == 'scg':

            error_trace=optimizer.scg(self.error_f,self.gradient_f,[XS,TS],n_epochs,error_convert_f,True,None)

        else:
            raise Exception("method must be 'sgd', 'adam', or 'scg'")

        self.total_epochs += len(error_trace)
        self.error_trace += error_trace

        # Return neural network object to allow applying other methods
        # after training, such as:    Y = nnet.train(X, T, 100, 0.01).use(X)

        return self

    def _forward(self, X):
        """Calculate outputs of each layer given inputs in X
        
        Parameters
        ----------
        X : input samples, standardized

        Returns
        -------
        Outputs of all layers as list
        """
        self.Ys = [X]
        # Append output of each layer to list in self.Ys, then return it.
        # ...
        for layeri in range(len(self.Wm)):
            self.Ys.append(np.tanh(self.Ys[layeri]@self.Wm[layeri][1:,:]+self.Wm[layeri][0:1,:]))
                           
        self.Ys.append(self.Ys[layeri]@self.Wm[-1][1:,:]+self.Wm[-1][0:1,:])
        
        return self.Ys

    # Function to be minimized by optimizer method, mean squared error
    def error_f(self, X, T):
        """Calculate output of net and its mean squared error 

        Parameters
        ----------
        X : two-dimensional numpy array
            number of samples  x  number of input components
        T : two-dimensional numpy array
            number of samples  x  number of output components

        Returns
        -------
        Mean square error as scalar float that is the mean
        square error over all samples
        """
        # Call _forward, calculate mean square error and return it.
        # ...
        Y_forwardpass=self._forward(X)
        mean_square_error = np.mean((T-Y_forwardpass[-1])**2)
        return mean_square_error

    # Gradient of function to be minimized for use by optimizer method
    def gradient_f(self, X, T):
        """Returns gradient wrt all weights. Assumes _forward already called.

        Parameters
        ----------
        X : two-dimensional numpy array
            number of samples  x  number of input components
        T : two-dimensional numpy array
            number of samples  x  number of output components

        Returns
        -------
        Vector of gradients of mean square error wrt all weights
        """

        # Assumes forward_pass just called with layer outputs saved in self.Ys.
        nos = X.shape[0]
        nop = T.shape[1]
        nol = len(self.nhubl) + 1

        # D is delta matrix to be back propagated
        D = -(T - self.Ys[-1]) / (nos * nop)

        # Step backwards through the layers to back-propagate the error (D)
        for layeri in range(nol - 1, -1, -1):
            # gradient of all but bias weights
            self.grad[layeri][1:, :] = self.Ys[layeri].T @ D
            # gradient of just the bias weights
            self.grad[layeri][0:1, :] = np.sum(D, axis=0)
            # Back-propagate this layer's delta to previous layer
            if layeri > 0:
                D = (D @ self.Wm[layeri][1:,:].T)*(1-self.Ys[layeri]**2)

        return self.Agrad

    def use(self, X):
        """Return the output of the network for input samples as rows in X

        Parameters
        ----------
        X : two-dimensional numpy array
            number of samples  x  number of input components, unstandardized

        Returns
        -------
        Output of neural network, unstandardized, as numpy array
        of shape  number of samples  x  number of outputs
        """

        # Standardize X
        # ...
        X_means=np.mean(X,axis=0)
        X_stds=np.std(X,axis=0)
        XS=(X-X_means)/X_stds
        YS=self._forward(XS)
        # Unstandardize output Y before returning it
        Y=YS[-1]*self.T_stds+self.T_means
        return Y

    def get_error_trace(self):
        """Returns list of standardized mean square error for each epoch"""
        return self.error_trace
