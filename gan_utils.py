import tensorflow.compat.v1 as tf
import numpy as np

## UTIL FOR GAN IMPUTER
class GanUtil:
    '''Utility functions for GAIN.
    (1) normalization: MinMax Normalizer
    (2) renormalization: Recover the data from normalzied data
    (3) rounding: Handlecategorical variables after imputation
    (4) rmse_loss: Evaluate imputed data in terms of RMSE
    (5) xavier_init: Xavier initialization
    (6) binary_sampler: sample binary random variables
    (7) uniform_sampler: sample uniform random variables
    (8) sample_batch_index: sample random batch index
    ''' 
    ##IF USING TF 2 use following import to still use TF < 2.0 Functionalities

    tf.disable_v2_behavior()

    def normalization (self, data, parameters=None):
        '''
        Normalize data in [0, 1] range.

        Args:
        - data: original data

        Returns:
        - norm_data: normalized data
        - norm_parameters: min_val, max_val for each feature for renormalization
        '''

        # Parameters
        _, dim = data.shape
        norm_data = data.copy()

        if parameters is None:
            # MixMax normalization
            min_val = np.zeros(dim)
            max_val = np.zeros(dim)

            # For each dimension
            for i in range(dim):
                min_val[i] = np.nanmin(norm_data[:,i])
                norm_data[:,i] = norm_data[:,i] - np.nanmin(norm_data[:,i])
                max_val[i] = np.nanmax(norm_data[:,i])
                norm_data[:,i] = norm_data[:,i] / (np.nanmax(norm_data[:,i]) + 1e-6)   

            # Return norm_parameters for renormalization
            norm_parameters = {'min_val': min_val,
                               'max_val': max_val}

        else:
            min_val = parameters['min_val']
            max_val = parameters['max_val']

            # For each dimension
            for i in range(dim):
                norm_data[:,i] = norm_data[:,i] - min_val[i]
                norm_data[:,i] = norm_data[:,i] / (max_val[i] + 1e-6)  

            norm_parameters = parameters    

        return norm_data, norm_parameters


    def renormalization (self, norm_data, norm_parameters):
        '''Renormalize data from [0, 1] range to the original range.

        Args:
        - norm_data: normalized data
        - norm_parameters: min_val, max_val for each feature for renormalization

        Returns:
        - renorm_data: renormalized original data
        '''

        min_val = norm_parameters['min_val']
        max_val = norm_parameters['max_val']

        _, dim = norm_data.shape
        renorm_data = norm_data.copy()

        for i in range(dim):
            renorm_data[:,i] = renorm_data[:,i] * (max_val[i] + 1e-6)   
            renorm_data[:,i] = renorm_data[:,i] + min_val[i]

        return renorm_data


    def rounding (self, imputed_data, data_x):
        '''Round imputed data for categorical variables.

        Args:
        - imputed_data: imputed data
        - data_x: original data with missing values

        Returns:
        - rounded_data: rounded imputed data
        '''

        _, dim = data_x.shape
        rounded_data = imputed_data.copy()

        for i in range(dim):
            temp = data_x[~np.isnan(data_x[:, i]), i]
            # Only for the categorical variable
            if len(np.unique(temp)) < 20:
                rounded_data[:, i] = np.round(rounded_data[:, i])

        return rounded_data


    def rmse_loss (self, ori_data, imputed_data, data_m):
        '''Compute RMSE loss between ori_data and imputed_data

        Args:
        - ori_data: original data without missing values
        - imputed_data: imputed data
        - data_m: indicator matrix for missingness

        Returns:
        - rmse: Root Mean Squared Error
        '''

        ori_data, norm_parameters = self.normalization(ori_data)
        imputed_data, _ = self.normalization(imputed_data, norm_parameters)

        # Only for missing values
        nominator = np.sum(((1-data_m) * ori_data - (1-data_m) * imputed_data)**2)
        denominator = np.sum(1-data_m)

        rmse = np.sqrt(nominator/float(denominator))

        return rmse


    def xavier_init(self, size):
        '''Xavier initialization.

        Args:
        - size: vector size

        Returns:
        - initialized random vector.
        '''
        in_dim = size[0]
        xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
        return tf.random_normal(shape = size, stddev = xavier_stddev)


    def binary_sampler(self, p, rows, cols):
        '''Sample binary random variables.

        Args:
        - p: probability of 1
        - rows: the number of rows
        - cols: the number of columns

        Returns:
        - binary_random_matrix: generated binary random matrix.
        '''
        unif_random_matrix = np.random.uniform(0., 1., size = [rows, cols])
        binary_random_matrix = 1*(unif_random_matrix < p)
        return binary_random_matrix


    def uniform_sampler(self, low, high, rows, cols):
        '''Sample uniform random variables.

        Args:
        - low: low limit
        - high: high limit
        - rows: the number of rows
        - cols: the number of columns

        Returns:
        - uniform_random_matrix: generated uniform random matrix.
        '''
        return np.random.uniform(low, high, size = [rows, cols])       


    def sample_batch_index(self, total, batch_size):
        '''Sample index of the mini-batch.

        Args:
        - total: total number of samples
        - batch_size: batch size

        Returns:
        - batch_idx: batch index
        '''
        total_idx = np.random.permutation(total)
        batch_idx = total_idx[:batch_size]
        return batch_idx

def data_loader (data_name, miss_rate):
    '''Loads datasets and introduce missingness.

    Args:
    - data_name: letter, spam, or mnist
    - miss_rate: the probability of missing components

    Returns:
    data_x: original data
    miss_data_x: data with missing values
    data_m: indicator matrix for missing components
    '''

    # Load data
    if data_name in ['letter', 'spam']:
        file_name = 'data/'+data_name+'.csv'
        data_x = np.loadtxt(file_name, delimiter=",", skiprows=1)
    elif data_name == 'mnist':
        (data_x, _), _ = mnist.load_data()
        data_x = np.reshape(np.asarray(data_x), [60000, 28*28]).astype(float)

    # Parameters
    no, dim = data_x.shape

    # Introduce missing data
    data_m = gan_util.binary_sampler(1-miss_rate, no, dim)
    miss_data_x = data_x.copy()
    miss_data_x[data_m == 0] = np.nan

    return data_x, miss_data_x, data_m