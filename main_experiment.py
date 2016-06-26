"""Code to run all the experiments on fMRI datasets.

Author: Sandro Vega-Pons
"""

import os
import numpy as np
from load_data import load_1000_funct_connectome, load_schizophrenia_data, load_kernel_matrix
from graph_kernels.gk_sp import GK_SP
from graph_kernels.gk_wl import GK_WL
from graph_kernels.gk_dce import GK_DCE
from graph_kernels.gk_dre import GK_DRE
from cbt_ktst_nbs import apply_svm, apply_ktst, apply_nbs, plot_statistic


def simple_experiment(K, y, n_folds=5, iterations=10000, subjects=False,
                      verbose=True, data_name='', plot=True,
                      random_state=None):
    """
    Application of CBT and KTST based on an already computed kernel matrix.
    """
    #CBT
    acc, acc_null, p_value = apply_svm(K, y, n_folds=n_folds,
                                       iterations=iterations,
                                       subjects=subjects,
                                       verbose=verbose,
                                       random_state=random_state)
    if plot:
        plot_statistic(acc, acc_null, p_value, data_name=data_name,
                       stats_name='Bal_Acc')
    if verbose:
        print ''

    #KTST
    mmd2u, mmd2u_null, p_value = apply_ktst(K, y, iterations=iterations,
                                            subjects=subjects,
                                            verbose=verbose)
    if plot:
        plot_statistic(mmd2u, mmd2u_null, p_value, data_name=data_name,
                       stats_name='$MMD^2_u$')
                       


def experiment_schizophrenia_data(data_path='data', n_folds=5,
                                  iterations=10000,
                                  verbose=True, plot=True, random_state=None):
    """Run the experiments on the Schizophrenia dataset

    Parameters:
    ----------
    data_path: string
        Path to the folder containing the dataset.
    n_folds: int
        The number of folds in a StratifiedKFold cross-validation
    iterations: int
        Number of iterations to compute the null distribution of
        balanced_accuracy and MMD^2_u
    verbose: bool
    plot: bool
        Whether to plot the results of the statistical tests.

    """
    name = 'Schizophrenia'
    if verbose:
        print '\nWorking on %s dataset...' % name
        print '-----------------------'
    X, y = load_schizophrenia_data(data_path, verbose=verbose)

    # DCE + RBF
    if verbose:
        print '\n### Results for DCE_Embedding ###'

    gk_dce = GK_DCE(kernel_vector_space='rbf')
    K_dce = gk_dce.compare_pairwise(X)
    simple_experiment(K_dce, y, n_folds=n_folds, iterations=iterations,
                      verbose=verbose, data_name=name + '_dce', plot=plot,
                      random_state=random_state)

    # DRE + RBF
    if verbose:
        print '\n### Results for DR_Embedding ###'

    gk_dre = GK_DRE(kernel_vector_space='rbf')
    K_dre = gk_dre.compare_pairwise(X)
    simple_experiment(K_dre, y, n_folds=n_folds, iterations=iterations,
                      verbose=verbose, data_name=name+'_dr',plot=plot,
                      random_state=random_state)

    # WL Kernel
    if verbose:
        print '\n### Results for WL_K_Embedding ###'
    gk_wl = GK_WL(th=0.2)
    K_wl = gk_wl.compare_pairwise(X)
    simple_experiment(K_wl, y, n_folds=n_folds,
                      iterations=iterations,
                      verbose=verbose, data_name=name+'_wl', plot=plot,
                      random_state=random_state)

    # SP Kernel 
    if verbose:
        print '\n### Results for SP_K_Embedding ###'
    gk_sp = GK_SP(th=0.2)
    K_sp = gk_sp.compare_pairwise(X)
    simple_experiment(K_sp, y, n_folds=n_folds,
                      iterations=iterations,
                      verbose=verbose, data_name=name+'_sp', plot=plot,
                      random_state=random_state)
#                      
    # NBS
    th = 0.5
    mxcmp, mxcmp_null, p_value = apply_nbs(X, y, th, iterations, verbose)
    if plot:
        plot_statistic(mxcmp, mxcmp_null, p_value, data_name=name+'_nbs',
                       stats_name='Max_comp_size')


def experiment_1000_func_conn_data(data_path='data', location='all', n_folds=5,
                                   iterations=10000, verbose=True, plot=True,
                                   random_state=None):
    """
    Run the experiments on the 1000_functional_connectome data.

    Parameters:
    ----------
    data_path: string
        Path to the folder containing the dataset.
    location: string
        If location=='all' we run the experiments for all locations, otherwise
        only the selected location is used.
    n_folds: int
        The number of folds in a StratifiedKFold cross-validation
    iterations: int
        Number of iterations to compute the null distribution of
        balanced_accuracy and MMD^2_u
    verbose: bool
    plot: bool
        Whether to plot the results of the statistical tests.
    """
    if location == 'all':
        locs = os.listdir(os.path.join(data_path, 'Functional_Connectomes',
                                       'Locations'))
        if verbose:
            print("")
            print("We will analyze the following datasets:")
            print("%s \n" % '\n'.join(locs))

    else:
        locs = [location]

    for name in locs:
        if verbose:
            print('')
            print('Working on %s dataset...' % name)
            print('-----------------------')

        X, y = load_1000_funct_connectome(data_path, name, verbose=verbose)

        # DCE + RBF
        if verbose:
            print('')
            print('### Results for DCE_Embedding ###')

        gk_dce = GK_DCE(kernel_vector_space='rbf')
        K_dce = gk_dce.compare_pairwise(X)
        simple_experiment(K_dce, y, n_folds=n_folds, iterations=iterations,
                      verbose=verbose, data_name=name + '_dce', plot=plot,
                      random_state=random_state)

        # DRE + RBF
        if verbose:
            print('')
            print('### Results for DR_Embedding ###')

        gk_dre = GK_DRE(kernel_vector_space='rbf')
        K_dre = gk_dre.compare_pairwise(X)
        simple_experiment(K_dre, y, n_folds=n_folds, iterations=iterations,
                          verbose=verbose, data_name=name+'_dre',plot=plot,
                          random_state=random_state)

        # WL Kernel
        if verbose:
            print '\n### Results for WL_K_Embedding ###'
        gk_wl = GK_WL(th=0.2)
        K_wl = gk_wl.compare_pairwise(X)
        simple_experiment(K_wl, y, n_folds=n_folds,
                          iterations=iterations,
                          verbose=verbose, data_name=name+'_wl', plot=plot,
                          random_state=random_state)
    
        # SP Kernel 
        if verbose:
            print '\n### Results for SP_K_Embedding ###'
        gk_sp = GK_SP(th=0.2)
        K_sp = gk_sp.compare_pairwise(X)
        simple_experiment(K_sp, y, n_folds=n_folds,
                          iterations=iterations,
                          verbose=verbose, data_name=name+'_sp', plot=plot,
                          random_state=random_state)
                          
        # NBS
        th = 1.0
        mxcmp, mxcmp_null, p_value = apply_nbs(X, y, th, iterations, verbose)
        if plot:
            plot_statistic(mxcmp, mxcmp_null, p_value, data_name=name+'_nbs',
                           stats_name='Max_comp_size')


def experiment_precomputed_matrix(data_path='data', study='wl_kernel',
                                  n_folds=5,
                                  iterations=10000, verbose=True, plot=True,
                                  random_state=None):
    """Run the experiments on the Uri dataset. For simplicity, already
    computed kernel matrices are used.

    Paramters:
    ---------
    data_path: string
        Path to the folder containing the dataset.
    study: string
        Name of the study (kernel method) to be used, e.g. 'wl_kernel' should
        contain the kernel matrix computed with the WL kernel.
    n_folds: int
        The number of folds in a StratifiedKFold cross-validation
    iterations: int
        Number of iterations to compute the null distribution of
        balanced_accuracy and MMD^2_u
    verbose: bool
        Verbosity
    plot: bool
        Whether to plot the results of the statistical tests.

    """
    name = 'Uri_' + study
    if verbose:
        print('')
        print('Working on %s dataset...' % name)
        print('-----------------------')

    K, y = load_kernel_matrix(data_path, study, verbose=verbose)

    simple_experiment(K, y, n_folds=n_folds, iterations=iterations,
                      subjects=True,
                      verbose=verbose, data_name=name, plot=plot,
                      random_state=random_state)


if __name__ == '__main__':

    # numpy seed
    seed = 0
    random_state = np.random.RandomState(seed)

    # Path to the folder containing all datasets
    data_path = 'data'

    # whether to use the schizophrenia dataset in the experiments
    schizophrenia_data = True

    # whether to use the 1000_functional_connectoms dataset in the experiments
    connectome_data = False

    # Location to be used inside the 1000_functional_connectome
    # (e.g. location='Leipzig'). If 'all', experiments are run on all
    # locations.
    location = 'Leipzig'
#    location = 'all'

    # Whether to use the Uri dataset. We will use already computed
    # kernel matrices on the Uri dataset. There will be a kernel
    # matrix for each kernel, e.g. WL and SP.
    uri_data = False

    # Number of iterations to compute the null distribution.
    iterations = 1000
    
    # Verbosity
    verbose = True

    # Whether to plot the results of test statistics, their
    # null-distributions and p-values.
    plot = True

    # Number of folds in the Stratified cross-validation (CBT only).
    n_folds = 5

    if schizophrenia_data:
        experiment_schizophrenia_data(data_path, n_folds, iterations, verbose,
                                      plot, random_state)

    if connectome_data:
        experiment_1000_func_conn_data(data_path, location, n_folds,
                                       iterations,
                                       verbose, plot, random_state)

    if uri_data:
        studies = os.listdir(os.path.join(data_path, 'precomputed_kernels'))
        for std in studies:
            experiment_precomputed_matrix(data_path, std, n_folds, iterations,
                                          verbose, plot, random_state)

    