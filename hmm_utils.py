import warnings
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Ellipse
from sklearn.mixture import GaussianMixture

import quasiparticleFunctions as qp


def set_plotting_backend(backend='inline'):
    """
    Set the matplotlib backend.
    
    Parameters
    ----------
    backend : str
        Either 'qt' or 'inline'
    """
    try:
        plt.close('all')  # Close any existing plots
        import IPython
        IPython.get_ipython().run_line_magic('matplotlib', backend)
    except (ImportError, AttributeError):
        if backend == 'qt':
            warnings.warn("Could not set Qt backend. Interactive features may not work.")
        else:
            warnings.warn("Could not set inline backend. Using default matplotlib backend.")

def reorder_gmm_states(
    gmm_result: Dict[str, Any],
    new_order: List[int]
) -> Dict[str, Any]:
    """
    Reorder GMM states based on provided mapping.
    
    Parameters
    ----------
    gmm_result : Dict[str, Any]
        Original GMM results dictionary
    new_order : List[int]
        List specifying new order of states
        
    Returns
    -------
    Dict[str, Any]
        Updated GMM results with reordered states
    """
    num_modes = len(gmm_result['means'])
    if len(new_order) != num_modes:
        raise ValueError("New order must have same length as number of states")
    
    # Create new result dictionary
    reordered = {}
    
    # Reorder means
    reordered['means'] = gmm_result['means'][new_order]
    
    # Reorder covariances
    reordered['covariances'] = gmm_result['covariances'][new_order]
    
    # Reorder populations and create new labels
    old_labels = gmm_result['labels']
    new_labels = np.zeros_like(old_labels)
    new_populations = np.zeros_like(gmm_result['populations'])
    
    for new_idx, old_idx in enumerate(new_order):
        mask = (old_labels == old_idx)
        new_labels[mask] = new_idx
        new_populations[new_idx] = gmm_result['populations'][old_idx]
    
    reordered['labels'] = new_labels
    reordered['populations'] = new_populations
    
    # Keep the model
    reordered['model'] = gmm_result['model']
    
    return reordered

def get_user_state_ordering(
    data: np.ndarray,
    gmm_result: Dict[str, Any],
    title: str = "Click on states in desired order (highest frequency first)"
) -> List[int]:
    """
    Get user input for state reordering using interactive plot.
    
    Parameters
    ----------
    data : np.ndarray
        The data points
    gmm_result : Dict[str, Any]
        GMM results dictionary
    title : str, optional
        Plot title
        
    Returns
    -------
    List[int]
        New ordering of states
    """
    # Close any existing figures
    plt.close('all')
    
    means = gmm_result['means']
    num_modes = len(means)
    
    # Create temporary plot for user input
    fig = plt.figure(figsize=(12, 10))
    
    # Plot data and states similar to plot_gmm_results
    h = plt.hist2d(data[:, 0], data[:, 1], bins=80, 
                   norm=LogNorm(), cmap='Greys')
    plt.colorbar(h[3], shrink=0.9, extend='both')
    
    # Plot current state assignments
    for i in range(num_modes):
        plt.scatter(means[i, 0], means[i, 1], c=f'C{i}', s=200, marker='x', linewidth=3,
                   label=f'Current State {i}\n(n={gmm_result["populations"][i]})')
        plt.text(means[i, 0], means[i, 1], f' {i}', fontsize=14, 
                 color=f'C{i}', fontweight='bold')
    
    plt.title(title)
    plt.grid(True, alpha=0.3)
    # Move legend outside the plot area
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.gca().set_aspect('equal')
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    
    # Get user clicks
    print(f"\nPlease click on the states in desired order (highest frequency first)")
    print(f"You need to click on {num_modes} states")
    
    clicks = plt.ginput(n=num_modes, timeout=-1)
    plt.close(fig)  # Close this specific figure
    plt.close('all')  # Make sure all figures are closed
    
    # Map clicks to nearest states
    new_order = []
    for click in clicks:
        distances = np.sqrt(np.sum((means - np.array(click))**2, axis=1))
        nearest_state = np.argmin(distances)
        if nearest_state in new_order:
            warnings.warn(f"State {nearest_state} was selected multiple times!")
        new_order.append(nearest_state)
    
    return new_order

def fit_gmm(
    data: np.ndarray,
    num_modes: int,
    covariance_type: str = 'full',
    init_params: str = 'kmeans',
    n_init: int = 3,
    max_iter: int = 300,
    tol: float = 1e-3,
    random_state: int = 42,
    verbose: int = 1,
    allow_reorder: bool = True
) -> Dict[str, Any]:
    """
    Fit a Gaussian Mixture Model to the data and extract means and covariances.
    Optionally allows interactive reordering of states.
    
    Parameters
    ----------
    data : np.ndarray
        The data to fit the GMM to. Can be either:
        - shape (n_samples, n_features) for sklearn format
        - shape (n_features, n_samples) for QP format
        Will be automatically transposed if needed.
    num_modes : int
        The number of components (modes) in the GMM.
    covariance_type : str, optional
        The type of covariance parameters, by default 'full'.
        Options: 'full', 'tied', 'diag', 'spherical'.
    init_params : str, optional
        The parameters to initialize, by default 'kmeans'.
    n_init : int, optional
        The number of initializations to try, by default 5.
    max_iter : int, optional
        The maximum number of iterations, by default 300.
    tol : float, optional
        The convergence threshold, by default 1e-4.
    random_state : int, optional
        The random state for reproducibility, by default 42.
    verbose : int, optional
        The verbosity level, by default 1.
    allow_reorder : bool, optional
        Whether to allow interactive state reordering, by default True
    
    Returns
    -------
    Dict[str, Any]
        A dictionary containing the GMM results with potentially reordered states
    """
    # Close any existing figures first
    plt.close('all')
    
    # Start with inline backend for initial plot
    set_plotting_backend('inline')
    
    # Check data shape and transpose if needed
    if data.shape[0] == 2 and data.shape[1] > 2:
        # Data is in (n_features, n_samples) format, transpose it
        data = data.T
    elif data.shape[1] == 2 and data.shape[0] > 2:
        # Data is already in (n_samples, n_features) format
        pass
    else:
        raise ValueError(f"Data shape {data.shape} is invalid. Must be either (2, n_samples) or (n_samples, 2)")
    
    # Initialize and fit Gaussian Mixture Model
    gmm = GaussianMixture(
        n_components=num_modes,
        covariance_type=covariance_type,
        init_params=init_params,
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
        verbose=verbose
    )
    
    # Fit the model
    gmm.fit(data)
    
    # Get the means and covariances
    means = gmm.means_
    covariances = gmm.covariances_
    
    # Get cluster assignments and populations
    labels = gmm.predict(data)
    cluster_populations = np.zeros(num_modes, dtype=int)
    for i in range(num_modes):
        cluster_populations[i] = np.sum(labels == i)
    
    # Create initial result dictionary
    result = {
        'means': means,
        'covariances': covariances,
        'labels': labels,
        'populations': cluster_populations,
        'model': gmm
    }
    
    # Show initial plot
    print("\nInitial state assignments:")
    plot_gmm_results(data, result)
    
    if allow_reorder:
        # Ask user if they want to reorder states
        while True:
            response = input("\nDo you want to reorder the states? (y/n): ").lower()
            if response in ['y', 'yes', 'n', 'no']:
                break
            print("Please answer 'y' or 'n'")
        
        if response in ['y', 'yes']:
            try:
                # Close any remaining figures before switching backend
                plt.close('all')
                
                # Switch to Qt backend for interactive plotting
                set_plotting_backend('qt')
                
                # Get new ordering from user
                new_order = get_user_state_ordering(data, result)
                
                # Close any remaining figures before switching back
                plt.close('all')
                
                # Switch back to inline backend
                set_plotting_backend('inline')
                
                # Reorder states
                result = reorder_gmm_states(result, new_order)
                
                # Show final plot
                print("\nFinal state assignments after reordering:")
                plot_gmm_results(data, result)
                
            except Exception as e:
                warnings.warn(f"Interactive reordering failed: {e}\nKeeping original state order.")
                plt.close('all')  # Make sure to close any remaining figures
                set_plotting_backend('inline')  # Ensure we're back to inline
    
    return result

def plot_gmm_results(
    data: np.ndarray,
    gmm_result: Dict[str, Any],
    title: str = 'I-Q Data with Gaussian Mixture Model\nMeans and 2σ Covariance Ellipses',
    xlabel: str = 'I [mV]',
    ylabel: str = 'Q [mV]',
    bins: int = 80,
    cmap: str = 'Greys',
    figsize: Tuple[int, int] = (10, 10),
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot the GMM results overlaid on the data.
    
    Parameters
    ----------
    data : np.ndarray
        The data used to fit the GMM, shape (n_samples, n_features).
    gmm_result : Dict[str, Any]
        The result from fit_gmm().
    title : str, optional
        The title of the plot, by default 'I-Q Data with Gaussian Mixture Model\nMeans and 2σ Covariance Ellipses'.
    xlabel : str, optional
        The label for the x-axis, by default 'I [mV]'.
    ylabel : str, optional
        The label for the y-axis, by default 'Q [mV]'.
    bins : int, optional
        The number of bins for the 2D histogram, by default 80.
    cmap : str, optional
        The colormap for the 2D histogram, by default 'Greys'.
    figsize : Tuple[int, int], optional
        The figure size, by default (10, 10).
    save_path : Optional[str], optional
        The path to save the plot to, by default None.
    show : bool, optional
        Whether to show the plot, by default True.
    
    Returns
    -------
    None
    
    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.randn(1000, 2)
    >>> result = fit_gmm(data, 3)
    >>> plot_gmm_results(data, result)
    """
    # Close any existing figures first
    plt.close('all')
    
    # Create the figure
    fig = plt.figure(figsize=figsize)
    
    # Plot 2D histogram of data
    h = plt.hist2d(data[:, 0], data[:, 1], bins=bins, 
                   norm=LogNorm(), cmap=plt.cm.get_cmap(cmap))
    plt.colorbar(h[3], shrink=0.9, extend='both')
    
    # Plot means and covariance ellipses
    for i in range(len(gmm_result['means'])):
        # Plot mean
        pop_label = ''
        if gmm_result.get('populations') is not None:
            pop_label = f'\n(n={gmm_result["populations"][i]})'
        plt.scatter(gmm_result['means'][i, 0], gmm_result['means'][i, 1], 
                   c=f'C{i}', s=200, marker='x', linewidth=3,
                   label=f'State {i}{pop_label}')
        
        # Add text label
        plt.text(gmm_result['means'][i, 0], gmm_result['means'][i, 1], 
                f' {i}', fontsize=14, color=f'C{i}', fontweight='bold')
        
        # Plot covariance ellipse
        eigenvals, eigenvecs = np.linalg.eigh(gmm_result['covariances'][i])
        angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
        width, height = 4 * np.sqrt(eigenvals)
        ellip = Ellipse(gmm_result['means'][i], width, height, angle, 
                       facecolor='none', edgecolor=f'C{i}', 
                       alpha=0.8, linestyle='--')
        plt.gca().add_patch(ellip)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.gca().set_aspect('equal')
    
    # Save if requested
    if save_path:
        plt.savefig(save_path)
    
    # Show or close
    if show:
        plt.show()
        plt.close(fig)  # Close after showing
    else:
        plt.close(fig)

def print_gmm_parameters(gmm_result: Dict[str, Any]) -> None:
    """
    Print the means and covariances of the GMM.
    
    Parameters
    ----------
    gmm_result : Dict[str, Any]
        The result from fit_gmm().
    
    Returns
    -------
    None
    
    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.randn(1000, 2)
    >>> result = fit_gmm(data, 3)
    >>> print_gmm_parameters(result)
    """
    means = gmm_result['means']
    covariances = gmm_result['covariances']
    num_modes = len(means)
    
    print("\nMeans:")
    for i in range(num_modes):
        print(f"State {i}: {means[i]}")
    
    print("\nCovariance matrices:")
    for i in range(num_modes):
        print(f"\nState {i}:")
        print(covariances[i])

def calculate_all_snrs(hmm_model) -> np.ndarray:
    """
    Calculate SNR between all pairs of states.
    
    Parameters
    ----------
    hmm_model : object
        HMM model with means_ and covars_ attributes
    
    Returns
    -------
    np.ndarray
        Array of SNR values between all pairs of states
    """
    num_modes = len(hmm_model.means_)
    snrs = []
    
    # Calculate SNR for all pairs of states
    for i in range(num_modes):
        for j in range(i + 1, num_modes):
            snr = qp.getSNRhmm(hmm_model, i, j)
            snrs.append(snr)
    
    return np.array(snrs)

def initialize_state_lists(num_modes: int) -> Dict[str, List]:
    """
    Initialize empty lists for storing state-related data.
    
    Parameters
    ----------
    num_modes : int
        Number of states/modes to initialize
        
    Returns
    -------
    Dict[str, List]
        Dictionary containing empty lists for all HMM analysis data
    """
    lists = {
        'MEANS': [],
        'HMM': [],
        'TAUS': [],
        'STARTS': [],
        'SNR': [],
        'Q': [],  # Store state assignments
        'time': [],  # Store time arrays
        'lifetimes': [],  # Store lifetime dictionaries
        'anti_lifetimes': [],  # Store anti-lifetime dictionaries
        'lifetime_constants': [],  # Store fitted lifetime constants
        'anti_lifetime_constants': [],  # Store fitted anti-lifetime constants
    }
    
    # Initialize state probability lists dynamically
    for i in range(num_modes):
        lists[f'P{i}'] = []
    
    return lists

def plot_time_series(
    time: np.ndarray,
    Q: np.ndarray,
    data: np.ndarray,
    sind: int = 1000,
    eind: int = 3000,
    title: str = None,
    save_path: Optional[str] = None
) -> None:
    """
    Plot time series data with HMM state assignments and I/Q data.
    
    Parameters
    ----------
    time : np.ndarray
        Time array
    Q : np.ndarray
        State assignments
    data : np.ndarray
        I/Q data array
    sind : int
        Start index for plotting
    eind : int
        End index for plotting
    title : str, optional
        Plot title
    save_path : Optional[str]
        Path to save the plot
    """
    fig, ax = plt.subplots(2, 1, figsize=[9, 6])
    
    # Plot state assignments
    ax[0].plot(time[sind:eind], Q[sind:eind], label='HMM')
    ax[0].legend()
    ax[0].set_ylabel('Occupation')
    
    # Plot I/Q data
    ax[1].set_ylabel('mV')
    ax[1].set_xlabel('Time [$\mu$s]')
    ax[1].plot(time[sind:eind], data[0, sind:eind], label='real')
    ax[1].plot(time[sind:eind], data[1, sind:eind], label='imag')
    ax[1].legend()
    
    if title:
        ax[0].set_title(title)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close()

def plot_lifetime_histograms(
    lifetimes: Dict[str, List],
    title_prefix: str,
    save_path: Optional[str] = None
) -> Dict[str, float]:
    """
    Plot lifetime histograms and fit exponential decays.
    
    Parameters
    ----------
    lifetimes : Dict[str, List]
        Dictionary of lifetimes for each state
    title_prefix : str
        Prefix for plot titles
    save_path : Optional[str]
        Path to save the plots
    
    Returns
    -------
    Dict[str, float]
        Dictionary of fitted time constants for each state
    """
    time_constants = {}
    for key in lifetimes:
        if len(lifetimes[key]):
            h = qp.fitAndPlotExpDecay(lifetimes[key], 
                                    cut=np.mean(lifetimes[key]), 
                                    bins=200)
            plt.title(f'{title_prefix} mode {key}')
            if save_path:
                plt.savefig(f'{save_path}_{key}.png')
            plt.show()
            plt.close()
            # Store the time constant if available
            if hasattr(h, 'time_constant'):
                time_constants[key] = h.time_constant
    return time_constants

def process_hmm_data(
    data: np.ndarray,
    num_modes: int,
    rate: float,
    previous_model: Optional[Any] = None,
    atten: Optional[float] = None,
    figure_path: Optional[str] = None
) -> Tuple[Any, Dict[str, Any]]:
    """
    Process HMM data with dynamic number of states.
    """
    # Initialize or copy model parameters
    GM = hmm.GaussianHMM(n_components=num_modes, 
                         covariance_type='full',
                         init_params='c',
                         n_iter=500,
                         tol=0.001)
    
    if previous_model is not None:
        GM.startprob_ = np.copy(previous_model.startprob_)
        GM.means_ = np.copy(previous_model.means_)
        GM.transmat_ = np.copy(previous_model.transmat_)
    
    # Fit the model
    GM.fit(data.T)
    
    # Calculate variances for all states
    GMvaris = []
    for i in range(num_modes):
        v, w = np.linalg.eigh(GM.covars_[i])
        u = w[0] / np.linalg.norm(w[0])
        angle = -1 * np.arctan2(u[1], u[0])
        GMvaris.append([np.sqrt(v[0]), np.sqrt(v[1]), angle])
    
    # Plot results
    ax = qp.plotComplexHist(data[0], data[1])
    colors = plt.cm.rainbow(np.linspace(0, 1, num_modes))
    qp.make_ellipses2(GM.means_, np.asarray(GMvaris), ax, colors)
    
    if atten is not None:
        plt.title(f'HMM DA = {atten}')
    
    if figure_path is not None:
        plt.savefig(f'{figure_path}hmmfit_DA{atten}.png')
    
    plt.show()
    plt.close()
    
    # Calculate state probabilities and get state assignments
    logprob, Q = GM.decode(data.T, algorithm='viterbi')
    state_probs = {}
    state_probs['MEANS'] = np.mean(Q)
    
    for i in range(num_modes):
        state_probs[f'P{i}'] = np.sum(Q == i) / Q.size
    
    # Calculate SNRs
    snrs = calculate_all_snrs(GM)
    state_probs['SNR'] = snrs
    
    # Time series analysis
    time = np.arange(len(Q)) / rate
    
    # Plot time series
    if figure_path is not None:
        plot_time_series(
            time=time,
            Q=Q,
            data=data,
            title=f'DA = {atten} dB' if atten is not None else None,
            save_path=f'{figure_path}exampleTimeSeries_DA_{atten}dB.png' if atten is not None else None
        )
    
    # Extract and plot lifetimes
    lt = qp.extractLifetimes(Q, time)
    state_probs['lifetimes'] = lt
    if figure_path is not None:
        lt_constants = plot_lifetime_histograms(
            lt,
            f'LT | DA {atten} dB' if atten is not None else 'LT',
            f'{figure_path}lifetime_DA_{atten}dB' if atten is not None else None
        )
        state_probs['lifetime_constants'] = lt_constants
    
    # Extract and plot anti-lifetimes
    alt = qp.extractAntiLifetimes(Q, time)
    state_probs['anti_lifetimes'] = alt
    if figure_path is not None:
        alt_constants = plot_lifetime_histograms(
            alt,
            f'antiLT | DA {atten} dB' if atten is not None else 'antiLT',
            f'{figure_path}antilifetime_DA_{atten}dB' if atten is not None else None
        )
        state_probs['anti_lifetime_constants'] = alt_constants
    
    # Print lifetime statistics
    for i in range(num_modes):
        print(f'HMM finds {len(lt[str(i)])} {i} QP events')
    
    # Store Q and time arrays
    state_probs['Q'] = Q
    state_probs['time'] = time
    
    return GM, state_probs

def getTauFromTrans(sr,trmat):
    n = np.shape(trmat)[0]
    return [1/sr/(1-trmat[i,i]) for i in range(n)]

def create_physics_based_transition_matrix(n_modes):
    """
    Create a physics-informed transition matrix for n_modes where:
    - High probability to stay in same state
    - Lower states (fewer QPs) are more stable
    - Transitions prefer adjacent states
    - Higher states (more QPs) are less stable
    
    Args:
        n_modes: Number of states/modes
    
    Returns:
        numpy array: Transition matrix
    """
    transmat = np.zeros((n_modes, n_modes))
    
    for i in range(n_modes):
        # Base self-transition probability decreases for higher states
        # Ground state (i=0) is most stable
        self_prob = 0.999 - (i * 0.005)
        
        if i == 0:  # Ground state
            # Very high probability to stay in ground state
            transmat[i, i] = self_prob
            # Small probability to transition to first excited state
            transmat[i, i+1] = 1 - self_prob
            
        elif i == n_modes-1:  # Highest state
            # Higher probability to decay
            transmat[i, i] = 0.9
            # Distribute remaining probability to lower states
            # More likely to decay to adjacent state
            remaining_prob = 0.1
            for j in range(i):
                transmat[i, j] = remaining_prob / (2**(i-j))
            # Normalize
            transmat[i] = transmat[i] / np.sum(transmat[i])
            
        else:  # Intermediate states
            transmat[i, i] = self_prob
            # Small probability to go up one state
            up_prob = 0.001
            transmat[i, i+1] = up_prob
            # Remaining probability distributed to lower states
            remaining_prob = 1 - self_prob - up_prob
            for j in range(i):
                transmat[i, j] = remaining_prob / (2**(i-j))
            # Normalize
            transmat[i] = transmat[i] / np.sum(transmat[i])
    
    return transmat

def generate_mode_probabilities(
    n_modes: int,
    base_prob: float = 0.98,
    decay_factor: float = 0.17,
    normalize: bool = True
) -> np.ndarray:
    """
    Generate a probability distribution for N modes following a decreasing trend.
    
    Parameters
    ----------
    n_modes : int
        The number of modes to generate probabilities for.
    base_prob : float, optional
        The probability for the first mode, by default 0.98.
    decay_factor : float, optional
        The factor by which each subsequent mode's probability decreases, by default 0.17.
    normalize : bool, optional
        Whether to normalize the probabilities to sum to 1, by default True.
    
    Returns
    -------
    np.ndarray
        An array of probabilities for each mode.
    
    Examples
    --------
    >>> generate_mode_probabilities(3)
    array([0.83760684, 0.14523529, 0.01715787])
    """
    if n_modes < 1:
        raise ValueError("Number of modes must be at least 1")
    
    # Generate probabilities following the pattern
    probs = np.zeros(n_modes)
    probs[0] = base_prob
    
    for i in range(1, n_modes):
        probs[i] = probs[i-1] * decay_factor
    
    # Normalize if requested
    if normalize:
        probs = probs / np.sum(probs)
    
    return probs

def generate_custom_mode_probabilities(
    n_modes: int,
    probabilities: List[float],
    normalize: bool = True
) -> np.ndarray:
    """
    Generate a probability distribution for N modes using custom probabilities.
    
    Parameters
    ----------
    n_modes : int
        The number of modes to generate probabilities for.
    probabilities : List[float]
        A list of probabilities for each mode. If shorter than n_modes,
        the remaining probabilities will be calculated using a decay factor.
    normalize : bool, optional
        Whether to normalize the probabilities to sum to 1, by default True.
    
    Returns
    -------
    np.ndarray
        An array of probabilities for each mode.
    
    Examples
    --------
    >>> generate_custom_mode_probabilities(3, [0.98, 0.17, 0.003])
    array([0.83760684, 0.14523529, 0.01715787])
    """
    if n_modes < 1:
        raise ValueError("Number of modes must be at least 1")
    
    if len(probabilities) > n_modes:
        raise ValueError("Number of provided probabilities exceeds the number of modes")
    
    # Initialize the result array
    probs = np.zeros(n_modes)
    
    # Fill in the provided probabilities
    for i in range(min(len(probabilities), n_modes)):
        probs[i] = probabilities[i]
    
    # If we need more probabilities, calculate them using a decay factor
    if len(probabilities) < n_modes and len(probabilities) > 0:
        # Calculate the decay factor based on the last two provided probabilities
        if len(probabilities) >= 2:
            decay_factor = probabilities[-1] / probabilities[-2]
        else:
            # If only one probability is provided, use a default decay factor
            decay_factor = 0.17
        
        # Fill in the remaining probabilities
        for i in range(len(probabilities), n_modes):
            probs[i] = probs[i-1] * decay_factor
    
    # Normalize if requested
    if normalize:
        probs = probs / np.sum(probs)
    
    return probs

def generate_exponential_mode_probabilities(
    n_modes: int,
    base_prob: float = 0.98,
    decay_rate: float = 0.8,
    normalize: bool = True
) -> np.ndarray:
    """
    Generate a probability distribution for N modes following an exponential decay.
    
    Parameters
    ----------
    n_modes : int
        The number of modes to generate probabilities for.
    base_prob : float, optional
        The probability for the first mode, by default 0.98.
    decay_rate : float, optional
        The exponential decay rate, by default 0.8.
    normalize : bool, optional
        Whether to normalize the probabilities to sum to 1, by default True.
    
    Returns
    -------
    np.ndarray
        An array of probabilities for each mode.
    
    Examples
    --------
    >>> generate_exponential_mode_probabilities(3)
    array([0.83760684, 0.13401709, 0.02837607])
    """
    if n_modes < 1:
        raise ValueError("Number of modes must be at least 1")
    
    # Generate probabilities following an exponential decay
    indices = np.arange(n_modes)
    probs = base_prob * (decay_rate ** indices)
    
    # Normalize if requested
    if normalize:
        probs = probs / np.sum(probs)
    
    return probs

def plot_mode_probabilities(
    probabilities: np.ndarray,
    title: str = "Mode Probability Distribution",
    save_path: Optional[str] = None
) -> None:
    """
    Plot the mode probability distribution.
    
    Parameters
    ----------
    probabilities : np.ndarray
        An array of probabilities for each mode.
    title : str, optional
        The title of the plot, by default "Mode Probability Distribution".
    save_path : Optional[str], optional
        The path to save the plot to, by default None.
    
    Returns
    -------
    None
    """
    plt.figure(figsize=(10, 6))
    modes = np.arange(1, len(probabilities) + 1)
    plt.bar(modes, probabilities, color='skyblue', alpha=0.7)
    plt.plot(modes, probabilities, 'ro-', linewidth=2, markersize=8)
    
    plt.xlabel('Mode')
    plt.ylabel('Probability')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.xticks(modes)
    
    # Add probability values on top of each bar
    for i, prob in enumerate(probabilities):
        plt.text(i + 1, prob + 0.01, f'{prob:.4f}', ha='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def apply_to_hmm(
    hmm_model,
    n_modes: int,
    method: str = 'default',
    **kwargs
) -> None:
    """
    Apply the generated probabilities to an HMM model's startprob_ attribute.
    
    Parameters
    ----------
    hmm_model : object
        The HMM model with a startprob_ attribute.
    n_modes : int
        The number of modes to generate probabilities for.
    method : str, optional
        The method to use for generating probabilities, by default 'default'.
        Options: 'default', 'custom', 'exponential'.
    **kwargs : dict
        Additional arguments to pass to the probability generation function.
    
    Returns
    -------
    None
    """
    if method == 'default':
        probs = generate_mode_probabilities(n_modes, **kwargs)
    elif method == 'custom':
        if 'probabilities' not in kwargs:
            raise ValueError("'probabilities' must be provided for the 'custom' method")
        probs = generate_custom_mode_probabilities(n_modes, **kwargs)
    elif method == 'exponential':
        probs = generate_exponential_mode_probabilities(n_modes, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Apply the probabilities to the HMM model
    hmm_model.startprob_ = probs

def plot_hmm_results(
    attens: np.ndarray,
    state_lists: Dict[str, List],
    num_modes: int,
    save_path: Optional[str] = None
) -> None:
    """
    Plot HMM analysis results with dynamic number of states.
    
    Parameters
    ----------
    attens : np.ndarray
        Array of attenuation values
    state_lists : Dict[str, List]
        Dictionary containing state data from HMM analysis
    num_modes : int
        Number of states/modes
    save_path : Optional[str]
        Path to save the plots, by default None
    """
    try:
        # Close any existing figures
        plt.close('all')
        
        # Plot 1: Mean occupation
        plt.figure(figsize=(10, 6))
        plt.plot(attens, state_lists['MEANS'])
        plt.title('Mean Occupation')
        plt.xlabel('Attenuation [dB]')
        plt.ylabel('Mean State')
        plt.grid(True, alpha=0.3)
        if save_path:
            plt.savefig(f'{save_path}/mean_occupation.png')
        plt.show()
        plt.close()
    except Exception as e:
        print(f"An error occurred while plotting mean occupation: {e}")

    try:
        # Plot 2: Occupation probabilities
        plt.figure(figsize=(10, 6))
        colors = plt.cm.rainbow(np.linspace(0, 1, num_modes))
        for i in range(num_modes):
            plt.plot(attens, state_lists[f'P{i}'], 
                    label=f'P{i}', 
                    color=colors[i])
        plt.title('Occupation Probabilities')
        plt.xlabel('Attenuation [dB]')
        plt.ylabel('Probability')
        plt.legend()
        plt.grid(True, alpha=0.3)
        if save_path:
            plt.savefig(f'{save_path}/occupation_probabilities.png')
        plt.show()
        plt.close()
    except Exception as e:
        print(f"An error occurred while plotting occupation probabilities: {e}")

    try:
        # Plot 3: Mode lifetimes
        plt.figure(figsize=(10, 6))
        TAUS = np.array(state_lists['TAUS'])
        for i in range(num_modes):
            plt.plot(attens, TAUS[:, i], 
                    label=f'$\\tau_{i}$', 
                    color=colors[i])
        plt.title('Mode Lifetimes')
        plt.xlabel('Attenuation [dB]')
        plt.ylabel('Lifetime')
        plt.legend()
        plt.grid(True, alpha=0.3)
        if save_path:
            plt.savefig(f'{save_path}/mode_lifetimes.png')
        plt.show()
        plt.close()
    except Exception as e:
        print(f"An error occurred while plotting mode lifetimes: {e}")
    
    try:
        # Plot 4: SNR between modes
        plt.figure(figsize=(10, 6))
        SNR = np.array(state_lists['SNR'])
        snr_idx = 0
        for i in range(num_modes):
            for j in range(i + 1, num_modes):
                plt.plot(attens, SNR[:, snr_idx], 
                        label=f'$SNR_{{{i}{j}}}$',
                        color=plt.cm.rainbow(snr_idx / (num_modes * (num_modes - 1) / 2)))
                snr_idx += 1
        plt.title('Mode SNR')
        plt.xlabel('Attenuation [dB]')
        plt.ylabel('SNR')
        plt.legend()
        plt.grid(True, alpha=0.3)
        if save_path:
            plt.savefig(f'{save_path}/mode_snr.png')
        plt.show()
        plt.close()
    except Exception as e:
        print(f"An error occurred while plotting mode SNR: {e}")

def get_means_covars(
    data: np.ndarray,
    num_modes: int,
    title: str = "Click to select mode centers",
    min_points_per_cluster: int = 1000
) -> Dict[str, Any]:
    """
    Interactive mode center selection and covariance calculation.
    
    Parameters
    ----------
    data : np.ndarray
        The data to analyze. Can be either:
        - shape (n_samples, n_features) for sklearn format
        - shape (n_features, n_samples) for QP format
        Will be automatically transposed if needed.
    num_modes : int
        Number of modes to identify
    title : str, optional
        Plot title for mode selection
    min_points_per_cluster : int, optional
        Minimum number of points to use for covariance calculation
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - means: Selected mode centers
        - covariances: Calculated covariances
        - labels: Point assignments
        - populations: Points per mode
        - model: Fitted GMM model
    """
    # Close any existing figures first
    plt.close('all')
    
    # Check data shape and transpose if needed
    if data.shape[0] == 2 and data.shape[1] > 2:
        # Data is in (n_features, n_samples) format, transpose it
        data = data.T
    elif data.shape[1] == 2 and data.shape[0] > 2:
        # Data is already in (n_samples, n_features) format
        pass
    else:
        raise ValueError(f"Data shape {data.shape} is invalid. Must be either (2, n_samples) or (n_samples, 2)")
    
    # Switch to Qt backend for interactive plotting
    set_plotting_backend('qt')
    
    # Create figure for mode selection
    plt.figure(figsize=(12, 10))
    h = plt.hist2d(data[:, 0], data[:, 1], bins=80, 
                   norm=LogNorm(), cmap='Greys')
    plt.colorbar(h[3], shrink=0.9, extend='both')
    plt.title(f"{title}\nSelect {num_modes} mode centers")
    plt.grid(True, alpha=0.3)
    plt.gca().set_aspect('equal')
    
    # Get user clicks for mode centers
    print(f"\nPlease click on {num_modes} mode centers")
    clicks = plt.ginput(n=num_modes, timeout=-1)
    plt.close('all')
    
    # Convert clicks to numpy array
    means = np.array(clicks)
    
    # Calculate distances to all points
    distances = np.zeros((data.shape[0], num_modes))
    for i in range(num_modes):
        distances[:, i] = np.sum((data - means[i])**2, axis=1)
    
    # Assign points to nearest center
    labels = np.argmin(distances, axis=1)
    
    # Calculate populations
    populations = np.zeros(num_modes, dtype=int)
    for i in range(num_modes):
        populations[i] = np.sum(labels == i)
    
    # Calculate covariances
    covariances = np.zeros((num_modes, 2, 2))
    for i in range(num_modes):
        # Get points assigned to this mode
        mask = (labels == i)
        points = data[mask]
        
        # If too few points, expand search radius
        if len(points) < min_points_per_cluster:
            # Sort distances to this center
            mode_distances = distances[:, i]
            closest_indices = np.argsort(mode_distances)
            # Take the closest min_points_per_cluster points
            points = data[closest_indices[:min_points_per_cluster]]
        
        # Calculate covariance
        covariances[i] = np.cov(points.T)
    
    # Create and fit GMM with fixed means
    gmm = GaussianMixture(
        n_components=num_modes,
        covariance_type='full',
        means_init=means,
        weights_init=populations/np.sum(populations),
        precisions_init=np.array([np.linalg.inv(cov) for cov in covariances])
    )
    
    # Fit with fixed means
    gmm._initialize_parameters(data, np.random.RandomState(42))
    gmm.means_ = means
    gmm.covariances_ = covariances
    gmm.weights_ = populations/np.sum(populations)
    gmm.precisions_cholesky_ = np.array([np.linalg.cholesky(np.linalg.inv(cov)).T 
                                        for cov in covariances])
    
    # Create result dictionary
    result = {
        'means': means,
        'covariances': covariances,
        'labels': labels,
        'populations': populations,
        'model': gmm
    }
    
    # Switch back to inline backend
    set_plotting_backend('inline')
    
    # Show result plot
    print("\nFinal mode assignments:")
    plot_gmm_results(data, result)
    
    return result

def find_atten_file_index(files: List[str], target_atten: int) -> int:
    """
    Find the index of a file with a specific attenuation value.
    
    Parameters
    ----------
    files : List[str]
        List of file paths in format like 'data/phi_0p450/DA00_SR10/L1A_20250414_154730.bin'
    target_atten : int
        Target attenuation value to find (e.g., 0 for 'DA00')
    
    Returns
    -------
    int
        Index of the file with matching attenuation, or -1 if not found
    
    Examples
    --------
    >>> files = ['data/phi_0p450/DA00_SR10/L1A_20250414_154730.bin',
    ...          'data/phi_0p450/DA05_SR10/L1A_20250414_154731.bin']
    >>> find_atten_file_index(files, 0)
    0
    >>> find_atten_file_index(files, 5)
    1
    """
    import re

    # Pattern to match 'DA' followed by digits
    pattern = r'DA(\d+)_'
    
    for i, filepath in enumerate(files):
        match = re.search(pattern, filepath)
        if match:
            file_atten = int(match.group(1))  # Extract and convert to int
            if file_atten == target_atten:
                return i
    
    return -1  # Return -1 if no match found

# Example usage
if __name__ == "__main__":
    # Generate some example data
    np.random.seed(42)
    n_samples = 10000
    
    # Generate data from 3 different Gaussian distributions
    data1 = np.random.randn(n_samples // 3, 2) + np.array([2, 2])
    data2 = np.random.randn(n_samples // 3, 2) + np.array([-2, -2])
    data3 = np.random.randn(n_samples // 3, 2) + np.array([0, 0])
    
    # Combine the data
    data = np.vstack([data1, data2, data3])
    
    # Fit the GMM
    result = fit_gmm(data, 3)
    
    # Plot the results
    plot_gmm_results(data, result)
    
    # Print the parameters
    print_gmm_parameters(result) 

class EllipseDragger:
    def __init__(self, ax, center, initial_radius=1.0):
        self.ax = ax
        self.center = np.array(center)
        self.major_length = initial_radius  # 1σ
        self.minor_length = initial_radius  # 1σ
        self.angle = 0.0  # radians, orientation of major axis

        # Create initial ellipse (1σ)
        self.ellipse = Ellipse(center, 2*initial_radius, 2*initial_radius, 
                              angle=0, fill=False, color='red', linewidth=2)
        self.ax.add_patch(self.ellipse)

        # Create handles for dragging
        self.major_handle = self.ax.plot([center[0], center[0] + initial_radius],
                                       [center[1], center[1]], 'ro-', linewidth=2)[0]
        self.minor_handle = self.ax.plot([center[0], center[0]],
                                       [center[1], center[1] + initial_radius], 'bo-', linewidth=2)[0]

        self.active_handle = None
        self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.ax.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.ax.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        major_dist = np.sqrt((event.xdata - self.major_handle.get_xdata()[1])**2 + 
                           (event.ydata - self.major_handle.get_ydata()[1])**2)
        minor_dist = np.sqrt((event.xdata - self.minor_handle.get_xdata()[1])**2 + 
                           (event.ydata - self.minor_handle.get_ydata()[1])**2)
        if major_dist < 0.1:
            self.active_handle = 'major'
        elif minor_dist < 0.1:
            self.active_handle = 'minor'
        else:
            self.active_handle = None

    def on_motion(self, event):
        if self.active_handle is None or event.inaxes != self.ax:
            return
        dx = event.xdata - self.center[0]
        dy = event.ydata - self.center[1]
        if self.active_handle == 'major':
            self.major_length = np.sqrt(dx**2 + dy**2)
            self.angle = np.arctan2(dy, dx)
        elif self.active_handle == 'minor':
            self.minor_length = np.sqrt(dx**2 + dy**2)
        self.update_ellipse()
        self.ax.figure.canvas.draw_idle()

    def on_release(self, event):
        self.active_handle = None

    def update_ellipse(self):
        # Update ellipse parameters (1σ)
        self.ellipse.set_angle(np.degrees(self.angle))
        self.ellipse.set_width(2 * self.major_length)
        self.ellipse.set_height(2 * self.minor_length)
        # Update handles
        self.major_handle.set_data(
            [self.center[0], self.center[0] + self.major_length * np.cos(self.angle)],
            [self.center[1], self.center[1] + self.major_length * np.sin(self.angle)]
        )
        self.minor_handle.set_data(
            [self.center[0], self.center[0] + self.minor_length * np.cos(self.angle + np.pi/2)],
            [self.center[1], self.center[1] + self.minor_length * np.sin(self.angle + np.pi/2)]
        )

    def get_covariance(self):
        # Convert ellipse parameters to covariance matrix (1σ)
        angle_rad = self.angle
        width = self.major_length
        height = self.minor_length
        R = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                      [np.sin(angle_rad),  np.cos(angle_rad)]])
        D = np.array([[width**2, 0], [0, height**2]])
        return R @ D @ R.T

def covariance_to_ellipse_params(mean, cov, nsig=1.0):
    """
    Convert a 2x2 covariance matrix to ellipse width, height, and angle (degrees).
    Returns (width, height, angle_degrees) for nsig-sigma ellipse (default 1σ).
    """
    import numpy as np
    from numpy.linalg import eigh
    vals, vecs = eigh(cov)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    width, height = 2 * nsig * np.sqrt(vals)
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    return width, height, angle

def get_means_covars_drag(
    data: np.ndarray,
    num_modes: int,
    title: str = "Click to set mean, then drag ellipse handles. Close window when done.",
    min_points_per_cluster: int = 500
) -> Dict[str, Any]:
    """
    For each mode, open a new figure, let the user click to set the mean, show a default circle, and allow adjustment. Show previous means/ellipses for context. No autofit.
    """
    print("fixed")
    plt.close('all')
    if data.shape[0] == 2 and data.shape[1] > 2:
        data = data.T
    elif data.shape[1] == 2 and data.shape[0] > 2:
        pass
    else:
        raise ValueError(f"Data shape {data.shape} is invalid. Must be either (2, n_samples) or (n_samples, 2)")
    set_plotting_backend('qt')
    means = np.zeros((num_modes, 2))
    covariances = np.zeros((num_modes, 2, 2))
    prev_means = []
    prev_covs = []
    for i in range(num_modes):
        fig, ax = plt.subplots(figsize=(12, 10))
        h = ax.hist2d(data[:, 0], data[:, 1], bins=80, norm=LogNorm(), cmap='Greys')
        plt.colorbar(h[3], shrink=0.9, extend='both')
        ax.set_title(f"Mode {i+1}/{num_modes}: Click to set mean, then drag ellipse handles. Close window when done.")
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        # Show previous means and ellipses
        for pm, pc in zip(prev_means, prev_covs):
            ax.plot(pm[0], pm[1], 'x', color='orange', markersize=12, alpha=0.5, label=None)
            from matplotlib.patches import Ellipse
            width, height, angle = covariance_to_ellipse_params(pm, pc, nsig=1.0)
            ellip = Ellipse(pm, width, height, angle, facecolor='none', edgecolor='orange', alpha=0.4, linestyle='--')
            ax.add_patch(ellip)
        print(f"\nMode {i+1}: Please click to set the mean (center) of the ellipse.")
        mean_click = plt.ginput(n=1, timeout=-1)[0]
        means[i] = mean_click
        ax.plot(mean_click[0], mean_click[1], 'rx', markersize=15, label='Mean')
        # Show a default circle (no autofit)
        default_radius = 0.5
        print(f"Now drag the ellipse handles to set the covariance. Close the window when done.")
        dragger = EllipseDragger(ax, means[i], default_radius)
        plt.draw()
        plt.show(block=True)
        covariances[i] = dragger.get_covariance()
        prev_means.append(means[i].copy())
        prev_covs.append(covariances[i].copy())
        plt.close(fig)
    # Calculate distances to all points using Mahalanobis distance
    distances = np.zeros((data.shape[0], num_modes))
    for i in range(num_modes):
        diff = data - means[i]
        inv_cov = np.linalg.inv(covariances[i])
        distances[:, i] = np.sum(diff @ inv_cov * diff, axis=1)
    labels = np.argmin(distances, axis=1)
    populations = np.zeros(num_modes, dtype=int)
    for i in range(num_modes):
        populations[i] = np.sum(labels == i)
    gmm = GaussianMixture(
        n_components=num_modes,
        covariance_type='full',
        means_init=means,
        weights_init=populations/np.sum(populations),
        precisions_init=np.array([np.linalg.inv(cov) for cov in covariances])
    )
    gmm._initialize_parameters(data, np.random.RandomState(42))
    gmm.means_ = means
    gmm.covariances_ = covariances
    gmm.weights_ = populations/np.sum(populations)
    gmm.precisions_cholesky_ = np.array([np.linalg.cholesky(np.linalg.inv(cov)).T for cov in covariances])
    result = {
        'means': means,
        'covariances': covariances,
        'labels': labels,
        'populations': populations,
        'model': gmm
    }
    set_plotting_backend('inline')
    print("\nFinal mode assignments:")
    # Use the same ellipse logic in the result plot for consistency
    def plot_gmm_results_with_user_ellipses(data, gmm_result, **kwargs):
        from matplotlib.patches import Ellipse
        plt.close('all')
        fig = plt.figure(figsize=kwargs.get('figsize', (10, 10)))
        h = plt.hist2d(data[:, 0], data[:, 1], bins=kwargs.get('bins', 80), 
                       norm=LogNorm(), cmap=plt.cm.get_cmap(kwargs.get('cmap', 'Greys')))
        plt.colorbar(h[3], shrink=0.9, extend='both')
        for i in range(len(gmm_result['means'])):
            plt.scatter(gmm_result['means'][i, 0], gmm_result['means'][i, 1], 
                        c=f'C{i}', s=200, marker='x', linewidth=3,
                        label=f'State {i}\n(n={gmm_result["populations"][i]})')
            plt.text(gmm_result['means'][i, 0], gmm_result['means'][i, 1], 
                     f' {i}', fontsize=14, color=f'C{i}', fontweight='bold')
            width, height, angle = covariance_to_ellipse_params(gmm_result['means'][i], gmm_result['covariances'][i], nsig=1.0)
            ellip = Ellipse(gmm_result['means'][i], width, height, angle, 
                            facecolor='none', edgecolor=f'C{i}', 
                            alpha=0.8, linestyle='--')
            plt.gca().add_patch(ellip)
        plt.xlabel(kwargs.get('xlabel', 'I [mV]'))
        plt.ylabel(kwargs.get('ylabel', 'Q [mV]'))
        plt.title(kwargs.get('title', 'I-Q Data with Gaussian Mixture Model\nMeans and 2σ Covariance Ellipses'))
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.gca().set_aspect('equal')
        if kwargs.get('save_path'):
            plt.savefig(kwargs['save_path'])
        if kwargs.get('show', True):
            plt.show()
            plt.close(fig)
        else:
            plt.close(fig)
    plot_gmm_results_with_user_ellipses(data, result)
    return result

def compress_and_delete_folder(folder_path: str) -> str:
    """
    Compress a folder into a zip file and delete the original folder if successful.
    
    Parameters
    ----------
    folder_path : str
        Path to the folder to be compressed and deleted
        
    Returns
    -------
    str
        Path to the created zip file
        
    Raises
    ------
    FileNotFoundError
        If the folder doesn't exist
    PermissionError
        If there are permission issues
    OSError
        For other operating system related errors
    """
    import os
    import shutil
    from pathlib import Path

    # Convert to Path object for better path handling
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    if not folder_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {folder_path}")
    
    # Create zip file path (same name as folder but with .zip extension)
    zip_path = folder_path.with_suffix('.zip')
    
    try:
        # Create the zip file
        shutil.make_archive(
            str(zip_path.with_suffix('')),  # Remove .zip as make_archive adds it
            'zip',
            folder_path.parent,
            folder_path.name
        )
        
        # Verify the zip file was created successfully
        if not zip_path.exists():
            raise OSError("Failed to create zip file")
        
        # Delete the original folder
        shutil.rmtree(folder_path)
        
        return str(zip_path)
        
    except Exception as e:
        # If anything fails, try to clean up the zip file if it was created
        if zip_path.exists():
            try:
                zip_path.unlink()
            except:
                pass
        raise e

def process_single_folder(folder_path: str) -> Tuple[str, bool]:
    try:
        print("Checking:", folder_path)
        zip_path = compress_and_delete_folder(folder_path)
        return (folder_path, True)
    except Exception as e:
        print(f"Error processing {folder_path}: {e}")
        return (folder_path, False)

def get_folder_size(path):
    """Return total size of files in a folder (in bytes)."""
    import os
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total


def process_bad_datasets_parallel(base_path: str, bad_datasets: dict, num_processes: int = None, test: bool = False) -> None:
    """
    Process multiple folders in parallel using multiprocessing.
    
    Parameters
    ----------
    base_path : str
        Base path where the folders are located
    bad_datasets : dict
        Dictionary mapping frequencies to lists of attenuation values
    num_processes : int, optional
        Number of parallel processes to use. If None, uses CPU count - 1
    test : bool, optional
        If True, only print the directories that would be processed, do not compress or delete anything
    """
    import multiprocessing as mp
    import os
    from pathlib import Path
    from typing import Tuple

    # Convert base path to Path object
    base_path = Path(base_path)
    
    # Prepare list of all folders to process
    folders_to_process = []
    for freq, attens in bad_datasets.items():
        freq_str = format_freq(freq)
        for atten in attens:
            folder_name = f"clearing_{freq_str}GHz_{atten}p0dBm"
            folder_path = base_path / folder_name
            print("Checking:", folder_path)
            if folder_path.exists():
                folders_to_process.append(str(folder_path))
    
    if not folders_to_process:
        print("No folders found to process!")
        return
    
    print(f"Found {len(folders_to_process)} folders to process")
    
    if test:
        print("Test mode enabled. The following directories would be compressed and deleted:")
        for folder in folders_to_process:
            print(folder)
        print("No folders were actually compressed or deleted.")
        return
    
    # Calculate total size of all folders before compression
    total_original_size = 0
    for folder in folders_to_process:
        total_original_size += get_folder_size(folder)
    
    # Determine number of processes
    if num_processes is None:
        num_processes = max(1, mp.cpu_count() - 1)
    
    # Process folders in parallel
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(process_single_folder, folders_to_process)
    
    # Report results
    successful = [path for path, success in results if success]
    failed = [path for path, success in results if not success]
    
    # Calculate total size of all created zip files
    total_zip_size = 0
    for folder in successful:
        zip_path = Path(folder).with_suffix('.zip')
        if zip_path.exists():
            total_zip_size += os.path.getsize(zip_path)
    
    storage_saved = total_original_size - total_zip_size
    storage_saved_gb = storage_saved / (1024**3)
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {len(successful)} folders")
    if failed:
        print(f"Failed to process: {len(failed)} folders")
        for path in failed:
            print(f"  - {path}")
    print(f"Total storage saved: {storage_saved_gb:.2f} GB")

def format_freq(freq):
    # Convert string like "8" or "7.5" to "8p00" or "7p50"
    f = float(freq)
    return f"{f:.2f}".replace('.', 'p')

def compress_and_delete_folders_from_list(folder_list, num_processes=None, test=False):
    """
    Compress and delete all folders in the given list, optionally in parallel.

    Parameters
    ----------
    folder_list : list of str
        List of folder paths to compress and delete.
    num_processes : int, optional
        Number of parallel processes to use. If None, uses CPU count - 1.
    test : bool, optional
        If True, only print the directories that would be processed, do not compress or delete anything.
    """
    import multiprocessing as mp
    import os
    from pathlib import Path

    # Prepare list of all folders to process
    folders_to_process = [str(Path(f)) for f in folder_list if Path(f).exists() and Path(f).is_dir()]

    if not folders_to_process:
        print("No folders found to process!")
        return

    print(f"Found {len(folders_to_process)} folders to process")

    if test:
        print("Test mode enabled. The following directories would be compressed and deleted:")
        for folder in folders_to_process:
            print(folder)
        print("No folders were actually compressed or deleted.")
        return

    # Calculate total size of all folders before compression
    total_original_size = 0
    for folder in folders_to_process:
        total_original_size += get_folder_size(folder)

    # Determine number of processes
    if num_processes is None:
        num_processes = max(1, mp.cpu_count() - 1)

    # Process folders in parallel
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(process_single_folder, folders_to_process)

    # Report results
    successful = [path for path, success in results if success]
    failed = [path for path, success in results if not success]

    # Calculate total size of all created zip files
    total_zip_size = 0
    for folder in successful:
        zip_path = Path(folder).with_suffix('.zip')
        if zip_path.exists():
            total_zip_size += os.path.getsize(zip_path)

    storage_saved = total_original_size - total_zip_size
    storage_saved_gb = storage_saved / (1024**3)

    print(f"\nProcessing complete!")
    print(f"Successfully processed: {len(successful)} folders")
    if failed:
        print(f"Failed to process: {len(failed)} folders")
        for path in failed:
            print(f"  - {path}")
    print(f"Total storage saved: {storage_saved_gb:.2f} GB")