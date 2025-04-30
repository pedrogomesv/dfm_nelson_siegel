
import numpy as np
import pandas as pd
import time
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO

from src.models.base_model import BaseYieldCurveModel
from src.models.baselines.dns import DynamicNelsonSiegelModel
from src.utils.kalman import KalmanFilter

class DynamicNelsonSiegelSvenssonModel(BaseYieldCurveModel):
    """
    Dynamic Nelson-Siegel-Svensson model for yield curve forecasting.
    
    This model extends the DNS model with an additional factor and decay parameter.
    """
    
    def __init__(self, maturities, method='two-step', lambda1_fixed=0.0609, lambda2_fixed=0.0292, n_factors=4, device=None):
        """
        Initialize the Dynamic Nelson-Siegel-Svensson model.
        
        Args:
            maturities: List of maturities in months
            method: Estimation method ('one-step' or 'two-step')
            lambda1_fixed: Fixed lambda1 parameter for two-step approach (if None, will be estimated)
            lambda2_fixed: Fixed lambda2 parameter for two-step approach (if None, will be estimated)
            n_factors: Number of factors (default: 4)
            device: PyTorch device to use (default: None, auto-detect)
        """
        name = f"DNSS_{method}"
        super().__init__(name, maturities)
        
        self.method = method
        self.lambda1_fixed = lambda1_fixed
        self.lambda2_fixed = lambda2_fixed
        self.n_factors = n_factors
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.lambda1_value = None
        self.lambda2_value = None
        self.transition_matrix = None
        self.transition_covariance = None
        self.observation_covariance = None
        self.initial_state = None
        self.initial_state_covariance = None
        self.observation_matrix = None
        self.data_mean = None
        
        self.filtered_factors = None
        self.smoothed_factors = None
    
    def _create_factor_loadings(self, lambda_values):
        """
        Create Nelson-Siegel-Svensson factor loadings for given lambdas.
        
        Args:
            lambda_values: Tuple of (lambda1, lambda2) parameters
            
        Returns:
            Array of factor loadings [maturities, factors]
        """
        lambda1, lambda2 = lambda_values
        
        maturities_years = np.array(self.maturities) / 12
        
        loadings = np.zeros((len(self.maturities), self.n_factors))
        
        loadings[:, 0] = 1.0
        
        for i, m in enumerate(maturities_years):
            loadings[i, 1] = (1 - np.exp(-lambda1 * m)) / (lambda1 * m)
        
        for i, m in enumerate(maturities_years):
            loadings[i, 2] = (1 - np.exp(-lambda1 * m)) / (lambda1 * m) - np.exp(-lambda1 * m)
        
        for i, m in enumerate(maturities_years):
            loadings[i, 3] = (1 - np.exp(-lambda2 * m)) / (lambda2 * m) - np.exp(-lambda2 * m)
        
        return loadings
    
    def _extract_factors_ols(self, yield_data, lambda_values):
        """
        Extract Nelson-Siegel-Svensson factors using OLS for two-step approach.
        
        Args:
            yield_data: Yield curve data
            lambda_values: Tuple of (lambda1, lambda2) parameters
            
        Returns:
            Array of extracted factors and loadings
        """
        loadings = self._create_factor_loadings(lambda_values)
        
        factors = np.zeros((len(yield_data), self.n_factors))
        
        for t in range(len(yield_data)):
            factors[t] = np.linalg.solve(loadings.T @ loadings, loadings.T @ yield_data.iloc[t].values)
        
        return factors, loadings
    
    def _grid_search_lambda(self, yield_data, lambda1_grid=None, lambda2_grid=None):
        """
        Grid search for optimal lambda values.
        
        Args:
            yield_data: Yield curve data
            lambda1_grid: Grid of lambda1 values to search
            lambda2_grid: Grid of lambda2 values to search
            
        Returns:
            Optimal lambda values
        """
        if lambda1_grid is None:
            lambda1_grid = np.linspace(0.01, 0.2, 10)
        
        if lambda2_grid is None:
            lambda2_grid = np.linspace(0.01, 0.2, 10)
        
        best_lambda1 = None
        best_lambda2 = None
        min_sse = float('inf')
        
        print(f"Grid searching for optimal lambda values...")
        
        for lambda1 in lambda1_grid:
            for lambda2 in lambda2_grid:
                if abs(lambda1 - lambda2) < 0.01:
                    continue
                
                factors, loadings = self._extract_factors_ols(yield_data, (lambda1, lambda2))
                
                reconstructed = factors @ loadings.T
                
                sse = np.sum((yield_data.values - reconstructed) ** 2)
                
                if sse < min_sse:
                    min_sse = sse
                    best_lambda1 = lambda1
                    best_lambda2 = lambda2
                    
                print(f"  Lambda1: {lambda1:.4f}, Lambda2: {lambda2:.4f}, SSE: {sse:.6f}")
        
        print(f"Best lambda1: {best_lambda1:.4f}, lambda2: {best_lambda2:.4f} with SSE: {min_sse:.6f}")
        
        return best_lambda1, best_lambda2
    
    def _fit_var(self, factors):
        """
        Fit a VAR(1) model to extracted factors.
        
        Args:
            factors: Extracted factors
            
        Returns:
            Transition matrix, transition covariance
        """
        y = factors[1:]
        x = factors[:-1]
        
        transition_matrix = np.linalg.solve(x.T @ x, x.T @ y).T
        
        residuals = y - x @ transition_matrix.T
        
        transition_covariance = (residuals.T @ residuals) / (len(residuals) - 1)
        
        return transition_matrix, transition_covariance
    
    def fit_two_step(self, train_data):
        """
        Fit DNSS model using two-step approach.
        
        Args:
            train_data: Training data
            
        Returns:
            self
        """
        print(f"Fitting DNSS model with two-step approach...")
        
        self.data_mean = train_data.values.mean(axis=0)
        
        if self.lambda1_fixed is None or self.lambda2_fixed is None:
            self.lambda1_value, self.lambda2_value = self._grid_search_lambda(train_data)
        else:
            print(f"Using fixed lambda1: {self.lambda1_fixed}, lambda2: {self.lambda2_fixed}")
            self.lambda1_value = self.lambda1_fixed
            self.lambda2_value = self.lambda2_fixed
        
        factors, loadings = self._extract_factors_ols(train_data, (self.lambda1_value, self.lambda2_value))
        
        transition_matrix, transition_covariance = self._fit_var(factors)
        
        self.transition_matrix = transition_matrix
        self.transition_covariance = transition_covariance
        self.observation_matrix = loadings
        
        residuals = train_data.values - (factors @ loadings.T)
        self.observation_covariance = np.diag(np.var(residuals, axis=0))
        
        self.initial_state = factors[0]
        self.initial_state_covariance = np.eye(self.n_factors) * 0.01
        
        kf = KalmanFilter(
            transition_matrix=self.transition_matrix,
            observation_matrix=self.observation_matrix,
            transition_covariance=self.transition_covariance,
            observation_covariance=self.observation_covariance,
            initial_state_mean=self.initial_state,
            initial_state_covariance=self.initial_state_covariance,
            use_torch=False
        )
        
        self.filtered_factors, _ = kf.filter(train_data.values)
        self.smoothed_factors, _ = kf.smooth(train_data.values)
        
        print(f"DNSS two-step fitting complete")
        return self
    
    def fit(self, train_data):
        """
        Fit the DNSS model using two-step approach (ignoring one-step option).
        
        Args:
            train_data: Training data
        
        Returns:
            self
        """
        start_time = time.time()
        
        print(f"Note: Using two-step approach for DNSS model (one-step disabled)")
        self.fit_two_step(train_data)
        
        self.is_fitted = True
        self.training_time = time.time() - start_time
        
        print(f"DNSS model fitted in {self.training_time:.2f} seconds")
        
        return self
    
    def predict(self, data, horizon):
        """
        Generate forecasts for the given horizon.
        
        Args:
            data: Input data
            horizon: Forecast horizon
            
        Returns:
            Array of forecasts
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        def normalize(x):
            if isinstance(x, pd.Series):
                return x.values - self.data_mean
            return x - self.data_mean
        
        def denormalize(x):
            return x + self.data_mean
        
        kf = KalmanFilter(
            transition_matrix=self.transition_matrix,
            transition_covariance=self.transition_covariance,
            observation_covariance=self.observation_covariance,
            initial_state_mean=self.initial_state,
            initial_state_covariance=self.initial_state_covariance,
            use_torch=False
        )
        
        forecasts = []
        
        current_state = self.filtered_factors[-1]
        current_cov = np.eye(self.n_factors) * 0.01
        
            
        for i in range(len(data) - horizon):
            current_state, current_cov = kf.update(
                current_state, 
                current_cov, 
                normalize(data.iloc[i])
            )
            
            future_states, _ = kf.forecast(current_state, current_cov, horizon)
            
            yield_forecast = denormalize(np.dot(self.observation_matrix, future_states[-1]))
            forecasts.append(yield_forecast)
    
        return np.array(forecasts)
    
    def plot_factors(self, output_dir=None):
        """
        Plot the estimated factors.
        
        Args:
            output_dir: Optional directory to save plots
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before plotting")
        
        plt.figure(figsize=(12, 8))
        
        factor_names = ['Level', 'Slope', 'Curvature 1', 'Curvature 2']
        
        for i in range(self.n_factors):
            plt.plot(self.smoothed_factors[:, i], linewidth=2, label=factor_names[i])
        
        plt.xlabel('Time')
        plt.ylabel('Factor Value')
        plt.title(f'{self.name} Estimated Factors')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True, parents=True)
            plt.savefig(output_path / f'{self.name}_factors.png', dpi=300, bbox_inches='tight')
        
        plt.close()
        
    def plot_factor_loadings(self, output_dir=None):
        """
        Plot the factor loadings.
        
        Args:
            output_dir: Optional directory to save the plot
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before plotting")
        
        plt.figure(figsize=(12, 8))
        
        maturities_years = np.array(self.maturities) / 12
        
        factor_names = ['Level', 'Slope', 'Curvature 1', 'Curvature 2']
        
        for i in range(self.n_factors):
            plt.plot(maturities_years, self.observation_matrix[:, i], 
                    linewidth=2, label=factor_names[i])
        
        plt.xlabel('Maturity (years)')
        plt.ylabel('Loading')
        plt.title(f'{self.name} Factor Loadings (λ1={self.lambda1_value:.4f}, λ2={self.lambda2_value:.4f})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True, parents=True)
            plt.savefig(output_path / f'{self.name}_factor_loadings.png', dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def save(self, output_dir):
        """
        Save the model parameters and plots.
        
        Args:
            output_dir: Directory to save the model
        """
        super().save(output_dir)
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        np.savez(
            output_path / f'{self.name}_model_params.npz',
            lambda1_value=self.lambda1_value,
            lambda2_value=self.lambda2_value,
            transition_matrix=self.transition_matrix,
            transition_covariance=self.transition_covariance,
            observation_covariance=self.observation_covariance,
            initial_state=self.initial_state,
            initial_state_covariance=self.initial_state_covariance,
            observation_matrix=self.observation_matrix,
            data_mean=self.data_mean
        )
        
        if self.filtered_factors is not None:
            np.save(output_path / f'{self.name}_filtered_factors.npy', self.filtered_factors)
        
        if self.smoothed_factors is not None:
            np.save(output_path / f'{self.name}_smoothed_factors.npy', self.smoothed_factors)
        
        self.plot_factor_loadings(output_dir)
        self.plot_factors(output_dir)
    
    def fit_one_step(self, train_data):
        """
        Fit DNSS model using one-step approach with PyTorch/Pyro.
        
        Args:
            train_data: Training data
            
        Returns:
            self
        """
        print(f"Fitting DNSS model with one-step approach using PyTorch/Pyro...")
        
        self.data_mean = train_data.values.mean(axis=0)
        
        train_tensor = torch.tensor(train_data.values, dtype=torch.float64, device=self.device)
        
        lambda1_param = torch.tensor(self.lambda1_fixed if self.lambda1_fixed is not None else 0.0609, 
                                    dtype=torch.float64, device=self.device, requires_grad=True)
        
        lambda2_param = torch.tensor(self.lambda2_fixed if self.lambda2_fixed is not None else 0.0292, 
                                    dtype=torch.float64, device=self.device, requires_grad=True)
        
        transition_matrix = torch.eye(self.n_factors, dtype=torch.float64, device=self.device)
        transition_matrix.requires_grad = True
        
        log_transition_cov = torch.log(torch.ones(self.n_factors, dtype=torch.float64, device=self.device) * 0.001)
        log_transition_cov.requires_grad = True
        
        log_observation_cov = torch.log(torch.ones(len(self.maturities), dtype=torch.float64, device=self.device) * 0.001)
        log_observation_cov.requires_grad = True
        
        def model(data):
            batch_size, obs_dim = data.shape
            
            maturities_years = torch.tensor(np.array(self.maturities) / 12, dtype=torch.float64, device=self.device)
            loadings = torch.zeros((obs_dim, self.n_factors), dtype=torch.float64, device=self.device)
            
            loadings[:, 0] = 1.0
            
            loadings[:, 1] = (1 - torch.exp(-lambda1_param * maturities_years)) / (lambda1_param * maturities_years)
            
            loadings[:, 2] = (1 - torch.exp(-lambda1_param * maturities_years)) / (lambda1_param * maturities_years) - torch.exp(-lambda1_param * maturities_years)
            
            loadings[:, 3] = (1 - torch.exp(-lambda2_param * maturities_years)) / (lambda2_param * maturities_years) - torch.exp(-lambda2_param * maturities_years)
            
            initial_mean = torch.zeros(self.n_factors, dtype=torch.float64, device=self.device)
            initial_cov = torch.eye(self.n_factors, dtype=torch.float64, device=self.device)
            
            transition_cov = torch.diag(torch.exp(log_transition_cov))
            observation_cov = torch.diag(torch.exp(log_observation_cov))
            
            lambda_diff = torch.abs(lambda1_param - lambda2_param)
            pyro.factor("lambda_diff_prior", -0.5 * (torch.max(torch.tensor(0.0, device=self.device), 
                                                          0.03 - lambda_diff) / 0.01) ** 2)
            
            hmm = dist.GaussianHMM(
                hidden_dim=self.n_factors,
                obs_dim=obs_dim,
                init_dist=dist.MultivariateNormal(initial_mean, initial_cov),
                trans_matrix=transition_matrix,
                trans_dist=dist.MultivariateNormal(
                    torch.zeros(self.n_factors, dtype=torch.float64, device=self.device),
                    transition_cov
                ),
                obs_matrix=loadings.t(),
                obs_dist=dist.MultivariateNormal(
                    torch.zeros(obs_dim, dtype=torch.float64, device=self.device),
                    observation_cov
                ),
                duration=batch_size
            )
            
            with pyro.plate('data', batch_size):
                return pyro.sample('obs', hmm, obs=data)
        
        def guide(data):
            pass
        
        optimizer = pyro.optim.Adam([
            {'params': [lambda1_param, lambda2_param], 'lr': 0.01},
            {'params': [transition_matrix], 'lr': 0.01},
            {'params': [log_transition_cov], 'lr': 0.01},
            {'params': [log_observation_cov], 'lr': 0.01}
        ])
        
        svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
        
        n_steps = 200
        for step in range(n_steps):
            loss = svi.step(train_tensor)
            
            if (step + 1) % 20 == 0:
                print(f"Step {step+1}/{n_steps} - Loss: {loss:.4f}, Lambda1: {lambda1_param.item():.4f}, Lambda2: {lambda2_param.item():.4f}")
        
        self.lambda1_value = lambda1_param.item()
        self.lambda2_value = lambda2_param.item()
        self.transition_matrix = transition_matrix.detach().cpu().numpy()
        self.transition_covariance = torch.diag(torch.exp(log_transition_cov)).detach().cpu().numpy()
        self.observation_covariance = torch.diag(torch.exp(log_observation_cov)).detach().cpu().numpy()
        
        self.observation_matrix = self._create_factor_loadings((self.lambda1_value, self.lambda2_value))
        
        self.initial_state = np.zeros(self.n_factors)
        self.initial_state_covariance = np.eye(self.n_factors)
        
        kf = KalmanFilter(
            transition_matrix=self.transition_matrix,
            observation_matrix=self.observation_matrix,
            transition_covariance=self.transition_covariance,
            observation_covariance=self.observation_covariance,
            initial_state_mean=self.initial_state,
            initial_state_covariance=self.initial_state_covariance,
            use_torch=False
        )
        
        self.filtered_factors, _ = kf.filter(train_data.values)
        self.smoothed_factors, _ = kf.smooth(train_data.values)
        
        print(f"DNSS one-step fitting complete.")
        print(f"Estimated lambda1: {self.lambda1_value:.6f}, lambda2: {self.lambda2_value:.6f}")
        
        return self