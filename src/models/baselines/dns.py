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
from src.utils.kalman import KalmanFilter

class DynamicNelsonSiegelModel(BaseYieldCurveModel):
    """
    Dynamic Nelson-Siegel model for yield curve forecasting.
    
    This implementation supports both two-step and one-step estimation approaches.
    """
    
    def __init__(self, maturities, method='two-step', lambda_fixed=0.0609, n_factors=3, device=None):
        """
        Initialize the Dynamic Nelson-Siegel model.
        
        Args:
            maturities: List of maturities in months
            method: Estimation method ('one-step' or 'two-step')
            lambda_fixed: Fixed lambda parameter for two-step approach (if None, will be estimated)
            n_factors: Number of factors (default: 3)
            device: PyTorch device to use (default: None, auto-detect)
        """
        name = f"DNS_{method}"
        super().__init__(name, maturities)
        
        self.method = method
        self.lambda_fixed = lambda_fixed
        self.n_factors = n_factors
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.lambda_value = None
        self.transition_matrix = None
        self.transition_covariance = None
        self.observation_covariance = None
        self.initial_state = None
        self.initial_state_covariance = None
        self.observation_matrix = None
        self.data_mean = None
        
        self.filtered_factors = None
        self.smoothed_factors = None
    
    def _create_factor_loadings(self, lambda_value):
        """
        Create Nelson-Siegel factor loadings for given lambda.
        
        Args:
            lambda_value: Lambda parameter
            
        Returns:
            Array of factor loadings [maturities, factors]
        """
        maturities_years = np.array(self.maturities) / 12
        
        loadings = np.zeros((len(self.maturities), self.n_factors))
        
        loadings[:, 0] = 1.0
        
        for i, m in enumerate(maturities_years):
            loadings[i, 1] = (1 - np.exp(-lambda_value * m)) / (lambda_value * m)
        
        for i, m in enumerate(maturities_years):
            loadings[i, 2] = (1 - np.exp(-lambda_value * m)) / (lambda_value * m) - np.exp(-lambda_value * m)
        
        return loadings
    
    def _extract_factors_ols(self, yield_data, lambda_value):
        """
        Extract Nelson-Siegel factors using OLS for two-step approach.
        
        Args:
            yield_data: Yield curve data
            lambda_value: Lambda parameter
            
        Returns:
            Array of extracted factors
        """
        loadings = self._create_factor_loadings(lambda_value)
        
        factors = np.zeros((len(yield_data), self.n_factors))
        
        for t in range(len(yield_data)):
            factors[t] = np.linalg.solve(loadings.T @ loadings, loadings.T @ yield_data.iloc[t].values)
        
        return factors, loadings
    
    def _grid_search_lambda(self, yield_data, lambda_grid=None):
        """
        Grid search for optimal lambda value.
        
        Args:
            yield_data: Yield curve data
            lambda_grid: Grid of lambda values to search
            
        Returns:
            Optimal lambda value
        """
        if lambda_grid is None:
            lambda_grid = np.linspace(0.01, 0.2, 20)
        
        best_lambda = None
        min_sse = float('inf')
        
        print(f"Grid searching for optimal lambda value...")
        
        for lambda_value in lambda_grid:
            factors, loadings = self._extract_factors_ols(yield_data, lambda_value)
            
            reconstructed = factors @ loadings.T
            
            sse = np.sum((yield_data.values - reconstructed) ** 2)
            
            if sse < min_sse:
                min_sse = sse
                best_lambda = lambda_value
                
            print(f"  Lambda: {lambda_value:.4f}, SSE: {sse:.6f}")
        
        print(f"Best lambda: {best_lambda:.4f} with SSE: {min_sse:.6f}")
        
        return best_lambda
    
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
        Fit DNS model using two-step approach.
        
        Args:
            train_data: Training data
            
        Returns:
            self
        """
        print(f"Fitting DNS model with two-step approach...")
        
        self.data_mean = train_data.values.mean(axis=0)
        
        if self.lambda_fixed is None:
            self.lambda_value = self._grid_search_lambda(train_data)
        else:
            print(f"Using fixed lambda: {self.lambda_fixed}")
            self.lambda_value = self.lambda_fixed
        
        factors, loadings = self._extract_factors_ols(train_data, self.lambda_value)
        
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
        
        print(f"DNS two-step fitting complete.")
        
        return self
    
    def fit_one_step(self, train_data):
        """
        Fit DNS model using one-step approach with PyTorch/Pyro.
        
        Args:
            train_data: Training data
            
        Returns:
            self
        """
        print(f"Fitting DNS model with one-step approach using PyTorch/Pyro...")
        
        self.data_mean = train_data.values.mean(axis=0)
        
        train_tensor = torch.tensor(train_data.values, dtype=torch.float64, device=self.device)
        
        lambda_param = torch.tensor(self.lambda_fixed if self.lambda_fixed is not None else 0.0609, 
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
            
            loadings[:, 1] = (1 - torch.exp(-lambda_param * maturities_years)) / (lambda_param * maturities_years)
            
            loadings[:, 2] = (1 - torch.exp(-lambda_param * maturities_years)) / (lambda_param * maturities_years) - torch.exp(-lambda_param * maturities_years)
            
            initial_mean = torch.zeros(self.n_factors, dtype=torch.float64, device=self.device)
            initial_cov = torch.eye(self.n_factors, dtype=torch.float64, device=self.device)
            
            transition_cov = torch.diag(torch.exp(log_transition_cov))
            observation_cov = torch.diag(torch.exp(log_observation_cov))
            
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
            {'params': [lambda_param], 'lr': 0.01},
            {'params': [transition_matrix], 'lr': 0.01},
            {'params': [log_transition_cov], 'lr': 0.01},
            {'params': [log_observation_cov], 'lr': 0.01}
        ])
        
        svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
        
        n_steps = 200
        for step in range(n_steps):
            loss = svi.step(train_tensor)
            
            if (step + 1) % 20 == 0:
                print(f"Step {step+1}/{n_steps} - Loss: {loss:.4f}, Lambda: {lambda_param.item():.4f}")
        
        self.lambda_value = lambda_param.item()
        self.transition_matrix = transition_matrix.detach().cpu().numpy()
        self.transition_covariance = torch.diag(torch.exp(log_transition_cov)).detach().cpu().numpy()
        self.observation_covariance = torch.diag(torch.exp(log_observation_cov)).detach().cpu().numpy()
        
        self.observation_matrix = self._create_factor_loadings(self.lambda_value)
        
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
        
        print(f"DNS one-step fitting complete.")
        print(f"Estimated lambda: {self.lambda_value:.6f}")
        
        return self

    def fit(self, train_data):
        """
        Fit the DNS model using two-step approach (ignoring one-step option).
        
        Args:
            train_data: Training data as pandas DataFrame
            
        Returns:
            self
        """
        start_time = time.time()
        
        if not isinstance(train_data, pd.DataFrame):
            raise ValueError("train_data must be a pandas DataFrame")
        
        print(f"Note: Using two-step approach for DNS model (one-step disabled)")
        self.fit_two_step(train_data)
        
        self.is_fitted = True
        self.training_time = time.time() - start_time
        
        print(f"DNS model fitted in {self.training_time:.2f} seconds")
        
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
            observation_matrix=self.observation_matrix,  
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
        
        factor_names = ['Level', 'Slope', 'Curvature']
        
        for i in range(self.n_factors):
            plt.plot(maturities_years, self.observation_matrix[:, i], 
                    linewidth=2, label=factor_names[i])
        
        plt.xlabel('Maturity (years)')
        plt.ylabel('Loading')
        plt.title(f'{self.name} Factor Loadings (λ={self.lambda_value:.4f})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True, parents=True)
            plt.savefig(output_path / f'{self.name}_factor_loadings.png', dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_factors(self, output_dir=None):
        """
        Plot the estimated factors.
        
        Args:
            output_dir: Optional directory to save the plot
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before plotting")
        
        plt.figure(figsize=(12, 8))
        
        factor_names = ['Level', 'Slope', 'Curvature']
        
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
            lambda_value=self.lambda_value,
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