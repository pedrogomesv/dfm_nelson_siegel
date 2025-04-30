# src/utils/kalman.py

import numpy as np
import torch
import pyro.distributions as dist
from scipy import linalg

class KalmanFilter:
    """
    Kalman Filter implementation for state-space models.
    
    This implementation uses PyTorch and Pyro for efficient computation when possible,
    with fallback to numpy/scipy for compatibility with existing code.
    """
    
    def __init__(self, transition_matrix, observation_matrix, transition_covariance,
                 observation_covariance, initial_state_mean, initial_state_covariance,
                 use_torch=False):
        """
        Initialize the Kalman Filter.
        
        Args:
            transition_matrix: State transition matrix (F)
            observation_matrix: Observation matrix (H)
            transition_covariance: State transition covariance matrix (Q)
            observation_covariance: Observation covariance matrix (R)
            initial_state_mean: Initial state mean
            initial_state_covariance: Initial state covariance
            use_torch: Whether to use PyTorch implementation (default: False)
        """
        self.use_torch = use_torch
        
        if use_torch:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Convert to PyTorch tensors if they're not already
            self.transition_matrix = self._to_torch_tensor(transition_matrix)
            self.observation_matrix = self._to_torch_tensor(observation_matrix)
            self.transition_covariance = self._to_torch_tensor(transition_covariance)
            self.observation_covariance = self._to_torch_tensor(observation_covariance)
            self.initial_state_mean = self._to_torch_tensor(initial_state_mean)
            self.initial_state_covariance = self._to_torch_tensor(initial_state_covariance)
        else:
            # Store as numpy arrays
            self.transition_matrix = np.asarray(transition_matrix)
            self.observation_matrix = np.asarray(observation_matrix)
            self.transition_covariance = np.asarray(transition_covariance)
            self.observation_covariance = np.asarray(observation_covariance)
            self.initial_state_mean = np.asarray(initial_state_mean)
            self.initial_state_covariance = np.asarray(initial_state_covariance)
        
        # Dimensions
        self.state_dim = self.transition_matrix.shape[0]
        self.obs_dim = self.observation_matrix.shape[0]
    
    def _to_torch_tensor(self, data):
        """Convert data to PyTorch tensor if it's not already"""
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        else:
            return torch.tensor(data, dtype=torch.float64, device=self.device)
    
    def _to_numpy(self, data):
        """Convert PyTorch tensor to numpy array"""
        if isinstance(data, torch.Tensor):
            return data.cpu().detach().numpy()
        return data
    
    def filter(self, observations):
        """
        Apply Kalman filter to the given observations.
        
        Args:
            observations: Array of observations [time, observation_dim]
            
        Returns:
            filtered_state_means: Array of filtered state means
            filtered_state_covariances: Array of filtered state covariances
        """
        if self.use_torch:
            return self._filter_torch(observations)
        else:
            return self._filter_numpy(observations)
    
    def _filter_torch(self, observations):
        """PyTorch implementation of Kalman filter"""
        # Convert observations to tensor if needed
        observations = self._to_torch_tensor(observations)
        
        # Initialize
        n_timesteps = len(observations)
        filtered_state_means = torch.zeros((n_timesteps, self.state_dim), 
                                          dtype=torch.float64, device=self.device)
        filtered_state_covariances = torch.zeros((n_timesteps, self.state_dim, self.state_dim),
                                                dtype=torch.float64, device=self.device)
        
        # Current state
        current_mean = self.initial_state_mean
        current_covariance = self.initial_state_covariance
        
        # Apply Kalman filter recursively
        for t in range(n_timesteps):
            # Prediction step
            predicted_mean = torch.matmul(self.transition_matrix, current_mean)
            predicted_covariance = (torch.matmul(self.transition_matrix, 
                                              torch.matmul(current_covariance, self.transition_matrix.t())) 
                                  + self.transition_covariance)
            
            # Compute Kalman gain
            innovation_covariance = (torch.matmul(self.observation_matrix, 
                                               torch.matmul(predicted_covariance, self.observation_matrix.t())) 
                                   + self.observation_covariance)
            
            # Use Cholesky for numerical stability
            L = torch.linalg.cholesky(innovation_covariance)
            kalman_gain = torch.matmul(predicted_covariance, 
                                    torch.matmul(self.observation_matrix.t(), 
                                               torch.linalg.solve(innovation_covariance, 
                                                               torch.eye(innovation_covariance.shape[0], 
                                                                       device=self.device))))
            
            # Update step
            innovation = observations[t] - torch.matmul(self.observation_matrix, predicted_mean)
            current_mean = predicted_mean + torch.matmul(kalman_gain, innovation)
            current_covariance = predicted_covariance - torch.matmul(kalman_gain, 
                                                                 torch.matmul(self.observation_matrix, 
                                                                           predicted_covariance))
            
            # Store filtered state
            filtered_state_means[t] = current_mean
            filtered_state_covariances[t] = current_covariance
        
        # Convert back to numpy
        return (filtered_state_means.cpu().detach().numpy(), 
                filtered_state_covariances.cpu().detach().numpy())
    
    def _filter_numpy(self, observations):
        """NumPy implementation of Kalman filter (kept for compatibility)"""
        # Initialize storage for filtered states and covariances
        n_timesteps = len(observations)
        filtered_state_means = np.zeros((n_timesteps, self.state_dim))
        filtered_state_covariances = np.zeros((n_timesteps, self.state_dim, self.state_dim))
        
        # Initialize with provided initial state
        current_mean = self.initial_state_mean
        current_covariance = self.initial_state_covariance
        
        # Apply Kalman filter recursively
        for t in range(n_timesteps):
            # Prediction step
            predicted_mean = np.dot(self.transition_matrix, current_mean)
            predicted_covariance = (np.dot(self.transition_matrix, 
                                        np.dot(current_covariance, self.transition_matrix.T)) 
                                  + self.transition_covariance)
            
            # Compute Kalman gain
            innovation_covariance = (np.dot(self.observation_matrix, 
                                         np.dot(predicted_covariance, self.observation_matrix.T)) 
                                   + self.observation_covariance)
            
            kalman_gain = np.dot(predicted_covariance, 
                               np.dot(self.observation_matrix.T, 
                                    linalg.inv(innovation_covariance)))
            
            # Update step
            innovation = observations[t] - np.dot(self.observation_matrix, predicted_mean)
            current_mean = predicted_mean + np.dot(kalman_gain, innovation)
            current_covariance = predicted_covariance - np.dot(kalman_gain, 
                                                            np.dot(self.observation_matrix, 
                                                                 predicted_covariance))
            
            # Store filtered state
            filtered_state_means[t] = current_mean
            filtered_state_covariances[t] = current_covariance
        
        return filtered_state_means, filtered_state_covariances
    
    def smooth(self, observations):
        """
        Apply Kalman smoother to the given observations.
        
        Args:
            observations: Array of observations [time, observation_dim]
            
        Returns:
            smoothed_state_means: Array of smoothed state means
            smoothed_state_covariances: Array of smoothed state covariances
        """
        if self.use_torch:
            return self._smooth_torch(observations)
        else:
            return self._smooth_numpy(observations)
    
    def _smooth_torch(self, observations):
        """PyTorch implementation of Kalman smoother"""
        # Run filter first
        filtered_state_means, filtered_state_covariances = self._filter_torch(observations)
        
        # Convert back to torch tensors
        filtered_state_means = self._to_torch_tensor(filtered_state_means)
        filtered_state_covariances = self._to_torch_tensor(filtered_state_covariances)
        
        # Initialize storage for smoothed states and covariances
        n_timesteps = len(observations)
        smoothed_state_means = torch.zeros_like(filtered_state_means)
        smoothed_state_covariances = torch.zeros_like(filtered_state_covariances)
        
        # Initialize with the last filtered state
        smoothed_state_means[-1] = filtered_state_means[-1]
        smoothed_state_covariances[-1] = filtered_state_covariances[-1]
        
        # Apply Kalman smoother recursively (backward pass)
        for t in range(n_timesteps - 2, -1, -1):
            # Predicted mean and covariance for t+1
            predicted_mean = torch.matmul(self.transition_matrix, filtered_state_means[t])
            predicted_covariance = (torch.matmul(self.transition_matrix, 
                                              torch.matmul(filtered_state_covariances[t], 
                                                        self.transition_matrix.t())) 
                                  + self.transition_covariance)
            
            # Compute smoother gain
            # Use solve instead of inverse for stability
            smoother_gain = torch.matmul(filtered_state_covariances[t], 
                                      torch.matmul(self.transition_matrix.t(), 
                                               torch.linalg.solve(predicted_covariance, 
                                                               torch.eye(predicted_covariance.shape[0], 
                                                                       device=self.device))))
            
            # Update step
            smoothed_state_means[t] = (filtered_state_means[t] 
                                     + torch.matmul(smoother_gain, 
                                                 smoothed_state_means[t+1] - predicted_mean))
            smoothed_state_covariances[t] = (filtered_state_covariances[t] 
                                         + torch.matmul(smoother_gain, 
                                                     torch.matmul(smoothed_state_covariances[t+1] - predicted_covariance, 
                                                               smoother_gain.t())))
        
        # Convert back to numpy
        return (smoothed_state_means.cpu().detach().numpy(), 
                smoothed_state_covariances.cpu().detach().numpy())
    
    def _smooth_numpy(self, observations):
        """NumPy implementation of Kalman smoother (kept for compatibility)"""
        # Run filter first
        filtered_state_means, filtered_state_covariances = self.filter(observations)
        
        # Initialize storage for smoothed states and covariances
        n_timesteps = len(observations)
        smoothed_state_means = np.zeros((n_timesteps, self.state_dim))
        smoothed_state_covariances = np.zeros((n_timesteps, self.state_dim, self.state_dim))
        
        # Initialize with the last filtered state
        smoothed_state_means[-1] = filtered_state_means[-1]
        smoothed_state_covariances[-1] = filtered_state_covariances[-1]
        
        # Apply Kalman smoother recursively (backward pass)
        for t in range(n_timesteps - 2, -1, -1):
            # Predicted mean and covariance for t+1
            predicted_mean = np.dot(self.transition_matrix, filtered_state_means[t])
            predicted_covariance = (np.dot(self.transition_matrix, 
                                        np.dot(filtered_state_covariances[t], self.transition_matrix.T)) 
                                  + self.transition_covariance)
            
            # Compute smoother gain
            smoother_gain = np.dot(filtered_state_covariances[t], 
                                np.dot(self.transition_matrix.T, 
                                     linalg.inv(predicted_covariance)))
            
            # Update step
            smoothed_state_means[t] = (filtered_state_means[t] 
                                     + np.dot(smoother_gain, 
                                           smoothed_state_means[t+1] - predicted_mean))
            smoothed_state_covariances[t] = (filtered_state_covariances[t] 
                                         + np.dot(smoother_gain, 
                                               np.dot(smoothed_state_covariances[t+1] - predicted_covariance, 
                                                    smoother_gain.T)))
        
        return smoothed_state_means, smoothed_state_covariances
    
    def forecast(self, filtered_state, filtered_state_cov, horizon):
        """
        Generate forecasts using the current filtered state.
        
        Args:
            filtered_state: Current filtered state mean
            filtered_state_cov: Current filtered state covariance
            horizon: Forecast horizon
            
        Returns:
            forecasted_state_means: Array of forecasted state means
            forecasted_state_covariances: Array of forecasted state covariances
        """
        if self.use_torch:
            # Convert to torch tensors if they're not already
            filtered_state = self._to_torch_tensor(filtered_state)
            filtered_state_cov = self._to_torch_tensor(filtered_state_cov)
            
            forecasted_state_means = torch.zeros((horizon, self.state_dim), 
                                               dtype=torch.float64, device=self.device)
            forecasted_state_covariances = torch.zeros((horizon, self.state_dim, self.state_dim),
                                                     dtype=torch.float64, device=self.device)
            
            # Initialize with the current filtered state
            current_mean = filtered_state
            current_covariance = filtered_state_cov
            
            # Apply forecast recursively
            for h in range(horizon):
                # Prediction step
                forecasted_mean = torch.matmul(self.transition_matrix, current_mean)
                forecasted_covariance = (torch.matmul(self.transition_matrix, 
                                                   torch.matmul(current_covariance, self.transition_matrix.t())) 
                                       + self.transition_covariance)
                
                # Store forecasted state
                forecasted_state_means[h] = forecasted_mean
                forecasted_state_covariances[h] = forecasted_covariance
                
                # Update for next step
                current_mean = forecasted_mean
                current_covariance = forecasted_covariance
            
            # Convert back to numpy
            return (forecasted_state_means.cpu().detach().numpy(), 
                    forecasted_state_covariances.cpu().detach().numpy())
        else:
            # NumPy implementation
            forecasted_state_means = np.zeros((horizon, self.state_dim))
            forecasted_state_covariances = np.zeros((horizon, self.state_dim, self.state_dim))
            
            # Initialize with the current filtered state
            current_mean = filtered_state
            current_covariance = filtered_state_cov
            
            # Apply forecast recursively
            for h in range(horizon):
                # Prediction step
                forecasted_mean = np.dot(self.transition_matrix, current_mean)
                forecasted_covariance = (np.dot(self.transition_matrix, 
                                             np.dot(current_covariance, self.transition_matrix.T)) 
                                       + self.transition_covariance)
                
                # Store forecasted state
                forecasted_state_means[h] = forecasted_mean
                forecasted_state_covariances[h] = forecasted_covariance
                
                # Update for next step
                current_mean = forecasted_mean
                current_covariance = forecasted_covariance
            
            return forecasted_state_means, forecasted_state_covariances
    
    def update(self, filtered_state, filtered_state_cov, observation):
        """
        Update the filtered state with a new observation.
        
        Args:
            filtered_state: Current filtered state mean
            filtered_state_cov: Current filtered state covariance
            observation: New observation
            
        Returns:
            updated_state: Updated state mean
            updated_state_cov: Updated state covariance
        """
        if self.use_torch:
            # Convert to torch tensors if they're not already
            filtered_state = self._to_torch_tensor(filtered_state)
            filtered_state_cov = self._to_torch_tensor(filtered_state_cov)
            observation = self._to_torch_tensor(observation)
            
            # Prediction step
            predicted_mean = torch.matmul(self.transition_matrix, filtered_state)
            predicted_covariance = (torch.matmul(self.transition_matrix, 
                                              torch.matmul(filtered_state_cov, self.transition_matrix.t())) 
                                  + self.transition_covariance)
            
            # Compute Kalman gain
            innovation_covariance = (torch.matmul(self.observation_matrix, 
                                               torch.matmul(predicted_covariance, self.observation_matrix.t())) 
                                   + self.observation_covariance)
            
            # Use solve instead of inverse for stability
            kalman_gain = torch.matmul(predicted_covariance, 
                                    torch.matmul(self.observation_matrix.t(), 
                                               torch.linalg.solve(innovation_covariance, 
                                                               torch.eye(innovation_covariance.shape[0], 
                                                                       device=self.device))))
            
            # Update step
            innovation = observation - torch.matmul(self.observation_matrix, predicted_mean)
            updated_state = predicted_mean + torch.matmul(kalman_gain, innovation)
            updated_state_cov = predicted_covariance - torch.matmul(kalman_gain, 
                                                                 torch.matmul(self.observation_matrix, 
                                                                           predicted_covariance))
            
            # Convert back to numpy
            return (updated_state.cpu().detach().numpy(), 
                    updated_state_cov.cpu().detach().numpy())
        else:
            # NumPy implementation
            # Prediction step
            predicted_mean = np.dot(self.transition_matrix, filtered_state)
            predicted_covariance = (np.dot(self.transition_matrix, 
                                        np.dot(filtered_state_cov, self.transition_matrix.T)) 
                                  + self.transition_covariance)
            
            # Compute Kalman gain
            innovation_covariance = (np.dot(self.observation_matrix, 
                                         np.dot(predicted_covariance, self.observation_matrix.T)) 
                                   + self.observation_covariance)
            
            kalman_gain = np.dot(predicted_covariance, 
                               np.dot(self.observation_matrix.T, 
                                    linalg.inv(innovation_covariance)))
            
            # Update step
            innovation = observation - np.dot(self.observation_matrix, predicted_mean)
            updated_state = predicted_mean + np.dot(kalman_gain, innovation)
            updated_state_cov = predicted_covariance - np.dot(kalman_gain, 
                                                           np.dot(self.observation_matrix, 
                                                                predicted_covariance))
            
            return updated_state, updated_state_cov