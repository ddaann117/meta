import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Optional, Union, List, Dict, Any, Callable
import numpy as np
import copy
import time
import os
import json
import logging
import datetime
import traceback
import uuid
import gc
from pathlib import Path
from contextlib import contextmanager

# Import configuration and safety systems
from eeg_config_loader import (
    get_config, is_safety_enabled, get_system_path, 
    update_config, config_manager
)
from safety_system import (
    safe_tensor_op, safe_training, safe_execution,
    check_connection_safety, check_redundancy, 
    safety_system, SafetyViolation
)

# Set up logger
logger = logging.getLogger("EEGModel")

# =====================================
# Model Components
# =====================================

class AdaptiveConnectionBlock(nn.Module):
    """Adaptive connection block that can self-modify its connectivity patterns."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        name: str = "connection_block"
    ):
        """
        Initialize adaptive connection block.
        
        Args:
            input_size: Number of input features
            hidden_size: Number of hidden features
            name: Name for this connection block
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.name = name
        
        # Get configuration parameters
        self.connection_growth_rate = get_config("model.self_improvement.connection_growth_rate", 0.01)
        self.pruning_threshold = get_config("model.self_improvement.pruning_threshold", 0.001)
        
        # Initialize logger
        self.logger = logging.getLogger(f"AdaptiveConnection.{name}")
        
        # Connectivity mask (1 = connection exists, 0 = no connection)
        self.register_buffer("connectivity_mask", torch.ones(hidden_size, input_size))
        
        # Connection strength tracker
        self.register_buffer("connection_importance", torch.ones(hidden_size, input_size))
        
        # Adaptive projection layer
        self.projection = nn.Linear(input_size, hidden_size)
        
        # Connection history for meta-learning
        self.connection_history = []
        
        self.logger.info(f"Initialized with {input_size} inputs, {hidden_size} outputs")
    
    @safe_tensor_op()
    def update_connections(self, gradient_flow: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Update connection importance based on gradient flow.
        
        Args:
            gradient_flow: Optional tensor of gradient information
            
        Returns:
            Statistics about the update
        """
        with torch.no_grad():
            # Update importance based on gradient magnitude
            if self.projection.weight.grad is not None:
                grad_magnitude = torch.abs(self.projection.weight.grad)
            else:
                # If no gradient yet, use small random values
                grad_magnitude = torch.rand_like(self.projection.weight) * 0.001
            
            # Update exponential moving average of importance
            self.connection_importance = 0.9 * self.connection_importance + 0.1 * grad_magnitude
            
            # Prune weak connections
            weak_connections = self.connection_importance < self.pruning_threshold
            self.connectivity_mask[weak_connections] = 0
            
            # Randomly grow new connections
            potential_new_connections = torch.rand_like(self.connectivity_mask) < self.connection_growth_rate
            potential_new_connections = potential_new_connections & (self.connectivity_mask == 0)
            self.connectivity_mask[potential_new_connections] = 1
            
            # Apply mask to weights
            masked_weights = self.projection.weight * self.connectivity_mask
            self.projection.weight.data.copy_(masked_weights)
            
            # Record connectivity statistics for meta-learning
            stats = {
                'density': self.connectivity_mask.mean().item(),
                'pruned': weak_connections.sum().item(),
                'added': potential_new_connections.sum().item(),
                'timestamp': time.time()
            }
            self.connection_history.append(stats)
            
            # Check connection safety if enabled
            if is_safety_enabled("connection_safety"):
                check_connection_safety(self.projection.weight.data, self.name)
            
            # Check redundancy if enabled
            if is_safety_enabled("redundancy_checking"):
                valid, redundancy_score = check_redundancy(self.projection.weight.data, self.name)
                stats['redundancy_score'] = redundancy_score
            
            # Log significant changes
            if stats['pruned'] > 0 or stats['added'] > 0:
                self.logger.debug(f"Updated connections: density={stats['density']:.4f}, "
                                f"pruned={stats['pruned']}, added={stats['added']}")
            
            return stats
    
    @safe_tensor_op()
    def optimize_redundancy(self) -> Dict[str, Any]:
        """
        Optimize connections by reducing redundancy.
        
        Returns:
            Optimization statistics
        """
        with torch.no_grad():
            # Only perform if redundancy checking is enabled
            if not is_safety_enabled("redundancy_checking"):
                return {"status": "redundancy_checking_disabled"}
                
            # Get redundancy threshold from config
            threshold = get_config("safety.redundancy_threshold", 0.95)
                
            # Calculate correlation matrix between output neurons
            norm_weights = self.projection.weight.data
            norm_weights = norm_weights - norm_weights.mean(dim=1, keepdim=True)
            norm_weights = norm_weights / (norm_weights.norm(dim=1, keepdim=True) + 1e-8)
            correlation = torch.mm(norm_weights, norm_weights.t())
            
            # Set diagonal to zero (self-correlation)
            correlation.fill_diagonal_(0)
            
            # Find highly correlated neurons
            highly_correlated = correlation > threshold
            
            # Calculate redundancy statistics
            num_redundant = highly_correlated.sum().item() // 2  # Count pairs once
            redundancy_score = correlation[highly_correlated].mean().item() if highly_correlated.sum() > 0 else 0.0
            
            # Only optimize if there are redundant connections
            if num_redundant == 0:
                return {
                    "status": "no_redundancy_found",
                    "redundancy_score": redundancy_score,
                    "threshold": threshold
                }
            
            # Create a mask for connections to modify
            modify_mask = torch.zeros_like(self.projection.weight.data, dtype=torch.bool)
            
            # For each pair of highly correlated neurons, slightly modify one of them
            for i in range(self.hidden_size):
                for j in range(i+1, self.hidden_size):
                    if correlation[i, j] > threshold:
                        # Modify the neuron with fewer nonzero connections
                        nonzeros_i = (self.projection.weight.data[i] != 0).sum()
                        nonzeros_j = (self.projection.weight.data[j] != 0).sum()
                        
                        if nonzeros_i <= nonzeros_j:
                            modify_mask[i] = True
                        else:
                            modify_mask[j] = True
            
            # Apply small random perturbations to break symmetry for redundant neurons
            perturbation = torch.randn_like(self.projection.weight.data) * 0.01
            optimized_weights = self.projection.weight.data.clone()
            optimized_weights[modify_mask] += perturbation[modify_mask]
            
            # Apply mask to maintain sparse connectivity
            optimized_weights = optimized_weights * self.connectivity_mask
            
            # Update weights
            self.projection.weight.data.copy_(optimized_weights)
            
            # Check new redundancy
            _, new_redundancy_score = check_redundancy(self.projection.weight.data, self.name)
            
            # Prepare optimization statistics
            opt_stats = {
                "status": "optimization_performed",
                "modified_neurons": modify_mask.sum().item(),
                "before_redundancy": redundancy_score,
                "after_redundancy": new_redundancy_score,
                "redundancy_reduction": redundancy_score - new_redundancy_score,
                "timestamp": time.time()
            }
            
            # Log optimization results
            self.logger.info(f"Optimized redundancy: "
                           f"from {opt_stats['before_redundancy']:.4f} to {opt_stats['after_redundancy']:.4f}")
            
            return opt_stats
    
    @safe_tensor_op()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with masked connections.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        # Validate input
        if x.dim() != 2 or x.size(1) != self.input_size:
            raise ValueError(
                f"Expected input shape (batch_size, {self.input_size}), "
                f"got {tuple(x.shape)}"
            )
        
        # Apply masked projection
        return self.projection(x)


class SelfImprovingEEGLSTM(nn.Module):
    """Self-improving LSTM model for EEG state prediction with adaptive connections."""
    
    def __init__(
        self, 
        input_size: Optional[int] = None, 
        hidden_size: Optional[int] = None, 
        output_size: Optional[int] = None, 
        num_layers: Optional[int] = None,
        dropout: Optional[float] = None,
        bidirectional: Optional[bool] = None,
        model_name: Optional[str] = None
    ):
        """
        Initialize self-improving LSTM model using config values or explicit parameters.
        
        Args:
            input_size: Number of expected features in the input (or None to use config)
            hidden_size: Number of features in the hidden state (or None to use config)
            output_size: Number of output classes (or None to use config)
            num_layers: Number of recurrent layers (or None to use config)
            dropout: Dropout probability (or None to use config)
            bidirectional: If True, use a bidirectional LSTM (or None to use config)
            model_name: Name for this model (or None to generate)
        """
        super().__init__()
        
        # Get parameters from config if not explicitly provided
        self.input_size = input_size or get_config("model.input_size", 64)
        self.hidden_size = hidden_size or get_config("model.hidden_size", 128)
        self.output_size = output_size or get_config("model.output_size", 5)
        self.num_layers = num_layers or get_config("model.num_layers", 2)
        self.dropout = dropout if dropout is not None else get_config("model.dropout", 0.2)
        self.bidirectional = bidirectional if bidirectional is not None else get_config("model.bidirectional", True)
        self.num_directions = 2 if self.bidirectional else 1
        self.model_name = model_name or f"EEG_LSTM_{uuid.uuid4().hex[:8]}"
        
        # Meta-learning parameters from config
        self.meta_learning_rate = get_config("model.self_improvement.meta_learning_rate", 0.001)
        
        # Set up logging
        self.logger = logging.getLogger(f"SelfImprovingEEGLSTM.{self.model_name}")
        self.logger.info(f"Initializing model with input_size={self.input_size}, hidden_size={self.hidden_size}, "
                        f"output_size={self.output_size}, num_layers={self.num_layers}")
        
        # Adaptive input projection
        self.input_adapter = AdaptiveConnectionBlock(
            input_size=self.input_size,
            hidden_size=self.input_size,
            name=f"{self.model_name}_input"
        )
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
            bidirectional=self.bidirectional
        )
        
        # Adaptive output projection
        fc_input_size = self.hidden_size * self.num_directions
        self.output_adapter = AdaptiveConnectionBlock(
            input_size=fc_input_size,
            hidden_size=fc_input_size,
            name=f"{self.model_name}_output"
        )
        
        # Output layer
        self.fc = nn.Linear(fc_input_size, self.output_size)
        self.softmax = nn.Softmax(dim=1)
        
        # Meta-learning components
        self.meta_parameters = nn.ParameterDict({
            'connection_growth_rate': nn.Parameter(torch.tensor(
                get_config("model.self_improvement.connection_growth_rate", 0.01)
            )),
            'pruning_threshold': nn.Parameter(torch.tensor(
                get_config("model.self_improvement.pruning_threshold", 0.001)
            )),
        })
        
        # Performance history for meta-learning
        self.performance_history = []
        self.meta_optimizer = None
        
        # Additional tracking buffers
        self.register_buffer("last_validation_loss", torch.tensor(float('inf')))
        self.register_buffer("best_validation_loss", torch.tensor(float('inf')))
        self.register_buffer("improvement_counter", torch.tensor(0))
        
        # Create initial checkpoint if recovery is enabled
        if is_safety_enabled("recovery"):
            self._create_initial_checkpoint()
    
    def _create_initial_checkpoint(self) -> None:
        """Create initial model checkpoint."""
        try:
            metadata = {
                'model_config': {
                    'input_size': self.input_size,
                    'hidden_size': self.hidden_size,
                    'output_size': self.output_size,
                    'num_layers': self.num_layers,
                    'bidirectional': self.bidirectional
                },
                'meta_parameters': {
                    'connection_growth_rate': self.meta_parameters['connection_growth_rate'].item(),
                    'pruning_threshold': self.meta_parameters['pruning_threshold'].item()
                }
            }
            
            # Save checkpoint
            checkpoint_path = get_system_path("backups")
            filename = os.path.join(checkpoint_path, f"{self.model_name}_initial.pt")
            
            torch.save({
                'model_state_dict': self.state_dict(),
                'metadata': metadata,
                'timestamp': datetime.datetime.now().isoformat()
            }, filename)
            
            self.logger.info(f"Initial checkpoint created at {filename}")
            
        except Exception as e:
            self.logger.warning(f"Failed to create initial checkpoint: {e}")
    
    @safe_training()
    def self_improve(self, validation_loss: float) -> Dict[str, Any]:
        """
        Update meta-parameters based on validation performance.
        
        Args:
            validation_loss: Current validation loss
            
        Returns:
            Information about the improvement process
        """
        # Convert to tensor for consistency
        validation_loss_tensor = torch.tensor(validation_loss, device=self.last_validation_loss.device)
        
        # Update tracking buffers
        self.last_validation_loss.copy_(validation_loss_tensor)
        if validation_loss < self.best_validation_loss:
            self.best_validation_loss.copy_(validation_loss_tensor)
            
            # Save checkpoint for best model if recovery is enabled
            if is_safety_enabled("recovery") and get_config("safety.auto_checkpoint", False):
                self._save_checkpoint("best")
        
        # Record performance for meta-learning
        timestamp = time.time()
        performance_record = {
            'loss': validation_loss,
            'timestamp': timestamp,
            'connection_growth_rate': self.meta_parameters['connection_growth_rate'].item(),
            'pruning_threshold': self.meta_parameters['pruning_threshold'].item(),
            'input_connectivity': self.input_adapter.connectivity_mask.mean().item(),
            'output_connectivity': self.output_adapter.connectivity_mask.mean().item(),
        }
        self.performance_history.append(performance_record)
        
        # Log performance
        self.logger.info(f"Performance record: loss={validation_loss:.6f}, "
                       f"input_connectivity={performance_record['input_connectivity']:.4f}, "
                       f"output_connectivity={performance_record['output_connectivity']:.4f}")
        
        # Need at least 5 data points before starting meta-optimization
        min_iterations = get_config("model.self_improvement.min_improvement_iterations", 5)
        if len(self.performance_history) < min_iterations:
            self.logger.info("Not enough performance history for meta-optimization")
            return {'status': 'insufficient_history'}
        
        # Initialize meta-optimizer if not already done
        if self.meta_optimizer is None:
            self.meta_optimizer = optim.Adam(self.meta_parameters.values(), lr=self.meta_learning_rate)
            self.logger.info("Meta-optimizer initialized")
        
        # Compute meta-gradient (simple heuristic based on recent performance trends)
        recent_losses = [record['loss'] for record in self.performance_history[-5:]]
        loss_trend = recent_losses[-1] - recent_losses[0]
        
        # Analyze trend and decide on strategy
        with torch.no_grad():
            if loss_trend < 0:  # Improving
                # Gradual reinforcement of current strategy
                meta_gradient_growth = -0.01  # Slightly reduce growth rate
                meta_gradient_threshold = -0.001  # Slightly reduce pruning
                status = 'reinforcing'
                self.improvement_counter += 1
            else:  # Declining or stagnant
                # Explore parameter space
                meta_gradient_growth = 0.05 * (torch.rand(1).item() - 0.5)  # -0.025 to 0.025
                meta_gradient_threshold = 0.01 * (torch.rand(1).item() - 0.5)  # -0.005 to 0.005
                status = 'exploring'
                # Reset counter if we're not improving
                self.improvement_counter.zero_()
            
            # Apply meta-gradients manually
            # Update growth rate (ensure it stays positive and within bounds)
            min_growth = get_config("model.self_improvement.min_growth_rate", 0.001)
            max_growth = get_config("model.self_improvement.max_growth_rate", 0.1)
            new_growth_rate = max(min_growth, min(max_growth, 
                self.meta_parameters['connection_growth_rate'] + meta_gradient_growth))
            self.meta_parameters['connection_growth_rate'].copy_(torch.tensor(new_growth_rate))
            
            # Update pruning threshold (ensure it stays positive and within bounds)
            min_threshold = get_config("model.self_improvement.min_threshold", 0.0001)
            max_threshold = get_config("model.self_improvement.max_threshold", 0.01)
            new_threshold = max(min_threshold, min(max_threshold, 
                self.meta_parameters['pruning_threshold'] + meta_gradient_threshold))
            self.meta_parameters['pruning_threshold'].copy_(torch.tensor(new_threshold))
        
        # Update adaptive blocks with new meta-parameters
        self.input_adapter.connection_growth_rate = self.meta_parameters['connection_growth_rate'].item()
        self.input_adapter.pruning_threshold = self.meta_parameters['pruning_threshold'].item()
        self.output_adapter.connection_growth_rate = self.meta_parameters['connection_growth_rate'].item()
        self.output_adapter.pruning_threshold = self.meta_parameters['pruning_threshold'].item()
        
        # Log meta-parameter updates
        self.logger.info(f"Updated meta-parameters: "
                       f"growth_rate={new_growth_rate:.6f}, threshold={new_threshold:.6f}")
        
        # Perform redundancy optimization periodically
        redundancy_frequency = get_config("model.self_improvement.redundancy_check_frequency", 10)
        if is_safety_enabled("redundancy_checking") and len(self.performance_history) % redundancy_frequency == 0:
            try:
                self.logger.info("Performing redundancy optimization")
                input_opt_stats = self.input_adapter.optimize_redundancy()
                output_opt_stats = self.output_adapter.optimize_redundancy()
                optimization_performed = True
            except Exception as e:
                self.logger.warning(f"Redundancy optimization failed: {e}")
                optimization_performed = False
        else:
            optimization_performed = False
        
        # Return improvement information
        return {
            'status': status,
            'loss_trend': loss_trend,
            'meta_updates': {
                'growth_rate': new_growth_rate,
                'threshold': new_threshold
            },
            'optimization_performed': optimization_performed,
            'improvement_counter': self.improvement_counter.item()
        }
    
    def _save_checkpoint(self, checkpoint_type: str = "periodic") -> str:
        """
        Save a model checkpoint.
        
        Args:
            checkpoint_type: Type of checkpoint (periodic, best, etc.)
            
        Returns:
            Path to the saved checkpoint
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_dir = get_system_path("backups")
        
        # Construct filename
        filename = os.path.join(checkpoint_dir, f"{self.model_name}_{checkpoint_type}_{timestamp}.pt")
        
        # Prepare checkpoint data
        metadata = {
            'checkpoint_type': checkpoint_type,
            'timestamp': timestamp,
            'validation_loss': self.last_validation_loss.item(),
            'best_validation_loss': self.best_validation_loss.item(),
            'meta_parameters': {
                'connection_growth_rate': self.meta_parameters['connection_growth_rate'].item(),
                'pruning_threshold': self.meta_parameters['pruning_threshold'].item()
            },
            'performance_history_length': len(self.performance_history)
        }
        
        # Save checkpoint
        torch.save({
            'model_state_dict': self.state_dict(),
            'metadata': metadata
        }, filename)
        
        self.logger.info(f"Checkpoint saved: {filename}")
        return filename
    
    @safe_tensor_op()
    def update_connections(self) -> Dict[str, Dict[str, Any]]:
        """
        Update connection patterns based on recent gradient flow.
        
        Returns:
            Connection update statistics
        """
        input_stats = self.input_adapter.update_connections(None)
        output_stats = self.output_adapter.update_connections(None)
        
        # Log update
        self.logger.debug(f"Connection update: input_density={input_stats['density']:.4f}, "
                        f"output_density={output_stats['density']:.4f}")
        
        return {
            'input_adapter': input_stats,
            'output_adapter': output_stats
        }
    
    @safe_tensor_op()
    def forward(
        self, 
        x: torch.Tensor, 
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the self-improving LSTM network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            hidden: Optional initial hidden state tuple (h_0, c_0)
        
        Returns:
            output: Predicted class probabilities (batch_size, output_size)
            hidden: Updated hidden state tuple (h_n, c_n)
        """
        # Validate input shape
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input, got {x.dim()}D")
            
        batch_size, seq_len, features = x.shape
        if features != self.input_size:
            raise ValueError(
                f"Expected input features: {self.input_size}, got: {features}"
            )
        
        # Input data validation if enabled
        if is_safety_enabled("data_validation"):
            for t in range(seq_len):
                safety_system.validate_input_data(x[:, t, :], f"input_t{t}")
        
        # Initialize hidden state if not provided
        if hidden is None:
            hidden = self.initialize_hidden(batch_size)
            
        # Validate hidden state
        h, c = hidden
        expected_h_shape = (self.num_layers * self.num_directions, batch_size, self.hidden_size)
        if h.shape != expected_h_shape:
            raise ValueError(
                f"Expected hidden state shape: {expected_h_shape}, got: {h.shape}"
            )
        
        # Process through adaptive input connections
        adapted_input = torch.zeros_like(x)
        for t in range(seq_len):
            adapted_input[:, t, :] = self.input_adapter(x[:, t, :])
            
        # Forward pass through LSTM
        try:
            lstm_out, hidden_next = self.lstm(adapted_input, hidden)
        except RuntimeError as e:
            # Handle specific LSTM errors
            if 'Expected hidden size' in str(e):
                self.logger.error(f"LSTM hidden size mismatch: {e}")
                raise ValueError(f"LSTM hidden size mismatch: {e}")
            elif 'batch sizes' in str(e):
                self.logger.error(f"LSTM batch size mismatch: {e}")
                raise ValueError(f"LSTM batch size mismatch: {e}")
            else:
                raise
        
        # Get the output of the last time step
        final_output = lstm_out[:, -1, :]
        
        # Process through adaptive output connections
        adapted_output = self.output_adapter(final_output)
        
        # Pass through fully connected layer and apply softmax
        logits = self.fc(adapted_output)
        output = self.softmax(logits)
        
        return output, hidden_next
    
    def initialize_hidden(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize hidden state and cell state for the LSTM.
        
        Args:
            batch_size: Batch size
            
        Returns:
            Tuple of (hidden_state, cell_state)
        """
        device = next(self.parameters()).device
        h0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size, device=device)
        return (h0, c0)
    
    def get_improvement_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the self-improvement process.
        
        Returns:
            Improvement statistics
        """
        if not self.performance_history:
            return {"status": "No improvement data available yet"}
        
        # Calculate key statistics
        latest = self.performance_history[-1]
        initial = self.performance_history[0]
        
        return {
            "total_iterations": len(self.performance_history),
            "initial_loss": initial['loss'],
            "current_loss": latest['loss'],
            "best_loss": self.best_validation_loss.item(),
            "improvement_percentage": (initial['loss'] - latest['loss']) / initial['loss'] * 100 
                if initial['loss'] != 0 else 0,
            "current_connectivity": {
                "input_layer": latest['input_connectivity'],
                "output_layer": latest['output_connectivity'],
            },
            "meta_parameters": {
                "connection_growth_rate": latest['connection_growth_rate'],
                "pruning_threshold": latest['pruning_threshold'],
            }
        }
    
    def save_model(self, path: Optional[str] = None) -> str:
        """
        Save the complete model with metadata.
        
        Args:
            path: Path to save the model (or None to use default)
            
        Returns:
            Path to the saved model
        """
        # Get default path if not provided
        if path is None:
            models_dir = get_system_path("models")
            path = os.path.join(models_dir, f"{self.model_name}.pt")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Prepare model data
        model_data = {
            'model_state_dict': self.state_dict(),
            'model_class': self.__class__.__name__,
            'model_config': {
                'input_size': self.input_size,
                'hidden_size': self.hidden_size,
                'output_size': self.output_size,
                'num_layers': self.num_layers,
                'bidirectional': self.bidirectional,
                'meta_learning_rate': self.meta_learning_rate,
            },
            'meta_parameters': {
                'connection_growth_rate': self.meta_parameters['connection_growth_rate'].item(),
                'pruning_threshold': self.meta_parameters['pruning_threshold'].item()
            },
            'performance_history': self.performance_history,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        # Save model
        torch.save(model_data, path)
        self.logger.info(f"Model saved to: {path}")
        return path
    
    @classmethod
    def load_model(cls, path: str, device: Optional[str] = None) -> 'SelfImprovingEEGLSTM':
        """
        Load model from file.
        
        Args:
            path: Path to the saved model
            device: Device to load the model to
            
        Returns:
            Loaded model instance
        """
        logger.info(f"Loading model from: {path}")
        
        try:
            # Load model data
            model_data = torch.load(path, map_location=device)
            
            # Verify model class
            if model_data.get('model_class', '') != cls.__name__:
                logger.warning(f"Model class mismatch: expected {cls.__name__}, "
                             f"got {model_data.get('model_class', 'unknown')}")
            
            # Create model instance
            config = model_data.get('model_config', {})
            
            # Extract required parameters with fallbacks
            input_size = config.get('input_size', get_config("model.input_size", 64))
            hidden_size = config.get('hidden_size', get_config("model.hidden_size", 128))
            output_size = config.get('output_size', get_config("model.output_size", 5))
            num_layers = config.get('num_layers', get_config("model.num_layers", 2))
            bidirectional = config.get('bidirectional', get_config("model.bidirectional", True))
            
            # Get model name from filename if available
            model_name = os.path.splitext(os.path.basename(path))[0]
            
            # Create model instance
            model = cls(
                input_size=input_size,
                hidden_size=hidden_size,
                output_size=output_size,
                num_layers=num_layers,
                bidirectional=bidirectional,
                model_name=model_name
            )
            
            # Load state dict
            model.load_state_dict(model_data['model_state_dict'])
            
            # Load performance history if available
            if 'performance_history' in model_data:
                model.performance_history = model_data['performance_history']
            
            # Load meta-parameters if available
            if 'meta_parameters' in model_data:
                meta_params = model_data['meta_parameters']
                if 'connection_growth_rate' in meta_params:
                    model.meta_parameters['connection_growth_rate'].data.copy_(
                        torch.tensor(meta_params['connection_growth_rate'])
                    )
                if 'pruning_threshold' in meta_params:
                    model.meta_parameters['pruning_threshold'].data.copy_(
                        torch.tensor(meta_params['pruning_threshold'])
                    )
            
            logger.info(f"Model loaded successfully from: {path}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Failed to load model: {e}")


# =====================================
# Training Functions
# =====================================

@safe_training()
def train_self_improving_model(
    model: SelfImprovingEEGLSTM,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    epochs: Optional[int] = None,
    learning_rate: Optional[float] = None,
    device: Optional[str] = None
) -> SelfImprovingEEGLSTM:
    """
    Train a self-improving EEG LSTM model.
    
    Args:
        model: SelfImprovingEEGLSTM model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        epochs: Number of training epochs (or None to use config)
        learning_rate: Learning rate for optimizer (or None to use config)
        device: Device to train on (or None to autodetect)
        
    Returns:
        Trained model
    """
    # Get parameters from config if not explicitly provided
    epochs = epochs or get_config("training.epochs", 100)
    learning_rate = learning_rate or get_config("training.learning_rate", 0.001)
    
    # Get other training parameters from config
    weight_decay = get_config("training.weight_decay", 0.0001)
    connection_update_freq = get_config("training.connection_update_frequency", 100)
    improvement_freq = get_config("training.improvement_frequency", 1)
    checkpoint_freq = get_config("training.checkpoint_frequency", 5)
    early_stopping_patience = get_config("training.patience", 15)
    
    # Determine device
    if device is None:
        device = "cuda" if torch.cuda.is_available() and get_config("system.cuda_enabled", True) else "cpu"
    
    logger.info(f"Starting training on {device} for {epochs} epochs with lr={learning_rate}")
    
    # Set up training
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Select optimizer based on config
    optimizer_name = get_config("training.optimizer", "adam").lower()
    if optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        momentum = get_config("training.momentum", 0.9)
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        logger.warning(f"Unknown optimizer: {optimizer_name}, using Adam")
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Set up learning rate scheduler if enabled
    scheduler = None
    scheduler_name = get_config("training.scheduler", "none").lower()
    if scheduler_name == "step":
        step_size = get_config("training.scheduler_step_size", 30)
        gamma = get_config("training.scheduler_gamma", 0.1)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_name == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_name == "plateau":
        patience = get_config("training.scheduler_patience", 10)
        factor = get_config("training.scheduler_factor", 0.5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience)
    
    # Set up early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Create stats tracking
    stats = {
        "train_losses": [],
        "val_losses": [],
        "train_accuracies": [],
        "val_accuracies": [],
        "learning_rates": [],
        "connection_densities": []
    }
    
    # Training loop
    try:
        for epoch in range(epochs):
            # Training phase
            model.train()
            running_loss = 0.0
            batch_count = 0
            correct = 0
            total = 0
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs, _ = model(inputs)
                loss = criterion(outputs, targets)
                
                # Backward pass and optimize
                loss.backward()
                
                # Apply gradient clipping if enabled
                if is_safety_enabled("gradient_safety") and get_config("safety.gradient_clipping", False):
                    clip_value = get_config("safety.gradient_clip_value", 1.0)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
                
                optimizer.step()
                
                # Update connection patterns periodically
                batch_count += 1
                if batch_count % connection_update_freq == 0:
                    update_stats = model.update_connections()
                    logger.debug(f"Connections updated at batch {batch_count}")
                
                # Track metrics
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
            
            train_loss = running_loss / len(train_loader)
            train_accuracy = 100 * correct / total if total > 0 else 0
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs, _ = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
            
            val_loss /= len(val_loader)
            val_accuracy = 100 * correct / total if total > 0 else 0
            
            # Update learning rate scheduler
            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
            
            # Track stats
            stats["train_losses"].append(train_loss)
            stats["val_losses"].append(val_loss)
            stats["train_accuracies"].append(train_accuracy)
            stats["val_accuracies"].append(val_accuracy)
            stats["learning_rates"].append(optimizer.param_groups[0]["lr"])
            stats["connection_densities"].append({
                "input": model.input_adapter.connectivity_mask.mean().item(),
                "output": model.output_adapter.connectivity_mask.mean().item()
            })
            
            # Print progress
            logger.info(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, '
                       f'Train Acc: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, '
                       f'Val Acc: {val_accuracy:.2f}%')
            
            # Self-improve meta-parameters
            if (epoch + 1) % improvement_freq == 0:
                if get_config("model.self_improvement.enabled", True):
                    improvement_info = model.self_improve(val_loss)
                    logger.info(f"Self-improvement status: {improvement_info['status']}")
                else:
                    logger.debug("Self-improvement disabled in config")
            
            # Save checkpoint periodically
            if is_safety_enabled("recovery") and (epoch + 1) % checkpoint_freq == 0:
                model._save_checkpoint(f"epoch_{epoch+1}")
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model
                if get_config("training.save_best_model", True):
                    model.save_model()
                    logger.info("New best model saved")
            else:
                patience_counter += 1
                logger.debug(f"Early stopping patience: {patience_counter}/{early_stopping_patience}")
                
                if get_config("training.early_stopping", True) and patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        # Save final model
        if get_config("training.save_final_model", True):
            model_path = get_system_path("models")
            final_path = os.path.join(model_path, f"{model.model_name}_final.pt")
            model.save_model(final_path)
        
        # Save training stats
        stats_path = get_system_path("stats")
        stats_file = os.path.join(stats_path, f"{model.model_name}_training_stats.json")
        
        with open(stats_file, 'w') as f:
            # Convert tensors to native types for JSON serialization
            serializable_stats = {}
            for key, value in stats.items():
                if isinstance(value, list) and value and isinstance(value[0], dict):
                    serializable_stats[key] = [{k: float(v) for k, v in item.items()} for item in value]
                else:
                    serializable_stats[key] = value
            
            json.dump(serializable_stats, f, indent=2)
            
        logger.info(f"Training stats saved to: {stats_file}")
        
        return model
        
    except Exception as e:
        logger.error(f"Training error: {e}")
        logger.error(traceback.format_exc())
        
        # Try to save checkpoint on error if recovery is enabled
        if is_safety_enabled("recovery"):
            try:
                error_checkpoint = model._save_checkpoint("error_recovery")
                logger.info(f"Error recovery checkpoint saved: {error_checkpoint}")
            except Exception as checkpoint_e:
                logger.error(f"Failed to save error recovery checkpoint: {checkpoint_e}")
        
        raise


@safe_tensor_op()
def recursive_prediction_with_improvement(
    eeg_data: torch.Tensor,
    model: SelfImprovingEEGLSTM,
    hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    improve_every_n_samples: Optional[int] = None,
    feedback_fn: Optional[Callable] = None,
    device: Optional[str] = None
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Predicts cognitive states recursively using a self-improving LSTM.
    
    Args:
        eeg_data: A tensor of shape (batch_size, seq_len, input_size)
        model: The SelfImprovingEEGLSTM model
        hidden_state: Optional initial hidden and cell states
        improve_every_n_samples: How often to trigger self-improvement (or None to use config)
        feedback_fn: Optional function to provide feedback signal for improvement
        device: Device to run prediction on (or None to autodetect)
        
    Returns:
        predictions: A tensor of shape (batch_size, output_size)
        hidden_state: The updated hidden state tuple
    """
    # Get parameters from config if not explicitly provided
    if improve_every_n_samples is None:
        improve_every_n_samples = get_config("prediction.improve_every_n_samples", 10)
    
    # Determine device
    if device is None:
        device = "cuda" if torch.cuda.is_available() and get_config("system.cuda_enabled", True) else "cpu"
    
    logger.info(f"Starting recursive prediction with improvement on {device}")
    
    # Move model and data to device
    model = model.to(device)
    eeg_data = eeg_data.to(device)
    
    batch_size, seq_len, input_size = eeg_data.shape
    
    # Validate input
    if input_size != model.input_size:
        raise ValueError(f"Input size mismatch: model expects {model.input_size}, got {input_size}")
    
    # Initialize hidden state if not provided
    if hidden_state is None:
        hidden_state = model.initialize_hidden(batch_size)
    else:
        # Move hidden state to device
        hidden_state = (hidden_state[0].to(device), hidden_state[1].to(device))
    
    # Set model to evaluation mode
    model.eval()
    
    # Validate input data if safety is enabled
    if is_safety_enabled("data_validation"):
        for t in range(seq_len):
            safety_system.validate_input_data(eeg_data[:, t, :], f"input_t{t}")
    
    try:
        # Process each time step sequentially
        current_hidden = hidden_state
        sample_count = 0
        improvement_cycles = 0
        
        for t in range(seq_len):
            # Extract current time step data
            current_input = eeg_data[:, t:t+1, :]
            
            # Forward pass for single time step
            with torch.no_grad():
                _, current_hidden = model(current_input, current_hidden)
            
            # Update sample count
            sample_count += 1
            
            # Self-improvement cycle if enabled
            if get_config("model.self_improvement.enabled", True) and feedback_fn is not None:
                if sample_count % improve_every_n_samples == 0:
                    # Get feedback signal
                    try:
                        feedback = feedback_fn(current_input, current_hidden)
                        improvement_info = model.self_improve(feedback)
                        improvement_cycles += 1
                        logger.info(f"Self-improvement cycle {improvement_cycles}: feedback={feedback:.6f}")
                    except Exception as e:
                        logger.warning(f"Self-improvement error: {e}")
        
        # Final prediction using the last hidden state
        final_input = eeg_data[:, -1:, :]
        with torch.no_grad():
            predictions, final_hidden = model(final_input, current_hidden)
        
        logger.info(f"Recursive prediction completed with {improvement_cycles} improvement cycles")
        return predictions, final_hidden
        
    except Exception as e:
        logger.error(f"Recursive prediction error: {e}")
        logger.error(traceback.format_exc())
        raise


@safe_tensor_op()
def batch_process_eeg(
    eeg_batch: torch.Tensor,
    model: SelfImprovingEEGLSTM,
    batch_size: Optional[int] = None,
    device: Optional[str] = None
) -> torch.Tensor:
    """
    Process a large batch of EEG data with memory-efficient batching.
    
    Args:
        eeg_batch: EEG data tensor of shape (total_samples, seq_len, input_size)
        model: Trained EEGStateLSTM model
        batch_size: Size of mini-batches to process at once (or None to use config)
        device: Device to run processing on (or None to autodetect)
        
    Returns:
        Tensor of predictions for all samples (total_samples, output_size)
    """
    # Get parameters from config if not explicitly provided
    if batch_size is None:
        batch_size = get_config("prediction.batch_size", 32)
    
    # Check memory safety if enabled
    if is_safety_enabled("memory_safety"):
        max_batch_size = get_config("safety.max_batch_size", 1024)
        if batch_size > max_batch_size:
            logger.warning(f"Batch size {batch_size} exceeds safety limit {max_batch_size}, reducing")
            batch_size = max_batch_size
    
    # Determine device
    if device is None:
        device = "cuda" if torch.cuda.is_available() and get_config("system.cuda_enabled", True) else "cpu"
    
    logger.info(f"Starting batch processing of {eeg_batch.shape[0]} samples on {device}")
    
    # Move model to device
    model = model.to(device)
    model.eval()  # Set model to evaluation mode
    
    total_samples = eeg_batch.shape[0]
    
    # Prepare output tensor on CPU (to avoid GPU memory issues)
    all_predictions = torch.zeros(total_samples, model.output_size)
    
    # Track processing time
    start_time = time.time()
    error_count = 0
    
    # Process in mini-batches
    with torch.no_grad():  # Disable gradient calculation
        for i in range(0, total_samples, batch_size):
            # Get current mini-batch indices
            end_idx = min(i + batch_size, total_samples)
            
            try:
                # Move mini-batch to device
                mini_batch = eeg_batch[i:end_idx].to(device)
                
                # Initialize hidden state for this mini-batch
                hidden = model.initialize_hidden(mini_batch.shape[0])
                
                # Get predictions
                batch_predictions, _ = model(mini_batch, hidden)
                
                # Store predictions (back to CPU)
                all_predictions[i:end_idx] = batch_predictions.cpu()
                
                # Log progress periodically
                if (i // batch_size) % 10 == 0 or end_idx == total_samples:
                    elapsed = time.time() - start_time
                    progress = end_idx / total_samples * 100
                    logger.info(f"Progress: {progress:.1f}% ({end_idx}/{total_samples}), "
                               f"Time elapsed: {elapsed:.1f}s")
                    
            except Exception as e:
                logger.error(f"Error processing batch {i}-{end_idx}: {e}")
                logger.error(traceback.format_exc())
                error_count += 1
                
                # Skip this batch and continue with the next one
                continue
                
            # Update connections periodically if self-improvement is enabled
            connection_update_freq = get_config("prediction.connection_update_frequency", 50)
            if get_config("model.self_improvement.enabled", True) and (i // batch_size) % connection_update_freq == 0 and i > 0:
                try:
                    model.update_connections()
                    logger.debug(f"Connections updated at batch {i//batch_size}")
                except Exception as e:
                    logger.warning(f"Connection update error: {e}")
            
            # Force garbage collection if memory safety is enabled
            if is_safety_enabled("memory_safety") and get_config("safety.force_garbage_collection", False):
                if (i // batch_size) % 5 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
    
    # Final processing time
    total_time = time.time() - start_time
    logger.info(f"Batch processing completed in {total_time:.1f}s with {error_count} errors")
    
    if error_count > 0:
        logger.warning(f"Completed with {error_count} batch errors")
    
    return all_predictions


# =====================================
# Utility Functions
# =====================================

def create_example_model(config_override: Optional[Dict[str, Any]] = None) -> SelfImprovingEEGLSTM:
    """
    Create an example model based on config.
    
    Args:
        config_override: Optional dictionary to override config parameters
        
    Returns:
        Initialized model
    """
    # Apply temporary config overrides if provided
    original_values = {}
    if config_override:
        for key, value in config_override.items():
            original_values[key] = get_config(key)
            update_config(key, value)
    
    # Create model using config values
    model = SelfImprovingEEGLSTM()
    
    # Log model creation
    logger.info(f"Created example model with input_size={model.input_size}, "
               f"hidden_size={model.hidden_size}, output_size={model.output_size}")
    
    # Restore original config values
    if config_override:
        for key, value in original_values.items():
            update_config(key, value)
    
    return model


def generate_synthetic_data(
    num_samples: Optional[int] = None,
    seq_len: Optional[int] = None,
    input_size: Optional[int] = None,
    output_size: Optional[int] = None,
    device: Optional[str] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate synthetic EEG data for testing.
    
    Args:
        num_samples: Number of samples to generate (or None to use config)
        seq_len: Sequence length (or None to use config)
        input_size: Number of input features (or None to use config)
        output_size: Number of output classes (or None to use config)
        device: Device to create data on (or None to use CPU)
        
    Returns:
        Tuple of (inputs, targets)
    """
    # Get parameters from config if not explicitly provided
    num_samples = num_samples or get_config("data.synthetic.num_samples", 100)
    seq_len = seq_len or get_config("data.synthetic.seq_len", 50)
    input_size = input_size or get_config("model.input_size", 64)
    output_size = output_size or get_config("model.output_size", 5)
    
    # Use CPU for data generation by default
    if device is None:
        device = "cpu"
    
    logger.info(f"Generating synthetic data: {num_samples} samples, {seq_len} time steps, {input_size} features")
    
    # Generate random input data
    inputs = torch.randn(num_samples, seq_len, input_size, device=device)
    
    # Generate random target classes
    targets = torch.randint(0, output_size, (num_samples,), device=device)
    
    return inputs, targets


# =====================================
# Main Example Usage
# =====================================

def main_example():
    """Example usage of the self-improving EEG LSTM model."""
    logger.info("Starting main example")
    
    # Select device
    device = "cuda" if torch.cuda.is_available() and get_config("system.cuda_enabled", True) else "cpu"
    logger.info(f"Using device: {device}")
    
    try:
        # Create model
        model = create_example_model()
        logger.info(f"Model created: {model.model_name}")
        
        # Generate synthetic data
        inputs, targets = generate_synthetic_data(
            num_samples=500,
            seq_len=50
        )
        
        # Create DataLoader
        from torch.utils.data import TensorDataset, DataLoader
        
        # Get batch size from config
        batch_size = get_config("training.batch_size", 32)
        
        # Split data into train, validation, and test sets
        train_ratio = 1.0 - get_config("training.validation_ratio", 0.15) - get_config("training.test_ratio", 0.15)
        val_ratio = get_config("training.validation_ratio", 0.15)
        
        train_size = int(train_ratio * len(inputs))
        val_size = int(val_ratio * len(inputs))
        
        # Shuffle data if enabled
        if get_config("training.shuffle", True):
            indices = torch.randperm(len(inputs))
            inputs = inputs[indices]
            targets = targets[indices]
        
        train_inputs = inputs[:train_size]
        train_targets = targets[:train_size]
        
        val_inputs = inputs[train_size:train_size+val_size]
        val_targets = targets[train_size:train_size+val_size]
        
        test_inputs = inputs[train_size+val_size:]
        test_targets = targets[train_size+val_size:]
        
        # Create datasets
        train_dataset = TensorDataset(train_inputs, train_targets)
        val_dataset = TensorDataset(val_inputs, val_targets)
        test_dataset = TensorDataset(test_inputs, test_targets)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=get_config("training.shuffle", True))
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        logger.info(f"Data prepared: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")
        
        # Train model
        trained_model = train_self_improving_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device
        )
        
        # Evaluate model
        test_loss = 0.0
        correct = 0
        total = 0
        
        trained_model.eval()
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs, _ = trained_model(inputs)
                loss = nn.CrossEntropyLoss()(outputs, targets)
                test_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        test_loss /= len(test_loader)
        accuracy = 100 * correct / total
        
        logger.info(f"Test results: loss={test_loss:.4f}, accuracy={accuracy:.2f}%")
        
        # Get improvement stats
        improvement_stats = trained_model.get_improvement_stats()
        logger.info(f"Improvement stats: {improvement_stats}")
        
        # Test batch processing
        batch_predictions = batch_process_eeg(test_inputs, trained_model, device=device)
        logger.info(f"Batch processing completed, output shape: {batch_predictions.shape}")
        
        # Save results
        results = {
            "model_name": trained_model.model_name,
            "test_loss": test_loss,
            "test_accuracy": accuracy,
            "improvement_stats": improvement_stats,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        results_path = get_system_path("stats")
        results_file = os.path.join(results_path, f"{trained_model.model_name}_results.json")
        
        with open(results_file, 'w') as f:
            # Convert nested values to native types for JSON serialization
            serializable_results = json.loads(json.dumps(results, default=lambda o: 
                o.item() if hasattr(o, 'item') else str(o)))
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved to: {results_file}")
        logger.info("Main example completed successfully")
        
    except Exception as e:
        logger.error(f"Main example error: {e}")
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    # Initialize config manager
    if not config_manager._initialized:
        # Load from default location
        config_path = os.path.join(get_system_path("config"), "system_config.yaml")
        config_manager.__init__(config_path)
    
    # Run example
    main_example()
