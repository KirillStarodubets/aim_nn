import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class AmuletDataset(Dataset):
    """Dataset for loading data from merged HDF5 file"""
    
    def __init__(self, hdf5_file: Path, train: bool = True, 
                 test_size: float = 0.1, random_state: int = 42):
        super().__init__()
        self.hdf5_file = hdf5_file
        self.train = train
        
        # Load and preprocess data
        self.X, self.y, self.group_info = self._load_and_preprocess_data()
        
        # Split into train/test
        X_train, X_test, y_train, y_test, info_train, info_test = train_test_split(
            self.X, self.y, self.group_info, test_size=test_size, random_state=random_state
        )
        
        if train:
            self.X_data = X_train
            self.y_data = y_train
            self.group_info_data = info_train
        else:
            self.X_data = X_test
            self.y_data = y_test
            self.group_info_data = info_test
            
        print(f"Loaded {len(self.X_data)} samples ({'train' if train else 'test'})")
    
    def _load_and_preprocess_data(self) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """Load and preprocess data from HDF5 file"""
        X_list = []
        y_list = []
        group_info_list = []
        
        try:
            with h5py.File(self.hdf5_file, 'r') as hdf:
                if 'merged_data' not in hdf:
                    raise ValueError("No data in merged_data group")
                
                merged_group = hdf['merged_data']
                
                for param_name in merged_group.keys():
                    param_group = merged_group[param_name]
                    
                    # Extract parameters from group name
                    parts = param_name.split('_')
                    if len(parts) < 2:
                        continue
                    
                    try:
                        beta = float(parts[0].replace('beta', ''))
                        ntot = float(parts[1].replace('ntot', ''))
                    except:
                        continue
                    
                    for iter_name in param_group.keys():
                        if not iter_name.startswith('iter'):
                            continue
                        
                        iter_group = param_group[iter_name]
                        
                        # Extract scalar parameters
                        scalar_params = {}
                        for key in ['U', 'J', 'degeneracy']:
                            if key in iter_group:
                                scalar_params[key] = iter_group[key][()]
                        
                        # Extract legendre cutoffs
                        gl_in_cutoff = 0
                        gl_out_cutoff = 0
                        if 'legendre_cutoffs_GL_in_positions' in iter_group:
                            gl_in_cutoff = iter_group['legendre_cutoffs_GL_in_positions'][()]
                        if 'legendre_cutoffs_GL_out_positions' in iter_group:
                            gl_out_cutoff = iter_group['legendre_cutoffs_GL_out_positions'][()]
                        
                        # Extract Gl_in data
                        gl_in_data = None
                        for ds_name in iter_group.keys():
                            if ds_name.startswith('G0l_in_column_'):
                                gl_in_data = iter_group[ds_name][()]
                                break
                        
                        if gl_in_data is None:
                            continue
                        
                        # Extract target variables
                        targets = {}
                        for key in ['magnetic_moment', 'coulomb_energy']:
                            if key in iter_group:
                                targets[key] = iter_group[key][()]
                        
                        # Find Gl_out data
                        gl_out_data = None
                        for ds_name in iter_group.keys():
                            if ds_name.startswith('Gl_out_column_'):
                                gl_out_data = iter_group[ds_name][()]
                                break
                        
                        if gl_out_data is None:
                            continue
                        
                        # Prepare input data
                        x_scalars = np.array([
                            scalar_params.get('U', 0.0),
                            scalar_params.get('J', 0.0),
                            beta,
                            scalar_params.get('degeneracy', 0.0),
                            ntot
                        ])
                        
                        # Process Gl_in (zero out data after cutoff)
                        gl_in_processed = self._process_gl_data_with_cutoff(gl_in_data, gl_in_cutoff)
                        
                        # Combine scalar parameters and Gl_in
                        x_combined = np.concatenate([x_scalars, gl_in_processed])
                        
                        # Prepare target data
                        y_scalars = np.array([
                            targets.get('magnetic_moment', 0.0),
                            targets.get('coulomb_energy', 0.0)
                        ])
                        
                        # Process Gl_out (zero out data after cutoff)
                        gl_out_processed = self._process_gl_data_with_cutoff(gl_out_data, gl_out_cutoff)
                        
                        # Combine target variables
                        y_combined = np.concatenate([y_scalars, gl_out_processed])
                        
                        X_list.append(x_combined)
                        y_list.append(y_combined)
                        
                        # Save group information for analysis
                        group_info = {
                            'group_name': f"{param_name}_{iter_name}",
                            'beta': beta,
                            'ntot': ntot,
                            'U': scalar_params.get('U', 0.0),
                            'J': scalar_params.get('J', 0.0),
                            'degeneracy': scalar_params.get('degeneracy', 0.0),
                            'gl_in_cutoff': gl_in_cutoff,
                            'gl_out_cutoff': gl_out_cutoff,
                            'magnetic_moment': targets.get('magnetic_moment', 0.0),
                            'coulomb_energy': targets.get('coulomb_energy', 0.0),
                            'iteration': iter_name
                        }
                        group_info_list.append(group_info)
        
        except Exception as e:
            print(f"Error loading data: {e}")
            return np.array([]), np.array([]), []
        
        print(f"Total loaded {len(X_list)} samples from {len(set([info['group_name'].split('_')[0] for info in group_info_list]))} unique parameter groups")
        return np.array(X_list), np.array(y_list), group_info_list
    
    def _process_gl_data_with_cutoff(self, gl_data: np.ndarray, cutoff: int) -> np.ndarray:
        """Process GL data (zero out data after cutoff)"""
        if cutoff > 0 and cutoff < len(gl_data):
            # Zero out data after cutoff
            processed = gl_data.copy()
            processed[cutoff:] = 0.0
            return processed
        else:
            return gl_data
    
    def __len__(self):
        return len(self.X_data)
    
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.X_data[idx])
        y = torch.FloatTensor(self.y_data[idx])
        return x, y

class AmuletMLP(nn.Module):
    """Multi-layer perceptron for predicting physical quantities"""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.Sigmoid(),
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class AmuletTrainer:
    """Class for training and evaluating the model"""
    
    def __init__(self, model: nn.Module, device: torch.device = None):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Training history
        self.train_losses = []
    
    def train(self, train_loader: DataLoader, num_epochs: int, learning_rate: float):
        """Train the model"""
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        for epoch in range(num_epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * batch_X.size(0)
            
            train_loss /= len(train_loader.dataset)
            self.train_losses.append(train_loss)
            
            if epoch % (num_epochs/10) == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}')
        
        print(f'Training completed. Final error: {train_loss:.6f}')
    
    def evaluate(self, data_loader: DataLoader, criterion: nn.Module = None) -> float:
        """Evaluate model on data"""
        if criterion is None:
            criterion = nn.MSELoss()
        
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch_X, batch_y in data_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                total_loss += loss.item() * batch_X.size(0)
        
        return total_loss / len(data_loader.dataset)
    
    def predict(self, data_loader: DataLoader, dataset: AmuletDataset) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """Make predictions on data and return group information"""
        self.model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in data_loader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(batch_y.numpy())
        
        return (np.concatenate(all_preds), 
                np.concatenate(all_targets), 
                dataset.group_info_data)
    
    def plot_training_history(self):
        """Plot training history"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Train Loss', alpha=0.8)
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.tight_layout()
        plt.show()
        
        
class AmuletVisualizer:
    """Class for visualizing results"""
    
    @staticmethod
    def plot_gl_comparison_by_group(predictions: np.ndarray, targets: np.ndarray, 
                                  group_info: List[Dict], gl_start_idx: int):
        """Compare predicted and actual GL data for all groups separately"""
        
        # GL data starts from index 2
        gl_preds = predictions[:, gl_start_idx:]
        gl_targets = targets[:, gl_start_idx:]
        
        # Group data by parameters (beta, ntot, U, J)
        grouped_data = {}
        for i, info in enumerate(group_info):
            key = (info['beta'], info['ntot'], info['U'], info['J'])
            if key not in grouped_data:
                grouped_data[key] = []
            grouped_data[key].append((i, info))
        
        # Create plots for each group
        num_groups = len(grouped_data)
        rows = int(np.ceil(np.sqrt(num_groups)))
        cols = int(np.ceil(num_groups / rows))
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*4))
        axes = axes.flatten() if num_groups > 1 else [axes]
        
        for idx, ((beta, ntot, U, J), samples) in enumerate(grouped_data.items()):
            ax = axes[idx]
            
            # Take first sample from group for visualization
            sample_idx, info = samples[0]
            
            pred_gl = gl_preds[sample_idx]
            target_gl = gl_targets[sample_idx]
            
            # Find non-zero values
            cutoff = info['gl_out_cutoff']
            valid_indices = np.arange(len(target_gl))
            if cutoff > 0:
                valid_indices = valid_indices[:cutoff]
            
            ax.plot(valid_indices, pred_gl[valid_indices], 'b-', label='Predicted', alpha=0.8, linewidth=2)
            ax.plot(valid_indices, target_gl[valid_indices], 'r--', label='Actual', alpha=0.8, linewidth=2)
            
            ax.set_xlabel('Index l')
            ax.set_ylabel('G(l)')
            ax.set_title(f'β={beta}, ntot={ntot:.2f}, U={U}, J={J}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(grouped_data), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def print_detailed_scalar_comparison(predictions: np.ndarray, targets: np.ndarray, 
                                       group_info: List[Dict]):
        """Print detailed comparison of predicted and actual scalar quantities"""
        
        mag_moment_pred = predictions[:, 0]
        mag_moment_target = targets[:, 0]
        coulomb_energy_pred = predictions[:, 1]
        coulomb_energy_target = targets[:, 1]
        
        # Calculate overall metrics
        mag_moment_mse = np.mean((mag_moment_pred - mag_moment_target) ** 2)
        mag_moment_mae = np.mean(np.abs(mag_moment_pred - mag_moment_target))
        mag_moment_rmse = np.sqrt(mag_moment_mse)
        
        coulomb_energy_mse = np.mean((coulomb_energy_pred - coulomb_energy_target) ** 2)
        coulomb_energy_mae = np.mean(np.abs(coulomb_energy_pred - coulomb_energy_target))
        coulomb_energy_rmse = np.sqrt(coulomb_energy_mse)
        
        print("\n" + "="*120)
        print("DETAILED COMPARISON OF SCALAR QUANTITIES FOR ALL GROUPS")
        print("="*120)
        
        print(f"\nOVERALL METRICS:")
        print(f"{'Metric':<15} {'Magnetic Moment':<20} {'Coulomb Energy':<20}")
        print(f"{'-'*15:<15} {'-'*20:<20} {'-'*20:<20}")
        print(f"{'MSE':<15} {mag_moment_mse:<20.6f} {coulomb_energy_mse:<20.6f}")
        print(f"{'MAE':<15} {mag_moment_mae:<20.6f} {coulomb_energy_mae:<20.6f}")
        print(f"{'RMSE':<15} {mag_moment_rmse:<20.6f} {coulomb_energy_rmse:<20.6f}")
        
        # Group by parameters
        grouped_scalars = {}
        for i, info in enumerate(group_info):
            key = (info['beta'], info['ntot'], info['U'], info['J'])
            if key not in grouped_scalars:
                grouped_scalars[key] = {
                    'mag_pred': [], 'mag_true': [], 
                    'energy_pred': [], 'energy_true': [],
                    'samples': []
                }
            
            grouped_scalars[key]['mag_pred'].append(mag_moment_pred[i])
            grouped_scalars[key]['mag_true'].append(mag_moment_target[i])
            grouped_scalars[key]['energy_pred'].append(coulomb_energy_pred[i])
            grouped_scalars[key]['energy_true'].append(coulomb_energy_target[i])
            grouped_scalars[key]['samples'].append(info['group_name'])
        
        print(f"\nDETAILED COMPARISON BY PARAMETER GROUPS:")
        print("-" * 120)
        print(f"{'Parameters':<25} {'Magnetic Moment':<25} {'Coulomb Energy':<25} {'Samples':<10}")
        print(f"{'':<25} {'Pred ± Err':<25} {'Pred ± Err':<25} {'':<10}")
        print("-" * 120)
        
        for (beta, ntot, U, J), data in grouped_scalars.items():
            # Magnetic moment statistics
            mag_pred_mean = np.mean(data['mag_pred'])
            mag_true_mean = np.mean(data['mag_true'])
            mag_error = abs(mag_pred_mean - mag_true_mean)
            mag_std = np.std(data['mag_pred'])
            
            # Coulomb energy statistics
            energy_pred_mean = np.mean(data['energy_pred'])
            energy_true_mean = np.mean(data['energy_true'])
            energy_error = abs(energy_pred_mean - energy_true_mean)
            energy_std = np.std(data['energy_pred'])
            
            params_str = f"β={beta}\nntot={ntot:.2f}\nU={U}, J={J}"
            
            mag_str = f"{mag_pred_mean:.4f} ± {mag_error:.4f}\n({mag_true_mean:.4f} true)"
            energy_str = f"{energy_pred_mean:.4f} ± {energy_error:.4f}\n({energy_true_mean:.4f} true)"
            
            samples_count = len(data['samples'])
            
            print(f"{params_str:<25} {mag_str:<25} {energy_str:<25} {samples_count:<10}")
        
        # Print individual errors for first few examples
        print(f"\nINDIVIDUAL ERRORS (first 10 examples):")
        print("-" * 100)
        print(f"{'Group':<20} {'Mag Moment':<15} {'Error':<10} {'Coul Energy':<15} {'Error':<10}")
        print("-" * 100)
        
        for i in range(min(10, len(predictions))):
            info = group_info[i]
            group_name = info['group_name'][:18] + '...' if len(info['group_name']) > 18 else info['group_name']
            
            mag_error = abs(mag_moment_pred[i] - mag_moment_target[i])
            energy_error = abs(coulomb_energy_pred[i] - coulomb_energy_target[i])
            
            print(f"{group_name:<20} {mag_moment_pred[i]:<15.4f} {mag_error:<10.4f} "
                  f"{coulomb_energy_pred[i]:<15.4f} {energy_error:<10.4f}")
            
            
def main():
    """Main execution function"""
    
    # Configuration
    CONFIG = {
        'hdf5_file': "merged_amulet_data.h5",
        'batch_size': 2**6,
        'hidden_dims': [2**10],
        'learning_rate': 0.0001,
        'num_epochs': 2000  
    }
    
    # Check if file exists
    hdf5_path = Path(CONFIG['hdf5_file'])
    if not hdf5_path.exists():
        print(f"Error: file {CONFIG['hdf5_file']} not found")
        return
    
    
    # Check file structure
    print("Checking HDF5 file structure...")
    try:
        with h5py.File(hdf5_path, 'r') as hdf:
            if 'merged_data' in hdf:
                merged_group = hdf['merged_data']
                print(f"Found parameter groups: {len(merged_group.keys())}")
                for param_name in merged_group.keys():
                    param_group = merged_group[param_name]
                    print(f"  {param_name}: {len(param_group.keys())} iterations")
                    for iter_name in param_group.keys():
                        print(f"    {iter_name}")
    except Exception as e:
        print(f"Error checking file: {e}")
        
        
    # 1. Load data
    print("Loading data...")
    train_dataset = AmuletDataset(hdf5_path, train=True, test_size=0.2)
    test_dataset = AmuletDataset(hdf5_path, train=False, test_size=0.2)
    
    if len(train_dataset) == 0 or len(test_dataset) == 0:
        print("Failed to load data. Check HDF5 file structure.")
        return
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    
    # Data dimensions
    input_dim = train_dataset[0][0].shape[0]
    output_dim = train_dataset[0][1].shape[0]
    
    print(f"Input dimension: {input_dim}")
    print(f"Output dimension: {output_dim}")
    print(f"Training set size: {len(train_dataset)}")
    print(f"Test set size: {len(test_dataset)}")
    
    # 2. Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = AmuletMLP(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=CONFIG['hidden_dims']
    )
    
    print(f"Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # 3. Training
    trainer = AmuletTrainer(model, device)
    print("Starting training...")
    
    trainer.train(
        train_loader=train_loader,
        num_epochs=CONFIG['num_epochs'],
        learning_rate=CONFIG['learning_rate']
    )
    
    # 4. Evaluation on test set
    print("\nEvaluating on test set...")
    test_loss = trainer.evaluate(test_loader)
    print(f"Test MSE: {test_loss:.6f}")
    
    # 5. Predictions and visualization
    print("\nGenerating predictions...")
    predictions, targets, group_info = trainer.predict(test_loader, test_dataset)
    
    # Training history visualization
    trainer.plot_training_history()
    
    # GL_out visualization by groups
    print("Visualizing GL_out data by parameter groups...")
    AmuletVisualizer.plot_gl_comparison_by_group(
        predictions, targets, group_info, gl_start_idx=2
    )
    
    # Detailed scalar quantities comparison
    AmuletVisualizer.print_detailed_scalar_comparison(predictions, targets, group_info)
    
    # 6. Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': CONFIG,
        'input_dim': input_dim,
        'output_dim': output_dim
    }, 'amulet_mlp_model.pth')
    
    print("Model saved to 'amulet_mlp_model.pth'")

if __name__ == "__main__":
    main()