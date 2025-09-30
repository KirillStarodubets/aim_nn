import re
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Union, List, Optional, Any
from dataclasses import dataclass
import copy

@dataclass
class FileInfo:
    """Container for file information and data"""
    path: Union[str, Path]
    name: str
    description: str = ""
    columns: Optional[Dict[int, np.ndarray]] = None

class AmuletDataParser:
    """Class for parsing AMULET data files"""
    
    # Regular expressions for pattern matching
    PATTERNS = {
        'version': re.compile(r"You are using version\s+(AMULET-[^\s]+)"),
        'beta': re.compile(r"Inverse temperature \[beta\]\s+([0-9.]+)"),
        'ntotal': re.compile(r"Total number of electrons \[ntotal\]\s+([0-9.]+)"),
        'impurity_name': re.compile(r"Impurity name \[name\]\s+(\w+)"),
        'degeneracy': re.compile(r"Degeneracy \[nlm\]\s+(\d+)"),
        'U': re.compile(r"Screened Coulomb interaction \[U\]\s+([0-9.E+-]+)"),
        'J': re.compile(r"Hund exchange coupling\s+\[J\]\s+([0-9.E+-]+)"),
        'legendre_cutoff': re.compile(r"Optimal Legendre cutoff[^\d]*([0-9]+)"),
        'magnetic_moment': re.compile(r"Instant squared magnetic moment,\s*< m_z\^2 >\s+([0-9.]+)"),
        'coulomb_energy': re.compile(r"Coulomb correlation energy 1,\s*U<nn>/2\s+([0-9.]+)")
    }
    
    @staticmethod
    def parse_amulet_data(file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """Extract metadata and physical quantities from amulet.out file"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            print(f"Error: file {file_path} not found")
            return None
        
        try:
            content = file_path.read_text(encoding='utf-8')
            result = {}
            
            # Extract basic parameters
            for key, pattern in AmuletDataParser.PATTERNS.items():
                if key in ['legendre_cutoff', 'magnetic_moment', 'coulomb_energy']:
                    continue  
                
                match = pattern.search(content)
                if match:
                    value = match.group(1)
                    # Automatic type conversion
                    if key in ['degeneracy', 'beta', 'ntotal', 'U', 'J']:
                        result[key] = float(value)
                    else:
                        result[key] = value
            
            # Extract Legendre cutoff values
            cutoff_matches = AmuletDataParser.PATTERNS['legendre_cutoff'].findall(content)
            if cutoff_matches:
                cutoffs = list(map(int, cutoff_matches))
                result["legendre_cutoffs_GL_in_positions"] = cutoffs[::2]
                result["legendre_cutoffs_GL_out_positions"] = cutoffs[1::2]
            
            # Extract iteration-dependent data
            iteration_data = {}
            for quantity in ['magnetic_moment', 'coulomb_energy']:
                matches = AmuletDataParser.PATTERNS[quantity].findall(content)
                if matches:
                    iteration_data[quantity] = list(map(float, matches))
            
            if iteration_data:
                result.update(iteration_data)
                max_iterations = max(len(data) for data in iteration_data.values())
                result['iterations'] = list(range(1, max_iterations + 1))
            
            return result
            
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return None

class DataProcessor:
    """Class for processing column data for Green's-Legendre coefficients"""
    
    @staticmethod
    def load_columns(file_path: Union[str, Path]) -> Optional[Dict[int, np.ndarray]]:
        """Load data from .dat file and return columns as numpy arrays"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            print(f"Error: file {file_path} not found")
            return None
        
        try:
            content = file_path.read_text().splitlines()
            data_blocks = []
            current_block = []
            
            for line in content:
                stripped = line.strip()
                if not stripped:
                    if current_block:
                        data_blocks.append(current_block)
                        current_block = []
                else:
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            current_block.append(float(parts[1]))
                        except ValueError:
                            continue
            
            if current_block:
                data_blocks.append(current_block)
            
            return {idx: np.array(block) for idx, block in enumerate(data_blocks)}
            
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            return None

class HDF5Manager:
    """Manager for HDF5 file operations"""
    
    @staticmethod
    def save_to_hdf5(metadata: Dict[str, Any], files_info: List[FileInfo], 
                    output_path: Union[str, Path]) -> bool:
        """Save data to HDF5 file with grouping by iterations"""
        output_path = Path(output_path)
        
        try:
            with h5py.File(output_path, 'w') as hdf:
                # Determine number of iterations
                num_iterations = len(metadata.get('iterations', []))
                if num_iterations == 0:
                    print("No iteration data found")
                    return False
                
                # Create main data group
                data_group = hdf.create_group("data")
                
                # Save data for each iteration
                for iter_idx in range(num_iterations):
                    iter_group = data_group.create_group(f"iter{iter_idx}")
                    
                    # Save metadata for this iteration
                    for key, value in metadata.items():
                        if key in ['iterations', 'magnetic_moment', 'coulomb_energy', 
                                 'legendre_cutoffs_GL_in_positions', 'legendre_cutoffs_GL_out_positions']:
                            # For iteration-dependent data, save only current iteration value
                            if isinstance(value, (list, np.ndarray)) and len(value) > iter_idx:
                                iter_group.create_dataset(key, data=value[iter_idx])
                            else:
                                iter_group.create_dataset(key, data=value)
                        else:
                            # For scalar data, save as is
                            HDF5Manager._create_dataset(iter_group, key, value)
                    
                    # Save column data for this iteration
                    for file_info in files_info:
                        if file_info.columns and iter_idx in file_info.columns:
                            dataset_name = f"{file_info.name}_column_{iter_idx}"
                            iter_group.create_dataset(dataset_name, data=file_info.columns[iter_idx])
            
            print(f"Data successfully saved to {output_path}")
            return True
            
        except Exception as e:
            print(f"Error saving to HDF5: {e}")
            return False
    
    @staticmethod
    def _create_dataset(group, key: str, value: Any):
        """Helper method for creating HDF5 datasets"""
        if isinstance(value, (str, bytes)):
            group.create_dataset(key, data=value)
        elif isinstance(value, (int, float)):
            group.create_dataset(key, data=value)
        elif isinstance(value, (list, np.ndarray)):
            group.create_dataset(key, data=np.array(value))

class Plotter:
    """Class for data visualization"""
    
    @staticmethod
    def plot_columns(hdf5_file: Union[str, Path], iteration: int, 
                    dataset_names: List[str], title: str = "", 
                    xlabel: str = "Index", ylabel: str = "Value"):
        """Plot data from HDF5 file for specific iteration"""
        hdf5_file = Path(hdf5_file)
        
        if not hdf5_file.exists():
            print(f"Error: file {hdf5_file} not found")
            return
        
        try:
            with h5py.File(hdf5_file, 'r') as hdf:
                iter_path = f"data/iter{iteration}"
                if iter_path not in hdf:
                    print(f"Iteration {iteration} not found")
                    return
                
                iter_group = hdf[iter_path]
                plt.figure(figsize=(12, 6))
                
                for dataset_name in dataset_names:
                    if dataset_name in iter_group:
                        data = iter_group[dataset_name][()]
                        plt.plot(data, label=dataset_name)
                
                plt.title(f"{title} (Iteration {iteration})")
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.tight_layout()
                plt.show()
                
        except Exception as e:
            print(f"Error reading HDF5 file: {e}")
    
    @staticmethod
    def plot_iteration_data(hdf5_file: Union[str, Path], quantity_name: str, 
                           title: str = "", ylabel: str = ""):
        """Plot iteration-dependent data from HDF5 file"""
        hdf5_file = Path(hdf5_file)
        
        if not hdf5_file.exists():
            print(f"Error: file {hdf5_file} not found")
            return
        
        try:
            with h5py.File(hdf5_file, 'r') as hdf:
                data_group = hdf['data']
                iterations = []
                values = []
                
                for iter_name in sorted(data_group.keys(), key=lambda x: int(x[4:])):
                    iter_group = data_group[iter_name]
                    if quantity_name in iter_group:
                        iterations.append(int(iter_name[4:]))
                        values.append(iter_group[quantity_name][()])
                
                if not values:
                    print(f"Data for {quantity_name} not found")
                    return
                
                plt.figure(figsize=(10, 6))
                plt.plot(iterations, values, 'o-', linewidth=2, markersize=6)
                plt.xlabel('Iteration', fontsize=12)
                plt.ylabel(ylabel or quantity_name, fontsize=12)
                plt.title(title or f"{quantity_name} vs iterations", fontsize=14)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()
                
        except Exception as e:
            print(f"Error reading HDF5 file: {e}")
    
    @staticmethod
    def plot_all_iteration_data(hdf5_file: Union[str, Path]):
        """Plot all iteration-dependent data from HDF5 file"""
        quantity_configs = {
            'magnetic_moment': {
                'title': "Instantaneous squared magnetic moment",
                'ylabel': r"$\langle m_z^2 \rangle$"
            },
            'coulomb_energy': {
                'title': "Coulomb energy",
                'ylabel': "E_C"
            }
        }
        
        for quantity, config in quantity_configs.items():
            Plotter.plot_iteration_data(
                hdf5_file, quantity, 
                title=config['title'],
                ylabel=config['ylabel']
            )


def show_database_structure(hdf5_file: Union[str, Path]):
    """
    Display the structure of HDF5 database file
    """
    hdf5_file = Path(hdf5_file)
    
    if not hdf5_file.exists():
        print(f"File {hdf5_file} not found")
        return
    
    print(f"\nDatabase structure: {hdf5_file.name}")
    print("=" * 50)
    
    try:
        with h5py.File(hdf5_file, 'r') as hdf:
            # Show top-level groups
            for group_name in hdf.keys():
                group = hdf[group_name]
                print(f"📁 {group_name}/")
                
                # Show group contents
                if isinstance(group, h5py.Group):
                    for item_name in group.keys():
                        item = group[item_name]
                        
                        if isinstance(item, h5py.Dataset):
                            print(f"   📊 {item_name} - {item.shape} {item.dtype}")
                        elif isinstance(item, h5py.Group):
                            print(f"   📁 {item_name}/")
                            
                            # Show datasets in nested groups
                            for dataset_name in item.keys():
                                dataset = item[dataset_name]
                                if isinstance(dataset, h5py.Dataset):
                                    print(f"      📊 {dataset_name} - {dataset.shape} {dataset.dtype}")
    
    except Exception as e:
        print(f"Error: {e}")
    
# In[]
            
def main():
    """Main execution function"""
    # Configuration
    CONFIG = {
        'input_file': "amulet.out",
        'output_file': "amulet_data.h5",
        'data_files': [
            FileInfo(
                path="G0l_in_1_1,1.dat",
                name="G0l_in",
                description="Input Green's function"
            ),
            FileInfo(
                path="Gl_out_qmc2_1_1,1.dat",
                name="Gl_out",
                description="Output Green's function"
            )
        ]
    }
    
    # 1. Parse metadata
    parsed_metadata = AmuletDataParser.parse_amulet_data(CONFIG['input_file'])
    
    if not parsed_metadata:
        print("Failed to extract metadata")
        return
    
    # Display basic parameters
    print("Extracted metadata:")
    print("=" * 50)
    print("BASIC PARAMETERS:")
    for key in ['version', 'beta', 'ntotal', 'impurity_name', 'degeneracy', 'U', 'J']:
        if key in parsed_metadata:
            print(f"{key}: {parsed_metadata[key]}")
            
    # DISPLAY ENERGY AND MAGNETIC MOMENT
    print("\n" + "=" * 50)
    print("PHYSICAL QUANTITIES:")
    
    if 'magnetic_moment' in parsed_metadata:
        moments = parsed_metadata['magnetic_moment']
        print(f"Magnetic moment <m_z²>: {moments}")
    
    if 'coulomb_energy' in parsed_metadata:
        energies = parsed_metadata['coulomb_energy']
        print(f"\nCoulomb energy U<nn>/2: {energies}")
                
    # 2. Process data files
    processed_files = []
    for file_info in CONFIG['data_files']:
        columns = DataProcessor.load_columns(file_info.path)
        if columns is not None:
            file_info.columns = columns
            processed_files.append(file_info)
            print(f"\nLoaded: {file_info.path} ({len(columns)} columns)")
    
    # 3. Save to HDF5
    if processed_files:
        success = HDF5Manager.save_to_hdf5(parsed_metadata, processed_files, CONFIG['output_file'])
        
        if success:
            show_database_structure(CONFIG['output_file'])
            # 4. Visualization
            Plotter.plot_all_iteration_data(CONFIG['output_file'])
            
            # Plot Green's functions for specific iteration
            i=0
            Plotter.plot_columns(
                CONFIG['output_file'], i,
                dataset_names=["G0l_in_column_0"],
                title="Green's functions", 
                ylabel="G(l)"
            )
            
            Plotter.plot_columns(
                CONFIG['output_file'], i,
                dataset_names=["Gl_out_column_0"],
                title="Green's functions", 
                ylabel="G(l)"
            )
            
    else:
        print("No data to save")

if __name__ == "__main__":
    main()