import h5py
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt

class HDF5Merger:
    """
    Class for merging multiple HDF5 files into one
    """
    
    @staticmethod
    def merge_hdf5_files(input_dirs: List[Path], output_file: Path) -> bool:
        """
        Merge multiple HDF5 files into one with grouping by parameters
        
        Args:
            input_dirs: List of paths to directories containing amulet_data.h5
            output_file: Path for output file
            
        Returns:
            bool: True if successful, False if error
        """
        try:
            with h5py.File(output_file, 'w') as output_hdf:
                # Create main group for all data
                main_group = output_hdf.create_group("merged_data")
                
                for input_dir in input_dirs:
                    h5_file = input_dir / "amulet_data.h5"
                    
                    if not h5_file.exists():
                        print(f"Warning: file {h5_file} not found, skipping")
                        continue
                    
                    # Create group with directory name (e.g.: beta10_ntot0.2)
                    dir_name = input_dir.name
                    dir_group = main_group.create_group(dir_name)
                    
                    # Copy all data from source file, skipping data/ folder
                    HDF5Merger._copy_hdf5_structure_without_data_folder(h5_file, dir_group)
                    
                    print(f"Added data from: {dir_name}")
            
            print(f"\nAll data successfully merged into: {output_file}")
            return True
            
        except Exception as e:
            print(f"Error merging files: {e}")
            return False
    
    @staticmethod
    def _copy_hdf5_structure_without_data_folder(source_file: Path, target_group: h5py.Group):
        """
        Copy HDF5 file structure, skipping data/ folder
        """
        with h5py.File(source_file, 'r') as source_hdf:
            for name, item in source_hdf.items():
                if isinstance(item, h5py.Group):
                    if name == 'data':
                        # Skip data folder and copy its contents directly
                        HDF5Merger._copy_group_contents(item, target_group)
                    else:
                        # Copy regular groups as is
                        new_group = target_group.create_group(name)
                        HDF5Merger._copy_group_contents(item, new_group)
                elif isinstance(item, h5py.Dataset):
                    # Copy dataset
                    target_group.create_dataset(name, data=item[()])
    
    @staticmethod
    def _copy_group_contents(source_group: h5py.Group, target_group: h5py.Group):
        """
        Copy group contents
        """
        for name, item in source_group.items():
            if isinstance(item, h5py.Group):
                new_group = target_group.create_group(name)
                HDF5Merger._copy_group_contents(item, new_group)
            elif isinstance(item, h5py.Dataset):
                target_group.create_dataset(name, data=item[()])
    
    @staticmethod
    def find_hdf5_directories(base_path: Path, pattern: str = "*") -> List[Path]:
        """
        Find all directories containing amulet_data.h5
        
        Args:
            base_path: Base directory for search
            pattern: Pattern for directory search
            
        Returns:
            List[Path]: List of directory paths
        """
        directories = []
        
        if not base_path.exists():
            print(f"Error: base directory {base_path} does not exist")
            return directories
        
        # Find all subdirectories matching the pattern
        for dir_path in base_path.glob(pattern):
            if dir_path.is_dir():
                h5_file = dir_path / "amulet_data.h5"
                if h5_file.exists():
                    directories.append(dir_path)
                    print(f"Found data directory: {dir_path.name}")
        
        return directories

def show_merged_structure(hdf5_file: Path, max_items: int = 5):
    """Display structure of merged HDF5 file"""
    if not hdf5_file.exists():
        print(f"File {hdf5_file} not found")
        return
    
    print(f"\nStructure of merged file: {hdf5_file.name}")
    print("=" * 60)
    
    try:
        with h5py.File(hdf5_file, 'r') as hdf:
            if 'merged_data' not in hdf:
                print("No data in merged_data group")
                return
            
            merged_group = hdf['merged_data']
            
            for param_group_name in sorted(merged_group.keys()):
                param_group = merged_group[param_group_name]
                print(f"📁 {param_group_name}/")
                
                # Show parameter group contents
                _print_group_structure(param_group, indent=3, max_items=max_items)
                
    except Exception as e:
        print(f"Error reading file: {e}")

def _print_group_structure(group: h5py.Group, indent: int = 0, max_items: int = 5):
    """Recursively print group structure with item limit"""
    indent_str = " " * indent
    items = list(group.items())
    
    for i, (name, item) in enumerate(items):
        if i >= max_items and len(items) > max_items:
            print(f"{indent_str}... and {len(items) - max_items} more items")
            break
            
        if isinstance(item, h5py.Group):
            print(f"{indent_str}📁 {name}/")
            _print_group_structure(item, indent + 2, max_items)
        elif isinstance(item, h5py.Dataset):
            print(f"{indent_str}📊 {name} - {item.shape} {item.dtype}")

class MergedDataAnalyzer:
    """Class for analyzing merged data"""
    
    @staticmethod
    def compare_parameters(merged_file: Path, parameter_name: str, 
                          dataset_path: str, iterations: List[int] = None):
        """
        Compare data for different parameters
        
        Args:
            merged_file: Path to merged file
            parameter_name: Parameter name for comparison (e.g., 'magnetic_moment')
            dataset_path: Path to dataset (e.g., 'iter0/magnetic_moment')
            iterations: List of iterations to compare (None for all)
        """
        if not merged_file.exists():
            print(f"File {merged_file} not found")
            return
        
        try:
            with h5py.File(merged_file, 'r') as hdf:
                if 'merged_data' not in hdf:
                    print("No merged data found")
                    return
                
                merged_group = hdf['merged_data']
                plt.figure(figsize=(12, 8))
                
                for param_name in sorted(merged_group.keys()):
                    param_group = merged_group[param_name]
                    
                    # Extract beta and ntot from group name
                    beta = param_name.split('_')[0].replace('beta', '')
                    ntot = param_name.split('_')[1].replace('ntot', '')
                    
                    try:
                        # Try to find dataset
                        dataset = param_group[dataset_path]
                        data = dataset[()]
                        
                        if iterations is None:
                            # If dataset is array, show all values
                            if hasattr(data, '__len__') and not isinstance(data, str):
                                plt.plot(data, 'o-', label=f"β={beta}, n={ntot}", markersize=4)
                            else:
                                # Scalar value
                                plt.scatter(0, data, label=f"β={beta}, n={ntot}", s=100)
                        else:
                            # Specific iterations
                            if hasattr(data, '__len__') and len(data) > max(iterations or [0]):
                                for iter_idx in iterations:
                                    plt.scatter(iter_idx, data[iter_idx], 
                                               label=f"β={beta}, n={ntot} iter{iter_idx}", s=100)
                    
                    except KeyError:
                        print(f"Dataset {dataset_path} not found in {param_name}")
                        continue
                
                plt.xlabel('Iteration' if iterations is None else 'Iteration index')
                plt.ylabel(parameter_name)
                plt.title(f"Comparison of {parameter_name} for different parameters")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()
                
        except Exception as e:
            print(f"Error analyzing data: {e}")
    
    @staticmethod
    def plot_gl_data(merged_file: Path, gl_type: str, iteration: int = 0, 
                    param_filter: str = None):
        """
        Plot gl_in or gl_out data for different parameters
        
        Args:
            merged_file: Path to merged file
            gl_type: 'G0l_in' or 'Gl_out'
            iteration: Iteration number
            param_filter: Parameter name filter (e.g., 'beta10_ntot0.2')
        """
        if not merged_file.exists():
            print(f"File {merged_file} not found")
            return
        
        if gl_type not in ['G0l_in', 'Gl_out']:
            print("gl_type must be 'G0l_in' or 'Gl_out'")
            return
        
        try:
            with h5py.File(merged_file, 'r') as hdf:
                if 'merged_data' not in hdf:
                    print("No merged data found")
                    return
                
                merged_group = hdf['merged_data']
                plt.figure(figsize=(12, 8))
                
                for param_name in sorted(merged_group.keys()):
                    if param_filter and param_filter not in param_name:
                        continue
                    
                    param_group = merged_group[param_name]
                    
                    # Extract beta and ntot from group name
                    beta = param_name.split('_')[0].replace('beta', '')
                    ntot = param_name.split('_')[1].replace('ntot', '')
                    
                    # Path to gl data (now without data/ folder)
                    dataset_name = f"{gl_type}_column_{iteration}"
                    iter_path = f"iter{iteration}"
                    
                    try:
                        # Check if iter group exists
                        if iter_path not in param_group:
                            print(f"Iteration {iteration} not found for {param_name}")
                            continue
                            
                        iter_group = param_group[iter_path]
                        
                        # Look for dataset with correct name
                        if dataset_name not in iter_group:
                            print(f"Dataset {dataset_name} not found in {param_name}")
                            continue
                            
                        dataset = iter_group[dataset_name]
                        data = dataset[()]
                        
                        plt.plot(data, 'o-', label=f"β={beta}, n={ntot}", markersize=3, linewidth=1)
                        
                    except KeyError:
                        print(f"Data {gl_type} not found for {param_name} iter{iteration}")
                        continue
                
                plt.xlabel('Index')
                plt.ylabel(gl_type)
                plt.title(f"{gl_type} data (iteration {iteration})")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()
                
        except Exception as e:
            print(f"Error plotting data: {e}")
    
    @staticmethod
    def list_available_parameters(merged_file: Path):
        """Show list of available parameters and datasets"""
        if not merged_file.exists():
            print(f"File {merged_file} not found")
            return
        
        try:
            with h5py.File(merged_file, 'r') as hdf:
                if 'merged_data' not in hdf:
                    print("No merged data found")
                    return
                
                merged_group = hdf['merged_data']
                print("Available parameters:")
                print("=" * 40)
                
                for param_name in sorted(merged_group.keys()):
                    print(f"📊 {param_name}")
                    
                    # Check for data existence
                    param_group = merged_group[param_name]
                    if 'iter0' in param_group:
                        iter_group = param_group['iter0']
                        print("   Example datasets in iter0:")
                        
                        # Find gl_in and gl_out datasets
                        gl_datasets = []
                        for ds_name in iter_group.keys():
                            if isinstance(iter_group[ds_name], h5py.Dataset):
                                if ds_name.startswith('G0l_in_') or ds_name.startswith('Gl_out_'):
                                    gl_datasets.append(ds_name)
                        
                        # Show gl datasets
                        for gl_ds in sorted(gl_datasets)[:5]:  # Show first 5
                            dataset = iter_group[gl_ds]
                            print(f"     📊 {gl_ds} - {dataset.shape}")
                        
                        if len(gl_datasets) > 5:
                            print(f"     ... and {len(gl_datasets) - 5} more gl datasets")
                        
                        break
                
        except Exception as e:
            print(f"Error: {e}")
    
    @staticmethod
    def inspect_parameter_structure(merged_file: Path, param_name: str):
        """Show detailed structure of specific parameter"""
        if not merged_file.exists():
            print(f"File {merged_file} not found")
            return
        
        try:
            with h5py.File(merged_file, 'r') as hdf:
                if 'merged_data' not in hdf:
                    print("No merged data found")
                    return
                
                if param_name not in hdf['merged_data']:
                    print(f"Parameter {param_name} not found")
                    return
                
                param_group = hdf['merged_data'][param_name]
                print(f"Detailed structure of parameter: {param_name}")
                print("=" * 50)
                
                _print_group_structure(param_group, indent=2, max_items=20)
                
        except Exception as e:
            print(f"Error: {e}")

def merge_amulet_data(base_dir: str, output_file: str = "merged_amulet_data.h5", 
                     pattern: str = "beta*_ntot*") -> bool:
    """
    Main function for merging files in Jupyter Notebook
    
    Args:
        base_dir: Path to base directory with parameter folders
        output_file: Output file name
        pattern: Pattern for directory search
        
    Returns:
        bool: True if successful, False if error
    """
    base_path = Path(base_dir)
    output_path = Path(output_file)
    
    print("Searching for AMULET data directories...")
    print("=" * 50)
    
    # Find all directories with data
    directories = HDF5Merger.find_hdf5_directories(base_path, pattern)
    
    if not directories:
        print("No directories with amulet_data.h5 found")
        print("Check path and search pattern")
        return False
    
    print(f"\nFound {len(directories)} data directories:")
    for dir_path in directories:
        print(f"  - {dir_path.name}")
    
    # Merge files
    print(f"\nMerging data into file: {output_file}")
    print("=" * 50)
    
    success = HDF5Merger.merge_hdf5_files(directories, output_path)
    
    if success:
        # Show merged file structure
        show_merged_structure(output_path)
        
        print(f"\n✅ Merge completed successfully!")
        print(f"File: {output_path.absolute()}")
        print(f"Total datasets: {len(directories)}")
    
    return success

if __name__ == "__main__":
    # Specify path to your data directory
    base_directory = "."  
    
    # Run merge
    success = merge_amulet_data(
        base_dir=base_directory,
        output_file="merged_amulet_data.h5",
        pattern="beta*_ntot*"
    )
    
    if success:
        # Analyze merged data
        analyzer = MergedDataAnalyzer()
        analyzer.list_available_parameters(Path("merged_amulet_data.h5"))
        
        # Example gl data plotting
        analyzer.plot_gl_data(Path("merged_amulet_data.h5"), "G0l_in", iteration=0)
        analyzer.plot_gl_data(Path("merged_amulet_data.h5"), "Gl_out", iteration=0)