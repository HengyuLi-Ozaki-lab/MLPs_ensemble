import os
from tqdm import tqdm
from ase import Atoms
from ase.io import read
from ase.io import Trajectory
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp import Xdatcar
from sklearn.model_selection import train_test_split

class InputParser:
    def __init__(self):
        self.supported_formats = ['cif', 'xyz', 'poscar', 'vasp']

    def parse_input(self, input_data):
        """
        Parse and standardize the input structure data.
        
        Args:
            input_data (str | Atoms | Structure): File path, ASE Atoms, or pymatgen Structure.
        
        Returns:
            tuple: (pymatgen_structure, ase_atoms)
        """
        if isinstance(input_data, str):  # File path
            return self._parse_file(input_data)
        elif isinstance(input_data, Atoms):  # ASE Atoms object
            pymatgen_structure = AseAtomsAdaptor.get_structure(input_data)
            return pymatgen_structure, input_data
        elif isinstance(input_data, Structure):  # pymatgen Structure object
            ase_atoms = AseAtomsAdaptor.get_atoms(input_data)
            return input_data, ase_atoms
        else:
            raise TypeError("Input must be a file path, ASE Atoms, or pymatgen Structure.")

    def _parse_file(self, file_path):
        """
        Internal function to parse structure file.
        
        Args:
            file_path (str): Path to the input structure file.
        
        Returns:
            tuple: (pymatgen_structure, ase_atoms)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} does not exist.")
        
        file_extension = file_path.split('.')[-1].lower()
        if file_extension not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        # Parse using ASE
        ase_atoms = read(file_path)

        # Convert to pymatgen Structure
        pymatgen_structure = AseAtomsAdaptor.get_structure(ase_atoms)
        
        return pymatgen_structure, ase_atoms

    def get_model_input(self, pymatgen_structure, ase_atoms, model_name):
        """
        Prepare input format for a specific model.
        
        Args:
            pymatgen_structure (Structure): Pymatgen structure object.
            ase_atoms (Atoms): ASE Atoms object.
            model_name (str): Name of the target model.
        
        Returns:
            object: Input structure formatted for the target model.
        """
        model_input_map = {
            'm3gnet': pymatgen_structure,
            'mattersim': ase_atoms,
            'eqv2': ase_atoms,
            'mace': ase_atoms,
        }
        
        if model_name not in model_input_map:
            raise ValueError(f"Unsupported model: {model_name}")
        
        return model_input_map[model_name]
    
    def batch_parse(self, directory_path, file_extensions=None):
        """
        Batch parse all structure files in the specified directory, including subdirectories,
        and filter by specified file extensions.
        
        Args:
            directory_path (str): Path to the directory containing structure files.
            file_extensions (list, optional): List of file extensions to filter (e.g., ['cif', 'xyz']).
                                              Defaults to self.supported_formats.
        
        Returns:
            list: A list of dictionaries with parsed structure data.
        """
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory {directory_path} does not exist.")
        
        # Use default formats if no file extensions are specified
        if file_extensions is None:
            file_extensions = self.supported_formats
        
        # Prepare storage for parsed structures
        parsed_structures = []
        all_files = []

        # Walk through directory and collect matching files
        for root, _, files in os.walk(directory_path):
            for file_name in files:
                file_extension = file_name.split('.')[-1].lower()
                if file_extension in file_extensions:
                    all_files.append(os.path.join(root, file_name))

        # Process files with progress bar
        for file_path in tqdm(all_files, desc="Parsing files"):
            try:
                pymatgen_structure, ase_atoms = self.parse_input(file_path)
                parsed_structures.append({
                    "file": file_path,
                    "pymatgen_structure": pymatgen_structure,
                    "ase_atoms": ase_atoms
                })
            except Exception as e:
                print(f"Failed to parse {file_path}: {e}")

        return parsed_structures
    
    def parse_xdatcar(self, directory_path, step_interval=1):
        """
        Parse all XDATCAR files in the specified directory and sample structures.

        Args:
            directory_path (str): Path to the directory containing XDATCAR files.
            step_interval (int): Interval for sampling structures (default is 1, meaning every step).

        Returns:
            list: A list of dictionaries with sampled structures.
                  Example:
                  [
                      {"file": "path/to/XDATCAR",
                       "sampled_structures": [structure1, structure2, ...]},
                      ...
                  ]
        """
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory {directory_path} does not exist.")
        
        # Prepare storage for sampled structures
        sampled_data = []
        all_xdatcar_files = []

        # Walk through directory to find XDATCAR files
        for root, _, files in os.walk(directory_path):
            for file_name in files:
                if file_name == "XDATCAR":
                    all_xdatcar_files.append(os.path.join(root, file_name))

        # Process XDATCAR files with progress bar
        for file_path in tqdm(all_xdatcar_files, desc="Processing XDATCAR files"):
            try:
                xdatcar = Xdatcar(file_path)
                all_structures = xdatcar.structures  # All structures in XDATCAR
                sampled_structures = [
                    all_structures[i] for i in range(0, len(all_structures), step_interval)
                ]

                # Convert pymatgen structures to ASE Atoms if needed
                sampled_atoms = [AseAtomsAdaptor.get_atoms(s) for s in sampled_structures]

                sampled_data.append({
                    "file": file_path,
                    "sampled_structures": sampled_structures,
                    "sampled_atoms": sampled_atoms
                })
            except Exception as e:
                print(f"Failed to process {file_path}: {e}")

        return sampled_data

def load_and_split_traj(traj_file, output_dir, test_size=0.2, random_state=42):
    """
    Load ASE .traj file, split into training and testing sets, and save to separate files.

    Args:
        traj_file (str): Path to the .traj file.
        output_dir (str): Directory to save the split datasets.
        test_size (float): Proportion of the data to include in the test set.
        random_state (int): Random seed for reproducibility.

    Returns:
        dict: A dictionary containing training and testing sets:
            {"train": list of structures,
             "test": list of structures}.
    """
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    traj = Trajectory(traj_file)
    data = []

    # Extract structures and energies
    for atoms in traj:
        try:
            # Convert ASE.Atoms to Pymatgen.Structure
            pmg_structure = Structure(
                lattice=atoms.cell,
                species=atoms.get_chemical_symbols(),
                coords=atoms.positions,
                coords_are_cartesian=True
            )
        except Exception as e:
            print(f"Warning: Failed to convert structure. {e}")
            pmg_structure = None

        data.append({
            "ase_atoms": atoms,
            "pymatgen_structure": pmg_structure,
            "total_energy": atoms.get_potential_energy(),
            "force": atoms.get_forces(),
            "stress": atoms.get_stress()
        })

    # Split the dataset
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)
    store_train, store_test = train_test_split(traj, test_size=test_size, random_state=random_state)
    '''
    # Save train and test datasets
    train_file = os.path.join(output_dir, "train_dataset.traj")
    test_file = os.path.join(output_dir, "test_dataset.traj")

    with Trajectory(train_file, "w") as train_traj:
        for atoms in store_train:
            train_traj.write(atoms)

    with Trajectory(test_file, "w") as test_traj:
        for atoms in store_test:
            test_traj.write(atoms)

    print(f"Training set saved to {train_file}")
    print(f"Test set saved to {test_file}")
    '''
    return {"train": train_data, "test": data}
