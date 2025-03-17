import sys
import os
import contextlib
import torch
from pymatgen.core import Structure
from model.model_base import BaseModel
from mattersim.forcefield import MatterSimCalculator
from utils.energy_correction import apply_mp2020_correction

class MatterSimModel(BaseModel):
    """
    Wrapper for MatterSim model using ASE calculator.
    """
    def __init__(self, load_path="MatterSim-v1.0.0-5M.pth", device="cuda"):
        """
        Initialize the MatterSim model.

        Args:
            load_path (str): Path to the pretrained MatterSim model.
            device (str): Device to use ("cpu" or "cuda").
        """
        super().__init__(model_name="MatterSim",required_input_format="ase")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.calc = MatterSimCalculator(load_path=load_path, device=device)

    @contextlib.contextmanager
    def _suppress_output(self):
        """
        Suppress stdout and stderr during the context.
        """
        with open(os.devnull, 'w') as devnull:
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            try:
                sys.stdout = devnull
                sys.stderr = devnull
                yield
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
    
    def predict(self, structure,prediction_type=None):
        """
        Predict energy, forces, and stress for a given ASE Atoms structure.

        Args:
            structure (ase.Atoms): The input structure as an ASE Atoms object.

        Returns:
            dict: A dictionary containing predicted properties.
        """
        try:
            # Assign calculator to the structure
            structure.calc = self.calc

            # Predict properties

            result = {}
            if not prediction_type or 'energy' in prediction_type:
                raw_energy = structure.get_total_energy()
                
                # Convert ASE.Atoms to Pymatgen.Structure for energy correction
                pmg_structure = Structure(
                    lattice=structure.cell,
                    species=structure.get_chemical_symbols(),
                    coords=structure.get_positions(),
                    coords_are_cartesian=True,
                )
                corrected_energy = apply_mp2020_correction(raw_energy, pmg_structure)
                result["energy"] = corrected_energy
                
            if not prediction_type or 'force' in prediction_type:
                result['forces'] = structure.get_forces().tolist()
                
            #energy = structure.get_potential_energy()
            #forces = structure.get_forces()
            #stress = structure.get_stress(voigt=False)
            '''
            return {
                "energy": energy,
                "forces": forces,
                "stress": stress,
            }
            '''
            return result
        except Exception as e:
            raise RuntimeError(f"MatterSim prediction failed: {e}")

    def predict_batch(self, structures):
        """
        Predict properties for a batch of structures with error handling.

        Args:
            structures (list of ase.Atoms): List of ASE Atoms objects.

        Returns:
            list of dict: List of prediction results for each structure, including errors.
        """
        results = []
        for idx, structure in enumerate(structures):
            try:
                result = self.predict(structure)
                results.append({
                    "index": idx,
                    **result,
                })
            except Exception as e:
                results.append({
                    "index": idx,
                    "error": str(e),
                })
        return results