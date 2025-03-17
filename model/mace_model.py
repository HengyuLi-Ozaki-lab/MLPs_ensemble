import sys
import os
import contextlib
from pymatgen.core import Structure
from model.model_base import BaseModel
from mace.calculators import mace_mp
from utils.energy_correction import apply_mp2020_correction

class MACEModel(BaseModel):
    """
    Wrapper for MACE model using ASE calculator.
    """
    def __init__(self, model="small", dispersion=False, default_dtype="float32", device="cuda"):
        """
        Initialize the MACE model with suppressed print output.

        Args:
            model (str): The MACE model type (e.g., "small", "large").
            dispersion (bool): Whether to include dispersion interactions.
            default_dtype (str): Data type for computations.
            device (str): Device to use ("cpu" or "cuda").
        """
        super().__init__(model_name="MACE",required_input_format="ase")
        
        with self._suppress_output():
            self.calc = mace_mp(
                model=model,
                dispersion=dispersion,
                default_dtype=default_dtype,
                device=device,
            )

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

    def predict(self, structure, prediction_type=None):
        """
        Predict energy and forces for a given ASE Atoms structure.

        Args:
            structure (ase.Atoms): The input structure as an ASE Atoms object.

        Returns:
            dict: A dictionary containing predicted properties.
        """
        try:
            # Assign calculator to the structure
            structure.calc = self.calc

            # Predict energy and forces
            #energy = structure.get_total_energy()
            #forces = structure.get_forces()
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

            return result
            '''
            return {
                "energy": energy,
                "forces": forces,
            }
            '''
        except Exception as e:
            raise RuntimeError(f"MACE prediction failed: {e}")

    def predict_batch(self, structures):
        """
        Predict energy and forces for a batch of structures with error handling.

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