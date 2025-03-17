import sys
import os
import contextlib
from model.model_base import BaseModel
from fairchem.core import OCPCalculator
from pymatgen.core import Structure
from utils.energy_correction import apply_mp2020_correction

class EqV2Model(BaseModel):
    """
    Wrapper for EqV2 model using fairchem OCPCalculator.
    """
    def __init__(self, pretrained_path, model_version="eqV2_31M_omat_mp_salex", use_cpu=False, seed=42):
        """
        Initialize the EqV2 model and suppress unwanted outputs.
        """
        super().__init__(model_name="EqV2",required_input_format="ase")
        # Initialize the OCPCalculator

        with self._suppress_output():
            """
            self.calc = OCPCalculator(
                model_name=model_version,
                local_cache=pretrained_path,
                cpu=use_cpu,
                seed=seed,
            )
            """
            self.calc = OCPCalculator(
                checkpoint_path="/home/lee/mlps/pretrained_models/eqV2_153M_omat_mp_salex.pt",
                cpu=False,
                seed=42,
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
            dict: A dictionary with energy and forces.
        """
        # Assign calculator to structure
        structure.calc = self.calc

        # Predict properties
        results = {}
        if not prediction_type or "energy" in prediction_type:
            raw_energy = structure.get_total_energy()
            
            # Convert ASE.Atoms to Pymatgen.Structure for energy correction
            pmg_structure = Structure(
                lattice=structure.cell,
                species=structure.get_chemical_symbols(),
                coords=structure.get_positions(),
                coords_are_cartesian=True,
            )
            corrected_energy = apply_mp2020_correction(raw_energy, pmg_structure)
            results["energy"] = corrected_energy

        if not prediction_type or "force" in prediction_type:
            results["forces"] = structure.get_forces().tolist()

        return results
        '''
        # Predict energy and forces
        result = {}
        #energy = structure.get_total_energy()
        #forces = structure.get_forces()
        if not prediction_type or 'energy' in prediction_type:
            result['energy'] = structure.get_total_energy()
        if not prediction_type or 'force' in prediction_type:
            result['forces'] = structure.get_forces()
        
        return result
        '''
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
                    "energy": result["energy"],
                    "forces": result["forces"]
                })
            except Exception as e:
                # 捕获单个结构预测错误
                results.append({
                    "index": idx,
                    "error": str(e)
                })
        return results