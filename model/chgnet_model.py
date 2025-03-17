import os
import contextlib
import sys
from model.model_base import BaseModel
from chgnet.model.model import CHGNet

class CHGNETModel(BaseModel):
    """
    Wrapper for CHGNet model.
    """
    def __init__(self):
        """
        Initialize the CHGNet model.
        """
        super().__init__(model_name="CHGNET",required_input_format="pymatgen")
        with self._suppress_output():
            self.model = CHGNet.load()  # Load the CHGNet pretrained model

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

    def predict(self, structure, prediction_type=None, return_site_energies=True, return_crystal_feas=True):
        """
        Predict energy, forces, and other properties using CHGNet.

        Args:
            structure (pymatgen.Structure): The input structure.
            return_site_energies (bool): Whether to return site energies.
            return_crystal_feas (bool): Whether to return crystal features.

        Returns:
            dict: A dictionary containing predicted properties.
        """
        try:
            prediction = self.model.predict_structure(
                structure,
                return_site_energies=return_site_energies,
                return_crystal_feas=return_crystal_feas,
            )

            result = {}

            if not prediction_type or 'energy' in prediction_type:
                result['energy'] = sum(prediction["site_energies"])
            if not prediction_type or 'force' in prediction_type:
                result['forces'] = prediction["f"].tolist()
            '''
            return {
                "total_energy": sum(prediction["site_energies"]),
                "site_energy": prediction["site_energies"],
                "forces": prediction["f"],
                "stress": prediction["s"],
                "magmom": prediction.get("m"),  # Optional: Only if magmom exists
            }
            '''
            return result
        except Exception as e:
            raise RuntimeError(f"CHGNET prediction failed: {e}")

    def predict_batch(self, structures):
        """
        Predict properties for a batch of structures with error handling.

        Args:
            structures (list of pymatgen.Structure): List of pymatgen Structure objects.

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

    def predict_batch(self, structures):
        """
        Predict properties for a batch of structures with error handling.

        Args:
            structures (list of pymatgen.Structure): List of pymatgen Structure objects.

        Returns:
            list of dict: List of prediction results for each structure, including errors.
        """
        results = []
        for idx, structure in enumerate(structures):
            try:
                prediction = self.model.predict_structure(
                    structure,
                    return_site_energies=return_site_energies,
                    return_crystal_feas=return_crystal_feas,
                )
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