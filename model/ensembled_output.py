import numpy as np
from tqdm import tqdm
from model.eqv2_model import EqV2Model
from model.chgnet_model import CHGNETModel
from model.mace_model import MACEModel
from model.matter_sim import MatterSimModel
from utils.conversion import ase_to_pymatgen

class ModelManager:
    """
    A manager to handle multiple ML models and unify their outputs.
    """
    def __init__(self, model_configs=None):
        """
        Initialize the ModelManager with specified model configurations.

        Args:
            model_configs (list of dict): List of model configuration dictionaries.
        """
        self.models = self._initialize_models(model_configs)

    def _get_structure_for_model(self, structure_data, model):
        """
        Get the appropriate structure format for the given model.

        Args:
            structure_data (dict): Dictionary containing "ase_atoms" and "pymatgen_structure".
            model (BaseModel): The model instance.

        Returns:
            The structure in the required format for the model.
        """
        required_format = model.required_input_format

        #print(f"Model {model.model_name} requires {required_format} format.")

        if required_format == "ase":
            if "ase_atoms" not in structure_data or structure_data["ase_atoms"] is None:
                raise ValueError(f"ASE Atoms not available for model {model.model_name}.")
            return structure_data["ase_atoms"]

        elif required_format == "pymatgen":
            if "pymatgen_structure" not in structure_data or structure_data["pymatgen_structure"] is None:
                raise ValueError(f"Pymatgen Structure not available for model {model.model_name}.")
            return structure_data["pymatgen_structure"]

        raise ValueError(f"Unsupported input format for model {model.model_name}: {required_format}")
    
    def _initialize_models(self, model_configs):
        """
        Initialize models based on configuration.

        Args:
            model_configs (list of dict): Model configuration list.

        Returns:
            list: List of model instances.
        """
        models = []
        for config in model_configs:
            model_name = config.get("name")
            if model_name == "EqV2":
                models.append(EqV2Model(**config.get("params", {})))
            elif model_name == "CHGNET":
                models.append(CHGNETModel(**config.get("params", {})))
            elif model_name == "MACE":
                models.append(MACEModel(**config.get("params", {})))
            elif model_name == "MatterSim":
                models.append(MatterSimModel(**config.get("params", {})))
            else:
                raise ValueError(f"Unsupported model name: {model_name}")
        return models

    def predict(self, structure, prediction_type=None):
        """
        Predict properties using all models for a single structure.

        Args:
            structure: The input structure (e.g., ASE Atoms or pymatgen Structure).

            prediction_type: The output of model (e.g. energy, force, stress, none=All)
        Returns:
            dict: Unified predictions from all models.
        """
        results = {}
        for model in self.models:
            try:
                structure = self._get_structure_for_model(structure, model)
                results[model.model_name] = model.predict(structure, prediction_type)
            except Exception as e:
                results[model.model_name] = {"error": str(e)}
        return results

    def predict_batch(self, structures, prediction_type=None):
        """
        Predict properties using all models for a batch of structures.

        Args:
            structures (list): List of input structures.

        Returns:
            list of dict: Unified predictions from all models for each structure.
        """
        unified_results = []
        for idx, structure in tqdm(enumerate(structures),desc="Prediction"):
            result = {"index": idx,"structure":structure["ase_atoms"].get_chemical_formula('reduce')}
            for model in self.models:
                try:
                    sub_structure = self._get_structure_for_model(structure, model)
                    result[model.model_name] = model.predict(sub_structure,prediction_type=prediction_type)
                except Exception as e:
                    result[model.model_name] = {"error": str(e)}
            unified_results.append(result)
        return unified_results
    
    @staticmethod
    def serialize_results(results):
        """
        Serialize prediction results to ensure JSON compatibility.
        """
        serialized = []
        for result in results:
            serialized_result = {}
            for key, value in result.items():
                if isinstance(value, np.ndarray):  # Convert ndarray to list
                    serialized_result[key] = value.tolist()
                elif isinstance(value, (np.float32, np.float64)):
                    serialized_result[key] = float(value)
                elif isinstance(value, (np.int32, np.int64)):
                    serialized_result[key] = int(value)
                elif isinstance(value, dict):
                    serialized_result[key] = ModelManager.serialize_results([value])[0]
                else:
                    serialized_result[key] = value
            serialized.append(serialized_result)
        return serialized

    def save_results(self, results, output_file):
        """
        Save unified results to a JSON file.
        """
        import json
        serialized_results = self.serialize_results(results)
        with open(output_file, "w") as f:
            json.dump(serialized_results, f, indent=4)

