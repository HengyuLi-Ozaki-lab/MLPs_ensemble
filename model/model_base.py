class BaseModel:
    """
    Base class for machine learning potential models.
    """
    def __init__(self, model_name, required_input_format):
        self.model_name = model_name
        self.required_input_format = required_input_format
    def predict(self, structure, prediction_type=None):
        """
        Predict properties based on the input structure.
        
        Args:
            structure: Input structure (ASE Atoms or pymatgen Structure).
        
        Returns:
            dict: Prediction results.
        """
        raise NotImplementedError("Subclasses must implement this method.")