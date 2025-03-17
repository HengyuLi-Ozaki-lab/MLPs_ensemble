from sklearn.linear_model import LinearRegression, BayesianRidge

def train_linear_model(dataframe, model_type="linear"):
    """
    Train a linear model using aggregated predictions.

    Args:
        dataframe (pd.DataFrame): DataFrame with model predictions and true energies.
        model_type (str): Type of linear model ('linear' for LinearRegression, 'bayesian' for BayesianRidge).

    Returns:
        object: Trained linear model.
    """
    # Features are model predictions
    X = dataframe.drop(columns=["True Energy"])
    # Target is the true energy
    y = dataframe["True Energy"]

    # Choose model
    if model_type == "linear":
        model = LinearRegression()
    elif model_type == "bayesian":
        model = BayesianRidge()
    else:
        raise ValueError("Unsupported model type. Use 'linear' or 'bayesian'.")

    # Train the model
    model.fit(X, y)
    print("Model training complete.")
    return model