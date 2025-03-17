from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from pymatgen.entries.computed_entries import ComputedStructureEntry
from pymatgen.core import Structure

def apply_mp2020_correction(energy, structure):
    """
    Apply MaterialsProject2020Compatibility correction to the predicted energy.

    Args:
        energy (float): The raw predicted energy (eV).
        structure (pymatgen.Structure): Input structure.

    Returns:
        float: Corrected energy.
    """
    # Compatibility processor
    compatibility = MaterialsProject2020Compatibility(check_potcar=False)

    # Define potcar_symbols for all elements in the structure
    #potcar_symbols = [f"PBE {el.symbol}" for el in structure.composition.elements]

    # Create a ComputedStructureEntry with the required parameters
    entry = ComputedStructureEntry(
        structure=structure,
        energy=energy,
        parameters={
            "is_hubbard": False,
            "run_type": "GGA"
            #"potcar_symbols": potcar_symbols
        }
    )

    try:
        # Apply the compatibility correction
        corrected_entry = compatibility.process_entry(entry)
        if corrected_entry:
            return corrected_entry.energy
        else:
            raise ValueError("Failed to apply energy correction.")
    except Exception as e:
        print(f"Error during energy correction: {e}")
        return energy  # Return raw energy if correction fails

def calculate_formation_energy(total_energy, chemical_composition, chemical_potentials):
    """
    Calculate the formation energy of a structure.

    Args:
        total_energy (float): Total energy of the structure (eV).
        chemical_composition (dict): Dictionary with element symbols as keys and counts as values.
        chemical_potentials (dict): Dictionary with element symbols as keys and chemical potentials as values.

    Returns:
        float: Formation energy (eV).
    """
    # Calculate the chemical energy contribution
    chemical_energy = sum(chemical_composition[element] * chemical_potentials[element]
                          for element in chemical_composition)

    # Formation energy
    formation_energy = total_energy - chemical_energy
    return formation_energy

def correct_dataset_to_formation_energy(dataset, chemical_potentials):
    """
    Correct total energies in a dataset to formation energies.

    Args:
        dataset (list): List of dictionaries with keys 'ase_atoms' and 'true_energy'.
        chemical_potentials (dict): Chemical potentials for each element.

    Returns:
        list: Dataset with formation energies added.
    """
    corrected_data = []

    for entry in dataset:
        atoms = entry["ase_atoms"]
        total_energy = entry["true_energy"]

        # Get the chemical composition (as a dictionary of element counts)
        chemical_composition = atoms.get_chemical_symbols()
        composition_dict = {element: chemical_composition.count(element) for element in set(chemical_composition)}

        # Calculate formation energy
        formation_energy = calculate_formation_energy(total_energy, composition_dict, chemical_potentials)

        # Add to corrected dataset
        corrected_entry = entry.copy()
        corrected_entry["formation_energy"] = formation_energy
        corrected_data.append(corrected_entry)

    return corrected_data