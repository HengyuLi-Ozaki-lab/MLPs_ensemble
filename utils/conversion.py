from pymatgen.core import Structure
from ase import Atoms

def ase_to_pymatgen(ase_atoms):
    """
    Convert ASE Atoms to pymatgen Structure.

    Args:
        ase_atoms (ase.Atoms): The input ASE Atoms object.

    Returns:
        pymatgen.Structure: The converted pymatgen Structure object.
    """
    try:
        symbols = [atom.symbol for atom in ase_atoms]
        positions = ase_atoms.positions
        cell = ase_atoms.cell
        pmg_structure = Structure(
            lattice=cell,
            species=symbols,
            coords=positions,
            coords_are_cartesian=True
        )
        return pmg_structure
    except Exception as e:
        raise ValueError(f"Failed to convert ASE Atoms to pymatgen Structure: {e}")