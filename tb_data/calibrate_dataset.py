from rdkit import Chem
import numpy as np
from tqdm import tqdm

def read_sdf_file(file_path):
    supplier = Chem.SDMolSupplier(file_path)
    molecules = [Chem.AddHs(mol) for mol in supplier if mol is not None]
    return molecules

# Example usage:
file_path = 'normalized_TB_inhibition_resazurin_62498_molecules_3d.sdf'
molecules = read_sdf_file(file_path)

def calculate_average_distance(molecules):
    bond_distances = {}
    for mol in molecules:
        conf = mol.GetConformer()
        for bond in mol.GetBonds():
            atom1 = bond.GetBeginAtom()
            atom2 = bond.GetEndAtom()
            bond_type = tuple(sorted((atom1.GetSymbol(), atom2.GetSymbol())))
            bond_order = bond.GetBondTypeAsDouble()
            atom1_pos = np.array(conf.GetAtomPosition(atom1.GetIdx()))
            atom2_pos = np.array(conf.GetAtomPosition(atom2.GetIdx()))
            distance = np.linalg.norm(atom1_pos - atom2_pos)
            if (bond_type, bond_order) not in bond_distances:
                bond_distances[(bond_type, bond_order)] = []
            bond_distances[(bond_type, bond_order)].append(distance)
    average_distances = {(bond, order): np.mean(distances) for (bond, order), distances in bond_distances.items()}
    return average_distances

# Example usage:
average_distances = calculate_average_distance(molecules)
for (bond, order), distance in average_distances.items():
    print(f"Average distance for {bond[0]}-{bond[1]} bond of order {order}: {100*distance}")

import pdb; pdb.set_trace()

