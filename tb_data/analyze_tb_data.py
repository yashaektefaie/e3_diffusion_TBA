import torch
import sys 
sys.path.append('../')
from qm9.analyze import check_stability
from qm9 import bond_analyze
from configs.datasets_config import get_dataset_info
from qm9 import dataset
import pickle
from rdkit import Chem
import numpy as np
from tqdm import tqdm

# train_dataset = torch.load('train_dataset.pt')
# test_dataset = torch.load('test_dataset.pt')
# val_dataset = torch.load('val_dataset.pt')

path = "/n/holystore01/LABS/mzitnik_lab/Users/yektefaie/e3_diffusion_for_molecules/outputs/debug_10/args.pickle"

with open(path, 'rb') as f:
        args = pickle.load(f) 

dataset_info = get_dataset_info('TB', False)
dataloaders, charge_scale = dataset.retrieve_dataloaders(args)
atom_decoder = dataset_info['atom_decoder']
mol_stable_count = 0
total = 0
for block in tqdm(dataloaders['test']):
        for i in range(block['positions'].shape[0]):
                atom_pos = block['positions'][i][block['atom_mask'][i]]
                one_hot = block['one_hot'][i][block['atom_mask'][i]]
                atoms_types = one_hot.int().argmax(axis=1)
                atoms = [atom_decoder[atom] for atom in atoms_types]

                #Code to create a smile string
                mol = Chem.rdchem.EditableMol(Chem.Mol())

                for atomic_num in atoms:
                        mol.AddAtom(Chem.Atom(Chem.GetPeriodicTable().GetAtomicNumber(atomic_num)))
                        
                # # Convert the editable molecule object to a regular molecule object
                # mol = mol.GetMol()
                converter = lambda x: Chem.GetPeriodicTable().GetElementSymbol(atoms[x])

                distances = {}
                nr_bonds = np.zeros(len(atom_pos), dtype='int')
                for i in range(len(atoms)):
                        for j in range(i + 1, len(atoms)):
                                p1 = np.array(atom_pos[i])
                                p2 = np.array(atom_pos[j])
                                dist = np.sqrt(np.sum((p1 - p2) ** 2))
                                atom1, atom2 = atoms[i], atoms[j]
                                # pair = sorted([atom_type[i], atom_type[j]])
                                order = bond_analyze.get_bond_order(atom1, atom2, dist)
                                if order == 1:
                                        mol.AddBond(i, j, Chem.BondType.SINGLE)
                                       #print(f"Adding single bond between {atom1} and {atom2} dist {dist*100}")
                                elif order == 2:
                                        mol.AddBond(i, j, Chem.BondType.DOUBLE)
                                        #print(f"Adding double bond between {atom1} and {atom2} dist {dist*100}")
                                elif order == 3:
                                        mol.AddBond(i, j, Chem.BondType.TRIPLE)
                                        #print(f"Adding triple bond between {atom1} and {atom2} dist {dist*100}")
                                if i not in distances:
                                        distances[i] = {}
                                if j not in distances:
                                        distances[j] = {}

                                distances[i][j] = [dist*100, order, p2]
                                distances[j][i] = [dist*100, order, p1]

                                nr_bonds[i] += order
                                nr_bonds[j] += order
        
                nr_stable_bonds = 0
                num = 0
                for atom_type_i, nr_bonds_i in zip(atoms, nr_bonds):
                        possible_bonds = bond_analyze.allowed_bonds[atom_type_i]
                        if type(possible_bonds) == int:
                                is_stable = possible_bonds == nr_bonds_i
                        else:
                                is_stable = nr_bonds_i in possible_bonds
                        if not is_stable:
                                atm_of_interest = atom_type_i
                                # print("Invalid bonds for molecule %s with %d bonds" % (atm_of_interest, nr_bonds_i))
                                # print("Distances")
                                # for i in distances[num]:
                                #         print(f"Atom {atm_of_interest} to {converter(i)} dist {distances[num][i][0]} order {distances[num][i][1]} position {distances[num][i][2]}")
                                # import pdb; pdb.set_trace()
                        nr_stable_bonds += int(is_stable)
                        num += 1

                molecule_stable = nr_stable_bonds == len(atom_pos)
                if molecule_stable:
                        print("Molecule is stable")
                        mol_stable_count += 1
                        # Convert the editable molecule to a final molecule
                        final_mol = mol.GetMol()

                        # Convert the molecule to a SMILES string
                        smiles = Chem.MolToSmiles(final_mol)
                        print(smiles)  # Prints: CO

                        mol = Chem.MolFromSmiles(smiles)
                        # Remove hydrogen atoms
                        mol = Chem.RemoveHs(mol)

                        # Convert back to SMILES string
                        smiles_no_h = Chem.MolToSmiles(mol)
                        print(smiles_no_h)
                # else:
                #         print(f"Is molecule stable {molecule_stable}")
                # import pdb; pdb.set_trace()
                total += 1

print(f"{mol_stable_count} out of {total} molecules are stable")
print(mol_stable_count/total)

                # print(smiles_no_h)

                # pt = Chem.GetPeriodicTable()

               

                # # Set the 3D coordinates of the atoms
                # conf = Chem.Conformer(mol.GetNumAtoms())
                # for i in range(mol.GetNumAtoms()):
                #         conf.SetAtomPosition(i, atom_pos[i].tolist())
                # mol.AddConformer(conf)

                # # Generate a SMILES string for the molecule
                # smiles = Chem.MolToSmiles(mol)
                # print(f"Smiles {smiles}")
                
                #import pdb; pdb.set_trace()

#import pdb; pdb.set_trace()