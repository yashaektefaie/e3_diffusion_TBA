import torch
from sklearn.model_selection import train_test_split
from rdkit import Chem

record_coords = False
record_smiles = False
record_tb_inhibition = False
count = 0

coordinates = []
atoms = []
smiles = []
tb_inhibition = []

for line in open('normalized_TB_inhibition_resazurin_62498_molecules_3d.sdf').readlines():
    if 'RDKit' in line:
        record_coords = True
        count = 0
        new_coord = []
        new_atoms = []
    
    if count > 2 and record_coords:
        data = [i for i in line.split(' ') if i]
        if data[0] != '1' and data[0] != 'M' and data[3] != '0\n' and data[3] != '1\n' and data[3] != '6\n':
            new_coord.append([float(data[0]), float(data[1]), float(data[2])])
            new_atoms.append(data[3])
        else:
            record_coords = False

    if 'SMILES' in line:
        record_smiles = True
    elif record_smiles:
        smiles.append(line.strip())
        record_smiles = False
    
    if 'tb_inhibition' in line:
        record_tb_inhibition = True
    elif record_tb_inhibition:
        tb_inhibition.append(float(line.strip()))
        record_tb_inhibition = False
        new_coord = torch.tensor(new_coord)
        # new_coord = (new_coord - new_coord.max())/(new_coord.max() - new_coord.min())
        # new_coord = new_coord - torch.mean(new_coord)
        coordinates.append(new_coord.tolist())
        atoms.append(new_atoms)
        
    count += 1

atom_charges = []
for i in atoms:
    atom_charges.append([Chem.GetPeriodicTable().GetAtomicNumber(j) for j in i])
# import pdb; pdb.set_trace()
max_length_atoms = max([len(i) for i in atoms])
for i in coordinates:
    while len(i) < max_length_atoms:
        i.append([0, 0, 0])

unique_atoms = list(set([j for i in atom_charges for j in i]))

for i in atom_charges:
    while len(i) < max_length_atoms:
        i.append(0)

coordinates = torch.tensor(coordinates)

atom_charges = torch.tensor(atom_charges)
tb_inhibition = torch.tensor(tb_inhibition)

def get_train_val_test(data, proportion_train):
    train_data, test_data = train_test_split(data, test_size=1-proportion_train, random_state=42)
    # val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42)
    return train_data, test_data, test_data

# Get the indices of the data
indices = list(range(len(coordinates)))

# Split the indices
train_indices, val_indices, test_indices = get_train_val_test(indices, 0.8)

# Use the indices to index into coordinates, smiles, and atoms
train_coords = coordinates[train_indices]
val_coords = coordinates[val_indices]
test_coords = coordinates[test_indices]

train_atom_charges = atom_charges[train_indices]
val_atom_charges = atom_charges[val_indices]
test_atom_charges = atom_charges[test_indices]

train_tb_inhibition = tb_inhibition[train_indices]
val_tb_inhibition = tb_inhibition[val_indices]
test_tb_inhibition = tb_inhibition[test_indices]

train_smiles = [smiles[i] for i in train_indices]
val_smiles = [smiles[i] for i in val_indices]
test_smiles = [smiles[i] for i in test_indices]

train_atoms = [atoms[i] for i in train_indices]
val_atoms = [atoms[i] for i in val_indices]
test_atoms = [atoms[i] for i in test_indices]

#'smiles': train_smiles,
#'smiles': val_smiles,
#'smiles': test_smiles,

train_dataset = {'positions': train_coords, 'num_atoms': torch.tensor([len(i) for i in train_atoms]), 
                'charges': train_atom_charges, 
                'tb_inhibition': train_tb_inhibition}

val_dataset = {'positions': val_coords, 'num_atoms': torch.tensor([len(i) for i in val_atoms]),
                'charges': val_atom_charges, 
                'tb_inhibition': val_tb_inhibition}

test_dataset = {'positions': test_coords, 'num_atoms': torch.tensor([len(i) for i in test_atoms]),
                'charges': test_atom_charges, 
                'tb_inhibition': test_tb_inhibition}

torch.save(train_dataset, 'train_dataset.pt')
torch.save(val_dataset, 'val_dataset.pt')
torch.save(test_dataset, 'test_dataset.pt')

#Flatten atoms list to find number of atoms
# unique_atoms = list(set([j for i in atoms for j in i]))
#VITAL THE CODE ASSUMES THIS!!! HAS TO BE SORTED
unique_atoms.sort()

n_atoms = len(unique_atoms)
atom_encoder = {}
atom_decoder = []

for n, i in enumerate(unique_atoms):
    #Get atom symbol from charge
    atom_encoder[Chem.GetPeriodicTable().GetElementSymbol(i)] = n
    atom_decoder.append(i)

n_nodes = {}
atom_types  = {}
max_nodes = 0
for i in atoms:
    if len(i) not in n_nodes:
        n_nodes[len(i)] = 1
    else:
        n_nodes[len(i)] += 1
    
    for j in i:
        if atom_encoder[j] not in atom_types:
            atom_types[atom_encoder[j]] = 1
        else:
            atom_types[atom_encoder[j]] += 1
    
    if len(i) > max_nodes:
        max_nodes = len(i)
    

with_h = True

print(atom_encoder)
print(atom_decoder)
print(n_nodes)
print(atom_types)
print(max_nodes)

