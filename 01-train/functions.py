import re
import os 
from ase.constraints import FixAtoms
import numpy as np
#from ocdata.core import Adsorbate, AdsorbateSlabConfig, Bulk, Slab
#import ocdata
from pathlib import Path
import json
from ase.io import read
from ase.db import connect
import sys
from tqdm import tqdm
import lmdb
import torch
from collections import OrderedDict
import pickle
from ase.calculators.singlepoint import SinglePointCalculator

#已知tags设置constraint
def set_constraints_from_tags(atoms):
    assert np.any(atoms.get_tags() == 1)
    tags = atoms.get_tags()
    indices_to_fix = np.where(tags == 0)[0]
    fix_constraint = FixAtoms(indices=indices_to_fix)
    atoms.set_constraint(fix_constraint)
    return atoms


def initial_constraints(atoms, n):
    atom_positions = atoms.get_positions()
    z_positions = [pos[2] for pos in atom_positions]
    sorted_indices = sorted(enumerate(z_positions), key=lambda x: x[1])
    min_32_indices = [index for index, value in sorted_indices[:n]]

    
    '''
    min_z, max_z = min(z_positions), max(z_positions)
    rounded_z_positions = [round(pos[2] - min_z) for pos in atom_positions]
    layer_identifiers = sorted(set(rounded_z_positions))
    atom_index_mapping = {index: layer_identifiers.index(rounded_z) for index, rounded_z in enumerate(rounded_z_positions)}
    min_3_layers_indices = [index for index, layer in atom_index_mapping.items() if layer < (int(len(layer_identifiers))-2)]
    fix_constraint = FixAtoms(indices=min_3_layers_indices)
    '''
    fix_constraint = FixAtoms(indices=min_32_indices)
    atoms.set_constraint(fix_constraint)
    return atoms


def set_tags_from_constraints(atoms):
    assert atoms.constraints
    num_atoms = len(atoms)
    tags = np.ones(num_atoms, dtype=int)
    match = re.search(r'\[FixAtoms\(indices=(\[.*\])\)\]', f'{atoms.constraints}')
    indices_str = match.group(1)
    indices = eval(indices_str)
    tags[indices] = 0
    atoms.set_tags(tags)
    return atoms

def get_absorbate(adsorbate_smiles):
    adsorbate_path = Path(ocdata.__file__).parent / Path('databases/pkls/adsorbates.pkl')
    adsorbate = Adsorbate(adsorbate_smiles_from_db=adsorbate_smiles, adsorbate_db_path = adsorbate_path)
    return adsorbate


def get_adslabs(adsorbate, bulk, specific_millers, num_sites = 50):
    slabs = Slab.from_bulk_get_specific_millers(bulk = bulk, specific_millers=specific_millers)    
    # Perform heuristic placements
    heuristic_adslabs = AdsorbateSlabConfig(slabs[0], adsorbate, mode="heuristic")
    # Perform random placements
    # (for AdsorbML we use `num_sites = 100` but we will use 4 for brevity here)
    random_adslabs = AdsorbateSlabConfig(slabs[0], adsorbate, mode="random_site_heuristic_placement", num_sites = num_sites)
    adslabs = [*heuristic_adslabs.atoms_list, *random_adslabs.atoms_list]
    return adslabs

def get_atoms_tags_from_json(filename_path,index_tags_path):
    file_name_without_extension = get_filename_from_path(filename_path)
    with open(index_tags_path, 'r') as file:
        json_data = json.load(file)
    atoms = 1
    for key, value in json_data.items():
        if file_name_without_extension in key:
            atoms = read("%s" % filename_path)
            indices = np.array(value)
            atoms.set_tags(indices)    
    if  atoms !=1  :
        pass
    else:
        print('Can not read atoms')
        assert atoms != 1
    return atoms 

def find_files_with_extension(directory, extension):
    found_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                found_files.append(os.path.join(root, file))
    if found_files:
        pass
        #print("Found the following files:")
    else:
        print(f"No files with {extension_to_find} extension found.")
    return found_files


def get_filename_from_path(file_path):
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    string_without_dot = re.sub(r'\.+$', '', file_name)
    return string_without_dot    


def get_relax_db(db_path, new_db_path):
    if os.path.exists(new_db_path):
        raise Exception('{db} exists. Please delete it before proceeding.')
    db = connect(db_path)
    names = []
    for row in db.select():
        names.append(row.name)
    names = list(OrderedDict.fromkeys(names))
    ids = []
    for name in names:
        one_ids = []
        for row in db.select(name = name):
            one_ids.append(row.id)
        ids.append(one_ids[-1])
    ids.sort()
    print(ids)
    with connect(new_db_path) as new_db:
        for _id in ids:
            row = db.get(id=int(_id))
            new_db.write(row.toatoms(), name = row.name)
    return Path(new_db_path).absolute()


def get_min_id_form_asedb(db,name):
    ids = []
    for row in db.select(name = name):
        ids.append(row.id)
    id = ids[-1]
    return id


def train_test_val_before_split(ase_db, ttv=(0.8, 0.1, .1), files=('train.db', 'test.db', 'val.db'), seed=42, before=False):
    for db in files:
        if os.path.exists(db):
            raise Exception('{db} exists. Please delete it before proceeding.')        
    src = connect(ase_db)
    N = src.count()
    ttv = np.array(ttv)
    ttv /= ttv.sum()
    train_end = int(N * ttv[0])
    test_end = train_end + int(N * ttv[1])
    ids = np.arange(1, N + 1)
    rng = np.random.default_rng(seed=42)
    if before == False:
        rng.shuffle(ids)
        train_ids = ids[0:train_end]
        test_id = ids[train_end:test_end]
        val_id = ids[test_end:]
    if before == True:
        train_ids = ids[0:train_end]
        test_id = ids[train_end:test_end]
        val_id = ids[test_end:]
        rng.shuffle(train_ids)
        rng.shuffle(test_id)
        rng.shuffle(val_id)
    with connect(files[0]) as train:
        for _id in train_ids:
            row = src.get(id=int(_id))
            train.write(row.toatoms(), name=row.name)
    with connect(files[1]) as test:
        for _id in test_id:
            row = src.get(id=int(_id))
            test.write(row.toatoms(), name=row.name)
    with connect(files[2]) as val:
        for _id in val_id:
            row = src.get(id=int(_id))
            val.write(row.toatoms(), name=row.name)    

    return [Path(f).absolute() for f in files]

#打乱a2g,sid,fid
def shuft_a2g_sid_fid(a2g,sid,fid):
    random_indices = np.random.permutation(len(a2g))
    new_a2g = []
    new_sid = []
    new_fid = []
    for i in random_indices:
        new_a2g.append(a2g[i])
        new_sid.append(sid[i])
        new_fid.append(fid[i])
    return new_a2g , new_sid, new_fid


def write_lmdb(lmdb_path, data_a2g, sid_set, fid_set):
    memory_size = sys.getsizeof(data_a2g)
    size = memory_size * 0.02

    lmdb1 = lmdb.open(
        "%s" % lmdb_path,
        map_size=1024 * 1024 * size,
        subdir=False,
        meminit=False,
        map_async=True,)

    for id, data in tqdm(enumerate(data_a2g), total=len(data_a2g)):
        data.sid = torch.LongTensor([sid_set[id]])
        data.fid = torch.LongTensor([fid_set[id]])  
        # Filter data if necessary
        # OCP filters adsorption energies > |10| eV and forces > |50| eV/A
        # no neighbor edge case check
        if data.edge_index.shape[1] == 0:
            print("no neighbors", traj_path)
            continue
        txn = lmdb1.begin(write=True)
        txn.put(f"{id}".encode("ascii"), pickle.dumps(data, protocol=-1))
        txn.commit()

    txn = lmdb1.begin(write=True)
    txn.put(f"length".encode("ascii"), pickle.dumps(len(data_a2g), protocol=-1))
    txn.commit()

    lmdb1.sync()
    lmdb1.close()

    return lmdb_path


def split_array(array, split):
    a = []
    number = split[0]+split[1]+split[2]
    num = len(array)//number+1

    if split[2] != 0:
        array1 = array[:-((split[1]+split[2])*num)]
        array2 = array[-((split[1]+split[2])*num):-(split[2]*num)]
        array3 = array[-(split[2]*num):]
        a.append(array1)
        a.append(array2)
        a.append(array3)
    else:
        array1 = array[:-(split[1]*num)]
        array2 = array[-((split[1]+split[2])*num):]
        array3 = []
        a.append(array1)
        a.append(array2)
        a.append(array3)
    return a

def get_sid_fid_from_db(db):
    names = []
    for row in db.select():
        names.append(row.name)
    names_list = list(OrderedDict.fromkeys(names))
    sid_set = []
    for j in names:
        for sid,l in enumerate(names_list):
            if j == l :
                sid_set.append(sid)
    n = 0
    a = 0
    fid_set=[]
    for id in range(len(sid_set)):
        if n == sid_set[id]:
            a += 1
        else:
            a = 1
        n = sid_set[id]
        fid_set.append(a-1)
    return sid_set ,fid_set

def get_ads_db(db_DFT_path, db_slab_path, E_ads_db_path):
    if os.path.exists(E_ads_db_path):
        raise Exception('{db} exists. Please delete it before proceeding.')
    E_mol = {'H_':-3.39414 ,'CO_':-14.805038 ,'CHO_':-16.777619 ,'COH_':-15.071628,
    'COCHO_': -32.688394,'COCOH_':-31.971514, 'COHCHO_':-35.824418,'CHOCHO_':-37.426760, 'COHCOH_':-36.033795,
    'CH2_':-11.534587 ,'CH3_':-17.556453,'CH2O_':-22.299941  ,
    'H2O_':-14.538906, 'CHOH_':-20.179686, 'CH2OH_':-24.64586,'CH4_':-24.015708, 'OCH3_':-24.145187,
    'HCOO_':-24.035638, 'COOH_':-24.214462  ,
    'CH_':-5.675593, 'HCOOH_':-30.252904,
    'O_' : -0.0763,
    'C_' : -0.023723, 'COO_': -23.100434,'CH3OH_':-30.415231, 'OH_':-7.292683,'O-CH3_':-24.145187,'CH3-OH_':-24.145187}

    db_DFT = connect(db_DFT_path)
    db_slab = connect(db_slab_path)
    
    slab_names_energys = {}
    for slab_row in db_slab.select():
        slab_names_energys[slab_row.name] = slab_row.toatoms().get_potential_energy()

    with connect(E_ads_db_path) as E_ads_db:
        for DFT_row in db_DFT.select():
            DFT_names = DFT_row.name
            print(DFT_names)
            DFT_atoms = DFT_row.toatoms()
            energy_dft = DFT_atoms.get_potential_energy()

            for slab_name_,slab_energy_  in slab_names_energys.items():
                if slab_name_ in DFT_names:
                    energy_slab = slab_energy_

            for mol_name_, mol_energy_ in E_mol.items():
                if DFT_names.split("_")[2]+"_" == mol_name_:
                    energy_mol = mol_energy_
            if energy_mol is None:
                print(DFT_names.split("_")[2]+"_")

            energy_ads = energy_dft - energy_slab - energy_mol
    
            energy_mol = None   
            energy_slab = None 
            energy_dft = None
            if energy_ads > 10:
                continue
            calc = SinglePointCalculator(atoms=DFT_atoms, energy=energy_ads, forces =DFT_atoms.get_forces())
            DFT_atoms.calc = calc
            E_ads_db.write(DFT_atoms, name = DFT_names)
            
    return E_ads_db_path

#


#