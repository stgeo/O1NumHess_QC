import importlib.util
import sys
import numpy as np
import math
from pathlib import Path
from typing import Dict

# Useful constants
bohr2angstrom = 0.529177249
angstrom2bohr = 1.0/bohr2angstrom

# Pyykko's radii, elements count from 1
# The radius of oxygen was too low. A value of 0.68 is more appropriate,
# but for simplicity we don't change it
covalent_radii = np.array([0.0,0.32,0.46,1.33,1.02,0.85,0.75,0.71,0.63,0.64,0.67,1.55,1.39,1.26,\
    1.16,1.11,1.03,0.99,0.96,1.96,1.71,1.48,1.36,1.34,1.22,1.19,1.16,1.1,\
    1.11,1.12,1.18,1.24,1.21,1.21,1.16,1.14,1.17,2.1,1.85,1.63,1.54,1.47,1.38,\
    1.28,1.25,1.25,1.2,1.28,1.36,1.42,1.4,1.4,1.36,1.33,1.31,2.32,1.96,1.8,\
    1.63,1.76,1.74,1.73,1.72,1.68,1.69,1.68,1.67,1.66,1.65,1.64,1.7,1.62,\
    1.52,1.46,1.37,1.31,1.29,1.22,1.23,1.24,1.33,1.44,1.44,1.51,1.45,1.47,\
    1.42,2.23,2.01,1.86,1.75,1.69,1.7,1.71,1.72,1.66,1.66,1.68,1.68,1.65,\
    1.67,1.73,1.76,1.61])*angstrom2bohr

periodic_table = ['X', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O',\
    'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr',\
    'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y',\
    'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',\
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm',\
    'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po',\
    'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es',\
    'Fm', 'Md', 'No', 'Lr']

def bond(xyz: np.ndarray, A: int, B: int) -> float:
    '''
    Given coordinates xyz, return bond length between atom A and B
    Input: xyz (np.ndarray, dimensions (natom,3), coordinates in Bohr)
           A (integer, serial number of the first atom)
           B (integer, serial number of the second atom)
    Output: float, bond length in Bohr
    '''
    return np.linalg.norm(xyz[A,:]-xyz[B,:])

def angle(xyz: np.ndarray, A: int, B: int, C: int) -> float:
    '''
    Given coordinates xyz, return A-B-C angle
    Input: xyz (np.ndarray, dimensions (natom,3), coordinates in Bohr)
           A (integer, serial number of the first atom)
           B (integer, serial number of the second atom)
           C (integer, serial number of the third atom)
    Output: float, angle in radians, range [0,pi]
    '''
    BA = xyz[A,:]-xyz[B,:]
    BC = xyz[C,:]-xyz[B,:]
    return math.acos(np.dot(BA,BC)/np.linalg.norm(BA)/np.linalg.norm(BC))

def cosangle(xyz: np.ndarray, A: int, B: int, C: int) -> float:
    '''
    Given coordinates xyz, return the cosine of A-B-C angle
    Input: xyz (np.ndarray, dimensions (natom,3), coordinates in Bohr)
           A (integer, serial number of the first atom)
           B (integer, serial number of the second atom)
           C (integer, serial number of the third atom)
    Output: float, cos angle
    '''
    BA = xyz[A,:]-xyz[B,:]
    BC = xyz[C,:]-xyz[B,:]
    return np.dot(BA,BC)/np.linalg.norm(BA)/np.linalg.norm(BC)

def dihedral(xyz: np.ndarray, A: int, B: int, C: int, D: int) -> float:
    '''
    Given coordinates xyz, return A-B-C-D dihedral
    Input: xyz (np.ndarray, dimensions (natom,3), coordinates in Bohr)
           A (integer, serial number of the first atom)
           B (integer, serial number of the second atom)
           C (integer, serial number of the third atom)
           D (integer, serial number of the fourth atom)
    Output: float, dihedral in radians, range (-pi,pi]
    '''
    AB = xyz[B,:]-xyz[A,:]
    CB = xyz[B,:]-xyz[C,:]
    CD = xyz[D,:]-xyz[C,:]
    normal1 = np.cross(AB,CB)
    normal2 = np.cross(CB,CD)
    dihed = math.acos(np.dot(normal1,normal2)/np.linalg.norm(normal1)/np.linalg.norm(normal2))
    # sign of dihedral
    if np.dot(np.cross(normal1,normal2),CB) > 0.:
        return -dihed
    else:
        return dihed

def getConfig(program: str, config_name: str = "") -> Dict[str, str]:
    config_folder = Path("~/.O1NumHess_QC").expanduser().absolute()

    file = (config_folder / f"{program}_config.py").absolute()
    if not file.exists():
        raise FileNotFoundError(f"the config file of {program}: {file} does not exists, refer to the document")
    module_name = f"{program}_config"
    spec = importlib.util.spec_from_file_location(module_name, file)
    if spec is None:
        raise ImportError(f"something wrong while importing the config file {file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module) # type: ignore (ignore the warning from Pylance)

    # print(module.config)
    try:
        if config_name == "":
            config = module.config[0]
        else:
            for dic in module.config:
                if dic["name"] == config_name:
                    config = dic
    except IndexError:
        raise AttributeError(f"the config file {file} is empty")
    except AttributeError:
        raise AttributeError(f"something wrong with the config file {file}")
    try:
        # TODO 检查config内容是否完整
        return config # type: ignore
    except NameError:
        raise AttributeError(f"the config file {file} does not have the config name: '{config_name}'")


if __name__ == "__main__":
    # test

    # geometry: H2O2
    xyz = np.array([[0.,2.,0.],[0.,0.,0.],[0.,0.,3.],[2.,0.,3.]])
    atoms = np.array([1,8,8,1])

    print(covalent_radii[atoms[2]]) # 1.1905273728047217
    print(bond(xyz,1,2)) # 3.0
    print(angle(xyz,0,1,2)) # 1.5707963267948966
    print(dihedral(xyz,0,1,2,3)) # -1.5707963267948966

    print(getConfig("BDF"))
    print(getConfig("BDF", "BDf")) # error test
