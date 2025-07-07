import importlib.util
import os
import sys
import numpy as np
import math
from pathlib import Path
from typing import Dict, Tuple, Union

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

# vdw radii: from UFF
# For large conjugated systems, it's beneficial to divide the hydrogen
# radius by 2. For other systems it's generally bad to do so
vdw_radii = np.array([0.0,2.886,2.362,2.451,2.745,4.083,3.851,3.66,3.5,3.364,3.243,2.983,\
    3.021,4.499,4.295,4.147,4.035,3.947,3.868,3.812,3.399,3.295,3.175,3.144,\
    3.023,2.961,2.912,2.872,2.834,3.495,2.763,4.383,4.28,4.23,4.205,4.189,\
    4.141,4.114,3.641,3.345,3.124,3.165,3.052,2.998,2.963,2.929,2.899,3.148,\
    2.848,4.463,4.392,4.42,4.47,4.5,4.404,4.517,3.703,3.522,3.556,3.606,\
    3.575,3.547,3.52,3.493,3.368,3.451,3.428,3.409,3.391,3.374,3.355,3.64,\
    3.141,3.17,3.069,2.954,3.12,2.84,2.754,3.293,2.705,4.347,4.297,4.37,\
    4.709,4.75,4.765,4.9,3.677,3.478,3.396,3.424,3.395,3.424,3.424,3.381,\
    3.326,3.339,3.313,3.299,3.286,3.274,3.248,3.236])/2.0*angstrom2bohr

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

def mominertia(xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Input: xyz (np.ndarray, dimensions (natom,3), coordinates in Bohr)
    Output: I (np.array, dimension (3), moments of inertia assuming
               all atomic masses are 1, in Bohr^2)
            ax (np.ndarray, dimensions (3,3), eigenvectors of moment
                of inertia tensor)
            barycen(np.array, dimension (3), barycenter in Bohr,
                    assuming all atomic masses are 1)
    '''
    N = xyz.shape[0]

    # Determine barycenter
    barycen = np.sum(xyz, axis=0)/N

    # Moment of inertia tensor
    Imat0 = 0.
    for k in range(N):
        vec = xyz[k,:] - barycen
        Imat0 += np.dot(vec,vec)
    Imat = Imat0*np.eye(3)
    for i in range(3):
        for j in range(3):
            for k in range(N):
                Imat[i,j] -= (xyz[k,i]-barycen[i])*(xyz[k,j]-barycen[j])

    I, ax = np.linalg.eig(Imat)
    return I, ax, barycen

def isLinear(xyz: np.ndarray, thresh: float = 1e-4) -> bool:
    '''
    Input: xyz (np.ndarray, dimensions (natom,3), coordinates in Bohr)
           thresh (float, the threshold for the smallest eigenvalue of
                   the moment of inertia tensor)
    Output: whether the molecule is linear
    '''
    I = momintertia(xyz)
    return (min(I)<thresh)

def vecTransRot(xyz: np.ndarray, thresh_lin: float = 1e-4) -> Tuple[np.ndarray, int]:
    '''
    Return the projection vectors for the translational and rotational degrees
    of freedom.
    Note that the masses of all atoms are treated as the same. Therefore, the
    projection vectors for rotation are not the same as the rotational modes
    as would be given by a vibrational analysis.
    Input: xyz (np.ndarray, dimensions (natom,3), coordinates in Bohr)
           thresh_lin (float, the threshold for the smallest eigenvalue of
                       the moment of inertia tensor; used for determining
                       whether the molecule is linear)
    Output: P (np.ndarray, dimensions (3*natom, 6), the projection vectors;
               when the molecule is linear, P[:,5] are zero)
            Ntr (number of translations and rotations, 5 or 6)
    '''
    I, ax, barycen = mominertia(xyz)

    N = xyz.shape[0]
    P = np.zeros([3*N,6])
    # Translations
    for i in range(N):
        P[3*i,0] = 1.0/math.sqrt(N)
        P[3*i+1,1] = 1.0/math.sqrt(N)
        P[3*i+2,2] = 1.0/math.sqrt(N)
    # Rotations
    Ntr = 3
    for j in range(3):
        if I[j] < thresh_lin:
            continue
        for i in range(N):
            P[3*i:3*i+3,Ntr] = np.cross(ax[:,j],xyz[i,:]-barycen)
        P[:,Ntr] /= np.linalg.norm(P[:,Ntr])
        Ntr += 1

    return P, Ntr

def symmetricBreathing(xyz: np.ndarray) -> np.ndarray:
    '''
    Return the vibrational modes corresponding to the symmetric vibration of
    the whole molecule, i.e. where the shape of the molecule is unchanged and
    the molecule merely changes its size.
    Input: xyz (np.ndarray, dimensions (natom,3), coordinates in Bohr)
    Output: P (np.array, dimension (3*natom), the mode)
    '''
    _, __, barycen = mominertia(xyz)
    N = xyz.shape[0]
    P = np.zeros(3*N)
    for i in range(N):
        P[3*i:3*i+3] = xyz[i,:]-barycen
    P /= np.linalg.norm(P)
    return P

def rotationGradient(xyz: np.ndarray,
                     g0: np.ndarray,
                     Nrot: int,
                     ) -> np.ndarray:
    '''
    When a molecule is not at its equilibrium geometry, perturbing the Cartesian
    coordinates along the rotational directions can result in a non-zero
    second-order change of the energy, contrary to when the molecule is at its
    equilibrium geometry.

    This function calculates the gradient change when the molecule is perturbed
    by an infinitesimal step length dx along the rotational axes, divided by dx.

    Input: xyz (np.ndarray, dimensions (natom,3), coordinates in Bohr;
                the caller must guarantee that there are at least 2 atoms)
           g0 (np.array, dimension (3*natom), gradients at equil. geom.)
           Nrot (number of rotational degrees of freedom, can only be 2 or 3)
    Output: g (np.ndarray, dimensions (3*natom,Nrot), the gradient change)
    '''
    N = xyz.shape[0]

    # generate the rotational axes
    I, ax, barycen = mominertia(xyz)
    if Nrot==2:
        ax = ax[:,I!=min(I)] # in this case the minimum I is guaranteed to be non-degenerate

    # loop over the rotational axes
    Prot = np.zeros([3*N,Nrot])
    g = np.zeros([3*N,Nrot])
    for j in range(Nrot):
        for i in range(N):
            Prot[3*i:3*i+3,j] = np.cross(ax[:,j],xyz[i,:]-barycen)
        Prot_norm = np.linalg.norm(Prot[:,j])

        for i in range(N):
            g[3*i:3*i+3,j] = np.cross(ax[:,j],g0[3*i:3*i+3])/Prot_norm

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

def getAbsPath(path: Union[str, Path]) -> Path:
    """parse path, expand `~`, make absolute, return normpath"""
    return Path(os.path.normpath(os.path.abspath(os.path.expanduser(path))))


if __name__ == "__main__":
    # test

    # geometry: H2O2
    xyz = np.array([[0.,2.,0.],[0.,0.,0.],[0.,0.,3.],[2.,0.,3.]])
    atoms = np.array([1,8,8,1])

    print(covalent_radii[atoms[2]]) # 1.1905273728047217
    print(bond(xyz,1,2)) # 3.0
    print(angle(xyz,0,1,2)) # 1.5707963267948966
    print(dihedral(xyz,0,1,2,3)) # -1.5707963267948966

    [I,ax,barycen] = mominertia(xyz)
    print(I) # [ 3.5755711 13.        13.4244289]
    print(ax)
    # [[ 3.50830058e-01 -7.07106781e-01 -6.13936699e-01]
    # [-3.50830058e-01 -7.07106781e-01  6.13936699e-01]
    # [ 8.68237606e-01  3.29961394e-15  4.96148626e-01]]
    print(barycen) # [0.5 0.5 1.5]

    print(vecTransRot(xyz))
    # (array([[ 5.00000000e-01,  0.00000000e+00,  0.00000000e+00,
    #    -4.10441542e-01,  2.94174203e-01, -4.54464235e-01],
    #   [ 0.00000000e+00,  5.00000000e-01,  0.00000000e+00,
    #     4.87203995e-02, -2.94174203e-01, -3.19050136e-01],
    #...
    #     1.85534247e-01,  3.92232270e-01, -1.67562058e-01]]), 6)


    print(getConfig("BDF"))
    # print(getConfig("BDF", "BDf")) # error test
    print(getAbsPath("~/abc/dfs"))
    print(getAbsPath("../abc/dfs"))
