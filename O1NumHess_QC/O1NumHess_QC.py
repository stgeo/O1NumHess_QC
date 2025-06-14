import numpy as np

from textwrap import dedent
from typing import List, Sequence, Tuple, Union
from pathlib import Path
import os

from O1NumHess import O1NumHess
from .utils import getConfig

class O1NumHess_QC:
    bohr2angstrom = 0.529177249
    angstrom2bohr = 1.8897259886

    def __init__(
        self,
        xyz_path: Union[str, Path],
        unit: str = "angstrom",
        encoding: str = "utf-8",
    ):
        # read the XYZ file, get path, coordinates and atoms
        self.xyz_path, self.xyz_bohr, self.atoms = self._readXYZ(xyz_path, encoding, unit)

    @property
    def xyz_angstrom(self) -> np.ndarray:
        return self.xyz_bohr * O1NumHess_QC.bohr2angstrom

    @staticmethod
    def _readXYZ(
        path: Union[str, Path],
        encoding: str = "utf-8",
        unit: str = "angstrom",
    ) -> Tuple[Path, np.ndarray, Tuple[str, ...]]:
        """
        Read atomic coordinates from an XYZ file into a numpy array.

        Return filepath, coordinates, atoms. Coordinates are in Bohr!

        There must be **only one** molecule in the file!

        XYZ format:
            <number of atoms>
            <comment (can be empty)>
            <symbol> <x> <y> <z>
            <symbol> <x> <y> <z>
            ...
        """
        # unit
        if unit.casefold() not in ["angstrom".casefold(), "bohr".casefold()]:
            raise ValueError(f"unit must be 'angstrom' or 'bohr' (case insensitive)")
        isBohr = False
        if unit.casefold() == "bohr".casefold():
            isBohr = True

        # handle path
        path = Path(path).absolute() # make path absolute
        if not path.is_file():
            raise FileNotFoundError(f"input XYZ file {path} not exists or not a file")
        # read file, remove empty lines and strip whitespace, keep first 2 lines (second line may be empty)
        lines = [line.strip() for i, line in enumerate(path.read_text(encoding).splitlines()) if i < 2 or line.strip()]

        # parse number of atoms from the first line
        try:
            n_atoms = int(lines[0])
        except ValueError:
            raise ValueError("First line of XYZ file must contain the number of atoms")
        # Check if the file contains only one molecule
        # TODO 备注：确保只有一个分子，多了报错
        assert len(lines) == n_atoms + 2, "lines not match, XYZ file is incorrect or contains more than one molecule"

        # Extract coordinates (skip the first two lines)
        try:
            coordinates: np.ndarray = np.array([[np.float(s) for s in line.split()[1:]] for line in lines[2:]])
            atoms = tuple(line.split()[0] for line in lines[2:]) # use tuple to make sure it is not editable
        except (IndexError, ValueError):
            raise ValueError("Could not parse XYZ file, the file may be incorrect")
        assert coordinates.shape == (n_atoms, 3), f"the coordinates shape {coordinates.shape} is incorrect"

        if not isBohr:
            coordinates = coordinates * O1NumHess_QC.angstrom2bohr
        return path, coordinates, atoms

    @staticmethod
    def _readEgrad1(egrad1_path: Union[str, Path], encoding: str = "utf-8") -> Tuple[np.float, np.ndarray]:
        """
        read energy and gradient from BDF output .egrad1 file.

        TODO 备注：补上BDF的输出单位
        """
        egrad1_path = Path(egrad1_path)
        if not egrad1_path.exists():
            raise FileNotFoundError(f"BDF output .egrad1 file: {egrad1_path} not found, error may occurred during calculating")
        lines = [line.strip() for line in egrad1_path.read_text(encoding).splitlines() if line.strip()]

        try:
            energy = np.float(lines[0].split()[-1])
            grad: np.ndarray = np.array([[np.float(s) for s in line.split()[1:]] for line in lines[2:]])
            assert grad.shape == (len(lines)-2, 3)
            return energy, grad
        except (AssertionError, IndexError, ValueError):
            raise ValueError(f"Could not parse BDF output .egrad1 file: {egrad1_path}")

    @staticmethod
    def _writeXYZ(xyz_bohr: np.ndarray, atoms: Sequence[str], path: Path, useBohr: bool=False, comment: str = "", encoding: str = "utf-8"):
        assert xyz_bohr.shape == (xyz_bohr.size // 3, 3) and xyz_bohr.shape[0] == len(atoms)
        xyz_out = xyz_bohr if useBohr else xyz_bohr * O1NumHess_QC.bohr2angstrom
        xyz_str = f"{len(atoms)}\n" + \
            comment.strip() + "\n" + \
            "\n".join([
                f"{atom:<3}{x:>26.13f}{y:>26.13f}{z:>26.13f}"
                for atom, (x, y, z) in zip(atoms, xyz_out)] # type: ignore
            ) + \
            "\n"
        path.write_text(xyz_str, encoding)

    def calcHessian_BDF(
        self,
        method: str,
        delta: float,
        core: int,
        mem: str,
        inp: Union[str, Path],
        # unit: str = "angstrom",
        encoding: str = "utf-8",
        tempdir: Union[Path, str] = "~/tmp",
        task_name: str = "",
        config_name: str = "",
    ) -> np.ndarray:
        """
        TODO 备注：单位直接从inp文件中读取，无需传入
        """
        # ========== check params
        config = getConfig("BDF", config_name)
        assert 0 < core and isinstance(core, int) # <= os.cpu_count()
        inp = Path(inp).absolute()
        if not inp.is_file():
            raise FileNotFoundError(f"input .inp file: {inp} not exists or not a file")
        tempdir = Path(tempdir)
        if str(tempdir).startswith("~"):
            tempdir = tempdir.expanduser()
        tempdir = tempdir.absolute()
        if not tempdir.exists():
            os.makedirs(tempdir, exist_ok=True)

        # ========== initialize o1nh
        x = self.xyz_bohr.reshape((self.xyz_bohr.size,))
        o1nh = O1NumHess(
            x,
            self._calcGrad_BDF,
            mem=mem,
            inp=inp,
            # unit=unit,
            encoding=encoding,
            tempdir=tempdir,
            task_name=task_name,
            config_name=config_name
        )
        # ========== use o1nh to calculate hessian
        if method.casefold() == "single".casefold():
            self.hessian = o1nh.singleSide(core, delta)
        elif method.casefold() == "double".casefold():
            self.hessian = o1nh.doubleSide(core, delta)
        else:
            raise ValueError(f"method {method} is not supported, only supported 'single' and 'double'")
        return self.hessian

    def _calcGrad_BDF(
        self,
        x_bohr: np.ndarray,
        index: int,
        core: int,
        mem: str, # "1G"
        inp: Union[str, Path], # TODO 备注：inp文件内部必须在“Geometry”块中包含file=xxx.xyz的写法，否则报错
        # unit: str = "angstrom", # TODO 备注：单位直接从inp文件读取，无需传入
        encoding: str = "utf-8",
        tempdir: Union[str, Path] = "~/tmp",
        task_name: str = "", # TODO 备注：BDF的输入文件名（因为BDF的运行需要xyz和inp文件，但是两个文件的文件名可能不一致），如果未设置统一以inp的文件名做为任务名
        config_name: str = "", # TODO 备注：BDF的配置文件名，如果不写，则以第一个配置文件为准，如果没有配置文件，报错
    ) -> np.ndarray:
        """
        TODO 备注：给定扰动后的x，（由于输入给o1nh的单位一定是angstrom，所以扰动回来的单位一定是angstrom）
        读取优化后的原始xyz字符串，生成新的坐标XYZ文件
        参数文件，调用BDF计算梯度并读取结果
        给定的输入文件不要求在当前路径下
        TODO 备注：output to specified folder (not supported by BDF now) 受BDF限制，输出路径只能在当前文件夹
        TODO 备注：输出的单位一定是Bohr，输出的形状是一维向量
        """
        # ========== check params
        config = getConfig("BDF", config_name)
        assert 0 < core and isinstance(core, int) # <= os.cpu_count()
        inp = Path(inp).absolute()
        if not inp.is_file():
            raise FileNotFoundError(f"input .inp file: {inp} not exists or not a file")
        tempdir = Path(tempdir)
        if str(tempdir).startswith("~"):
            tempdir = tempdir.expanduser()
        tempdir = tempdir.absolute()
        if not tempdir.exists():
            os.makedirs(tempdir, exist_ok=True)

        # ========== make sure the input x is valid
        x_bohr = np.array(x_bohr)
        assert x_bohr.size == self.xyz_bohr.size, f"the input size of x is {x_bohr.size}, different with the initial molecular size {self.xyz_bohr.size}"

        # ========== generate filename for BDF files
        # print(Path("."))
        suffix = str(index).zfill(len(str(x_bohr.size * 2))) # use index and x.size to generate a suffix with proper length
        task_name = task_name if task_name else inp.stem
        task_name = f"{task_name}_{suffix}"
        xyz_out_path = Path(f"{task_name}.xyz").absolute()
        inp_out_path = Path(f"{task_name}.inp").absolute()
        sh_out_path = Path(f"{task_name}.sh").absolute()
        egrad1_in_path = Path(f"{task_name}.egrad1").absolute()

        # ========== generate new .inp file for BDF
        # read inp file, drop comments and right spaces
        inp_str = inp.read_text(encoding).splitlines()
        inp_str = [line.rstrip() if "#" not in line else line.split("#")[0].rstrip() for line in inp_str if "#" not in line or line.split("#")[0].strip()]
        # print("\n".join(inp_str))

        # find "unit"
        useBohr = False
        for i in range(len(inp_str)-1):
            if (inp_str[i].strip().casefold() == "Unit".casefold() and \
                inp_str[i+1].strip().casefold() == "Bohr".casefold()
            ) or inp_str[i].casefold().find("unit=bohr".casefold()) > -1:
                useBohr = True

        # find the .xyz file line
        file_line = []
        for i in range(1, len(inp_str)-1):
            if (
                inp_str[i-1].strip().casefold() == "Geometry".casefold() and \
                inp_str[i].strip().casefold().startswith("file=".casefold()) and \
                inp_str[i+1].strip().casefold() == "End geometry".casefold() and \
                inp_str[i].strip()[5:] == self.xyz_path.name
                # the filename must be equal with the xyz file
            ):
                file_line.append(i)
        if file_line == []:
            raise ValueError(f"the inp file {inp} does not contain proper Geometry info, it must be written as 'file={self.xyz_path.name}' in 'Geometry' part, no spaces and slash '/'")

        # replace the .xyz file with the new one
        for i in file_line:
            inp_str[i] = inp_str[i].replace(self.xyz_path.name, xyz_out_path.name)
        # print("\n".join(inp_str))

        # output file
        inp_out_path.write_text("\n".join(inp_str), encoding)

        # ========== generate new .xyz file for BDF
        self._writeXYZ(x_bohr.reshape(x_bohr.size // 3, 3), self.atoms, xyz_out_path, useBohr, encoding=encoding)

        # ========== generate new .sh file to run BDF
        sh_out_path.write_text(config["bash"] + \
            dedent(f"""
            export OMP_NUM_THREADS={core}
            export OMP_STACKSIZE={mem}

            rm -rf {Path(tempdir) / task_name}
            {config["path"]} -tmpdir {Path(tempdir) / task_name} -r {inp_out_path.name} > {task_name}.out
            rm -rf .{task_name}.wrk
            """
        ), "utf-8")

        # ========== calculate
        os.system(f"bash {sh_out_path}")

        # ========== read result
        _, grad = self._readEgrad1(egrad1_in_path, encoding)
        assert grad.shape == self.xyz_bohr.shape, f"the grad shape from BDF output .egrad1 file: {egrad1_in_path} is {grad.shape}, different with the initial molecular shape {self.xyz_bohr.shape}"
        return grad.reshape((self.xyz_bohr.size,))
