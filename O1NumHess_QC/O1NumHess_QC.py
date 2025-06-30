import numpy as np

from textwrap import dedent
from typing import List, Sequence, Tuple, Union
from pathlib import Path
import os
import re

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
            raise ValueError(f"unit must be 'angstrom' or 'bohr' (case insensitive), '{unit}' is given")
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
        # Check if the file contains **only one** molecule
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

    @staticmethod
    def _readEgrad1(egrad1_path: Union[str, Path]) -> Tuple[np.float, np.ndarray]:
        """
        read energy and gradient from BDF output .egrad1 file.

        TODO 备注：BDF的输出单位
        """
        egrad1_path = Path(egrad1_path).absolute()
        if not egrad1_path.exists():
            raise FileNotFoundError(f"BDF output .egrad1 file: {egrad1_path} not found, error may occurred during calculating")
        lines = [line.strip() for line in egrad1_path.read_text().splitlines() if line.strip()]

        try:
            energy = np.float(lines[0].split()[-1])
            grad: np.ndarray = np.array([[np.float(s) for s in line.split()[1:]] for line in lines[2:]])
            assert grad.shape == (len(lines)-2, 3)
            return energy, grad
        except (AssertionError, IndexError, ValueError):
            raise ValueError(f"Could not parse BDF output .egrad1 file: {egrad1_path}")

    @staticmethod
    def _readEngrad(engrad_path: Union[str, Path]) ->  Tuple[np.float, np.ndarray]:
        """TODO output dimension is 1"""
        engrad_path = Path(engrad_path).absolute()
        if not engrad_path.exists():
            raise FileNotFoundError(f"ORCA output .engrad file: {engrad_path} not found, error may occurred during calculating")
        lines = engrad_path.read_text().splitlines()

        try:
            is_eng = False
            for line in lines:
                if is_eng and "#" not in line:
                    eng = float(line)
                    break
                if "energy".casefold() in line.casefold():
                    is_eng = True

            is_grad = False
            _grad = []
            for line in lines:
                if is_grad and "the".casefold() in line.casefold():
                    is_grad = False
                if is_grad and "#" not in line:
                    _grad.append(float(line))
                if "gradient".casefold() in line.casefold():
                    is_grad = True
            grad: np.ndarray = np.array(_grad)
            print(grad)
            return eng, grad
        except NameError:
            raise ValueError(f"Could not parse ORCA output .engrad file: {engrad_path}")

    def _O1NH(
        self,
        grad_func: (...),
        method: str,
        delta: float,
        core: int,
        total_cores: Union[int, None],
        **kwargs_for_grad_func,
    ) -> np.ndarray:
        """
        interface with O1NH
        """
        # ========== initialize O1NH
        o1nh = O1NumHess(
            x=self.xyz_bohr.reshape((self.xyz_bohr.size,)),
            grad_func=grad_func,
            **kwargs_for_grad_func
        )
        # ========== use o1nh to calculate hessian
        if method.casefold() == "single".casefold():
            self.hessian = o1nh.singleSide(delta=delta, core=core, total_cores=total_cores)
        elif method.casefold() == "double".casefold():
            self.hessian = o1nh.doubleSide(delta=delta, core=core, total_cores=total_cores)
        else:
            raise ValueError(f"method {method} is not supported, only supported 'single' and 'double'")
        return self.hessian

    def calcHessian_BDF(
        self,
        method: str,
        delta: float,
        core: int,
        mem: str,
        total_cores: Union[int, None] = None,
        inp: Union[str, Path] = ...,
        encoding: str = "utf-8",
        tempdir: Union[Path, str] = "~/tmp",
        task_name: str = "",
        config_name: str = "",
    ) -> np.ndarray:
        """
        TODO 备注：单位直接从inp文件中读取，无需传入
        """
        # ========== check params
        _ = getConfig("BDF", config_name)

        inp = Path(inp).absolute()
        if not inp.is_file():
            raise FileNotFoundError(f"input .inp file: {inp} not exists or not a file")

        tempdir = Path(tempdir)
        if str(tempdir).startswith("~"):
            tempdir = tempdir.expanduser()
        tempdir = tempdir.absolute()
        if not tempdir.exists():
            os.makedirs(tempdir, exist_ok=True)

        task_name = task_name if task_name else inp.stem

        # ========== interface with O1NH
        return self._O1NH(
            grad_func=self._calcGrad_BDF,
            method=method,
            delta=delta,
            core=core,
            total_cores=total_cores,
            **{
                "mem": mem,
                "inp": inp,
                "encoding": encoding,
                "tempdir": tempdir,
                "task_name": task_name,
                "config_name": config_name,
            }
        )

    def _calcGrad_BDF(
        self,
        x_bohr: np.ndarray,
        index: int,
        core: int,
        mem: str, # "1G"
        inp: Union[str, Path], # TODO 备注：inp文件内部必须在“Geometry”块中包含file=xxx.xyz的写法，否则报错
        encoding: str = "utf-8",
        tempdir: Union[str, Path] = "~/tmp",
        task_name: str = "", # TODO 备注：BDF的输入文件名（因为BDF的运行需要xyz和inp文件，但是两个文件的文件名可能不一致），如果未设置统一以inp的文件名做为任务名
        config_name: str = "", # TODO 备注：BDF的配置文件名，如果不写，则以第一个配置文件为准，如果没有配置文件，报错
    ) -> np.ndarray:
        """
        调用BDF计算一次梯度并读取结果

        给定扰动后的x，（采用bohr作为单位），生成新的坐标XYZ文件
        给定BDF计算的参数文件inp，（不要求在当前工作路径下），从中读取计算时采用的单位和参数，生成新的配置文件
        生成调用BDF的.sh文件，并执行，执行完毕后读取.egrad1文件提取梯度

        output to specified folder (not supported by BDF now) 受BDF限制，输出路径只能在当前文件夹
        输出的梯度单位一定是Bohr，输出的形状是一维向量
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

        task_name = task_name if task_name else inp.stem

        # ========== make sure the input x is valid
        x_bohr = np.array(x_bohr)
        assert x_bohr.size == self.xyz_bohr.size, f"the input size of x is {x_bohr.size}, different with the initial molecular size {self.xyz_bohr.size}"

        # ========== generate filename for BDF files
        # print(Path("."))
        suffix = str(index).zfill(len(str(x_bohr.size * 2))) # use index and x.size to generate a suffix with proper length
        task_name = f"{task_name}_{suffix}"
        tempdir = tempdir / task_name
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
        self._writeXYZ(x_bohr.reshape(x_bohr.size // 3, 3), self.atoms, xyz_out_path, useBohr)

        # ========== generate new .sh file to run BDF
        sh_out_path.write_text(config["bash"] + \
            dedent(f"""
            export OMP_NUM_THREADS={core}
            export OMP_STACKSIZE={mem}

            rm -rf {tempdir}
            {config["path"]} -tmpdir {tempdir} -r {inp_out_path.name} > {task_name}.out
            rm -rf .{task_name}.wrk
            """
        ), "utf-8")

        # ========== calculate
        os.system(f"bash {sh_out_path}")

        # ========== read result
        _, grad = self._readEgrad1(egrad1_in_path)
        assert grad.shape == self.xyz_bohr.shape, f"the grad shape from BDF output .egrad1 file: {egrad1_in_path} is {grad.shape}, different with the initial molecular shape {self.xyz_bohr.shape}"
        return grad.reshape((self.xyz_bohr.size,))

    def calcHessian_ORCA(
        self,
        method: str,
        delta: float,
        total_cores: Union[int, None] = None,
        inp: Union[str, Path] = ...,
        encoding: str = "utf-8",
        tempdir: Union[Path, str] = "~/tmp",
        task_name: str = "",
        config_name: str = "",
    ) -> np.ndarray:
        """
        并行的core参数写在inp文件里
        """
        # ========== check params
        _ = getConfig("ORCA", config_name)

        inp = Path(inp).absolute()
        if not inp.is_file():
            raise FileNotFoundError(f"input .inp file: {inp} not exists or not a file")

        tempdir = Path(tempdir)
        if str(tempdir).startswith("~"):
            tempdir = tempdir.expanduser()
        tempdir = tempdir.absolute()
        if not tempdir.exists():
            os.makedirs(tempdir, exist_ok=True)

        task_name = task_name if task_name else inp.stem

        inp_str = inp.read_text(encoding=encoding)
        # ========== get "core"
        match = re.search(r"^\s*!.*?PAL(\d+)|^\s*%\s*pal\s*nprocs\s*(\d+)", inp_str, re.MULTILINE | re.IGNORECASE)
        if match is None:
            raise ValueError(f"inp file {inp} does not contain parallel information like 'PAL' or '%pal nprocs'")
        core = int(match.group(1) if match.group(1) else match.group(2))

        # ========== make sure to calculate gradient in ORCA
        if re.search(r"^\s*!.*?EnGrad", inp_str, re.MULTILINE | re.IGNORECASE) is None:
            raise ValueError(f"inp file {inp} does not contain parameter 'EnGrad', cannot calculate gradient in ORCA")

        # ========== interface with O1NH
        return self._O1NH(
            grad_func=self._calcGrad_ORCA,
            method=method,
            delta=delta,
            core=core,
            total_cores=total_cores,
            **{
                "inp": inp,
                "encoding": encoding,
                "tempdir": tempdir,
                "task_name": task_name,
                "config_name": config_name,
            }
        )

    def _calcGrad_ORCA(
        self,
        x_bohr: np.ndarray,
        index: int,
        core: int,
        inp: Union[str, Path],
        encoding: str = "utf-8",
        tempdir: Union[str, Path] = "~/tmp",
        task_name: str = "",
        config_name: str = "",
    ) -> np.ndarray:
        # ========== check params
        config = getConfig("ORCA", config_name)
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

        task_name = task_name if task_name else inp.stem

        # ========== make sure the input x is valid
        x_bohr = np.array(x_bohr)
        assert x_bohr.size == self.xyz_bohr.size, f"the input size of x is {x_bohr.size}, different with the initial molecular size {self.xyz_bohr.size}"

        # ========== generate filename and path for ORCA files
        # print(Path("."))
        cwd = Path(".").absolute()
        suffix = str(index).zfill(len(str(x_bohr.size * 2))) # use index and x.size to generate a suffix with proper length
        task_name = f"{task_name}_{suffix}"
        tempdir = tempdir / task_name
        xyz_out_path = Path(f"{task_name}.xyz").absolute()
        inp_out_path = Path(f"{task_name}.inp").absolute()
        sh_out_path = Path(f"{task_name}.sh").absolute()
        engrad_in_path = Path(f"{task_name}.engrad").absolute()

        # ========== generate new .inp file for ORCA
        inp_str = inp.read_text(encoding)

        # make sure to calculate gradient in ORCA
        if re.search(r"^\s*!.*?EnGrad", inp_str, re.MULTILINE | re.IGNORECASE) is None:
            raise ValueError(f"inp file {inp} does not contain parameter 'EnGrad', cannot calculate gradient in ORCA")

        # find "pal" and replace with current core
        match = re.search(r"^\s*!.*?PAL(\d+)|^\s*%\s*pal\s*nprocs\s*(\d+)", inp_str, re.MULTILINE | re.IGNORECASE)
        if match is None:
            raise ValueError(f"inp file {inp} does not contain parallel information like 'PAL' or '%pal nprocs'")
        pal1 = r"(^\s*!.*?PAL)(\d+)"
        pal2 = r"(^\s*%\s*pal\s*nprocs\s*)(\d+)"
        inp_str = re.sub(pal1, rf"\g<1>{core}", inp_str, flags=re.MULTILINE | re.IGNORECASE) # replace PAL
        inp_str = re.sub(pal2, rf"\g<1>{core}", inp_str, flags=re.MULTILINE | re.IGNORECASE) # replace nprocs
        # print(inp_str)

        # find "unit"
        useBohr = True if re.search(r"^\s*!.*?Bohrs", inp_str, re.MULTILINE | re.IGNORECASE) else False

        # find and replace the .xyz file line
        if re.search(r"(^\s*\*\s*xyzfile\s*\d+\s*\d+\s*)(.+\.xyz)", inp_str, re.MULTILINE | re.IGNORECASE) is None:
            raise ValueError(f"inp file {inp} does not contain molecular coordinate information, coordinates must be specified by xyzfile format")
        inp_str = re.sub(r"(^\s*\*\s*xyzfile\s*\d+\s*\d+\s*)(.+\.xyz)", rf"\g<1>{xyz_out_path.name}", inp_str, flags=re.MULTILINE | re.IGNORECASE)
        # print(inp_str)

        # output file
        inp_out_path.write_text(inp_str, encoding=encoding)

        # ========== generate new .xyz file for ORCA
        self._writeXYZ(x_bohr.reshape(x_bohr.size // 3, 3), self.atoms, xyz_out_path, useBohr)

        print(f"{tempdir}")
        # ========== generate new .sh file to run ORCA
        sh_out_path.write_text(config["bash"] + \
            dedent(f"""
            mkdir {tempdir}
            cp {inp_out_path} {tempdir}
            cp {xyz_out_path} {tempdir}

            cd {tempdir}
            {config["path"]} {inp_out_path.name} >& {task_name}.out

            rm -f *.inp *.xyz *.tmp*
            cp * {cwd}

            rm -rf {tempdir}
            """
        ), "utf-8")
        # cd to tempdir, calculate, del non-result file, copy the rest result file back

        # ========== calculate
        os.system(f"bash {sh_out_path}")

        # ========== read result
        _, grad = self._readEngrad(engrad_in_path)
        assert grad.size == self.xyz_bohr.size, f"the grad size from ORCA output .engrad file: {engrad_in_path} is {grad.size}, different with the initial molecular shape {self.xyz_bohr.size}"
        return grad.reshape((self.xyz_bohr.size,))

