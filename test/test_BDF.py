import os
import shutil
from textwrap import dedent
import unittest
import numpy as np
from pathlib import Path
from O1NumHess_QC import O1NumHess_QC
from O1NumHess_QC.utils import bohr2angstrom, angstrom2bohr


class TestBDF(unittest.TestCase):
    def setUp(self):
        print("\n" + "="*50)
        print(f"Starting test: {self._testMethodName}")
        print("="*50)

    def tearDown(self):
        print("-"*50)
        print(f"Finished test: {self._testMethodName}")
        print("-"*50 + "\n")

    def test_unitConvert(self):
        self.assertAlmostEqual(bohr2angstrom * angstrom2bohr, 1)

    def test_readAndWriteXYZ(self):
        path = Path("_testR&W.xyz")
        xyz_bohr = np.random.random((12, 3))
        atoms = ["C", "C", "C", "C", "C", "C", "H", "H", "H", "H", "H", "H"]
        # write in Bohr, read in Angstrom and been converted to bohr, manually convert unit
        O1NumHess_QC._writeXYZ(xyz_bohr, atoms, path, useBohr=True)
        _, xyz_bohr_, _atoms = O1NumHess_QC._readXYZ(path, unit="angstrom")
        np.testing.assert_array_almost_equal(xyz_bohr_ * bohr2angstrom, xyz_bohr)
        self.assertListEqual(atoms, list(_atoms))
        os.remove(path)

    def test_total_cores(self):
        xyz = self._generateXYZ("_test_total_cores.xyz")
        inp = self._generateInp("_test_total_cores.inp", xyz)
        qc = O1NumHess_QC(xyz)

        with self.assertRaises(ValueError):
            qc.calcHessian_BDF("single", 1e-3, core=2, mem="2G", total_cores=999, inp=inp)
        with self.assertRaises(ValueError):
            qc.calcHessian_BDF("single", 1e-3, core=2, mem="2G", total_cores=-1, inp=inp)
        # with self.assertWarns(RuntimeWarning):
        #     qc.calcHessian_BDF("single", 1e-3, core=2, mem="2G", total_cores=None, inp=inp)
        with self.assertRaises(TypeError):
            qc.calcHessian_BDF("single", 1e-3, core=2, mem="2G", total_cores=1.2, inp=inp) # type: ignore
        os.remove(xyz)
        os.remove(inp)

    def test_BDF(self):
        cur_dir = Path(".").absolute()
        test_dir = Path("./_test_BDF").absolute()
        os.makedirs(test_dir, exist_ok=True)

        test_xyz = self._generateXYZ("test.xyz")
        test_inp = self._generateInp("test.inp", test_xyz)

        try:
            os.chdir(test_dir)
            qc = O1NumHess_QC(test_xyz)
            eng_ = -232.108204353561
            grad_: np.ndarray = np.array([
                [ 0.0042475274,  0.0060127667, -0.0000039839],
                [-0.0031002205,  0.0066626468, -0.0000030526],
                [-0.0073204325,  0.0006716961,  0.0000008478],
                [-0.0042366843, -0.0060243853, -0.0000048821],
                [ 0.0030982019, -0.0066557785,  0.0000061277],
                [ 0.0072992591, -0.0006745076, -0.0000023993],
                [-0.0042945127, -0.0061028118, -0.0000007873],
                [-0.0074275957,  0.0006603016,  0.0000007442],
                [-0.0031308833,  0.0067766069,  0.0000040330],
                [ 0.0043069792,  0.0060984600, -0.0000008635],
                [ 0.0031303714, -0.0067593045,  0.0000035547],
                [ 0.0074279899, -0.0006656904,  0.0000006613]
            ])

            index = 0
            grad = qc._calcGrad_BDF(qc.xyz_bohr, index, 4, "4G", test_inp)
            eng, _ = qc._readEgrad1(test_dir / f"test_{index:02}.egrad1")
            self.assertAlmostEqual(eng_, eng)
            np.testing.assert_array_almost_equal(grad_.reshape((grad_.size,)), grad)
        finally:
            os.chdir(cur_dir)
            os.remove(test_xyz)
            os.remove(test_inp)
            shutil.rmtree(test_dir)

    def _generateInp(self, path, xyz):
        test_inp = Path(path).absolute()
        test_inp.write_text(dedent(f"""
            $COMPASS
            Title
            benzene grad
            Basis
            cc-pvdz
            Geometry
            file={xyz.name}
            End geometry
            $END
            $xuanyuan
            $end
            $scf
            RKS
            dft
            B3LYP
            spinmulti
            1
            $end
            $resp
            geom
            $end
            """).strip()
        )
        return test_inp

    def test_BDF_Hessian(self):
        cur_dir = Path(".").absolute()
        test_dir = Path("./test_BDF").absolute()
        os.makedirs(test_dir, exist_ok=True)

        test_xyz = Path("test.xyz").absolute()
        # To make the test finish quickly, we have to choose a small molecule.
        # This may not capture all possible bugs.
        test_xyz.write_text(dedent("""
            4

            H 0. 1. 0.
            O 0. 0. 0.
            O 0. 0. 1.5
            H 1. 0. 1.5
            """).lstrip()
        )
        test_inp = Path("test.inp").absolute()
        test_inp.write_text(dedent(f"""
            $COMPASS
            Title
            H2O2 grad
            Basis
            sto-3g
            Geometry
            file={test_xyz.name}
            End geometry
            $END

            $xuanyuan
            $end

            $scf
            RHF
            $end

            $resp
            geom
            $end
            """).strip()
        )

        try:
            os.chdir(test_dir)
            qc = O1NumHess_QC(test_xyz)
            hessian = qc.calcHessian_BDF(
                method = "o1numhess",
                delta = 1e-3,
                core = 1,
                mem = "4G",
                inp = test_inp,
                tempdir = "~/tmp",
                config_name = "BDF",
            )
            print(hessian)
        finally:
            os.chdir(cur_dir)
            os.remove(test_xyz)
            os.remove(test_inp)
            shutil.rmtree(test_dir)

    def _generateXYZ(self, path):
        test_xyz = Path(path).absolute()
        test_xyz.write_text(dedent("""
            12

            C         -0.26396        0.29101        0.87006
            C         -0.72047       -0.95180        0.41795
            C          0.12718       -2.06420        0.45530
            C          1.43129       -1.93382        0.94486
            C          1.88770       -0.69106        1.39714
            C          1.04011        0.42134        1.35968
            H         -0.91966        1.15163        0.84107
            H          1.39331        1.38276        1.70952
            H          2.89652       -0.59018        1.77601
            H          2.08713       -2.79433        0.97377
            H         -1.72939       -1.05266        0.03928
            H         -0.22596       -3.02564        0.10546
            """).lstrip()
        )

        return test_xyz

if __name__ == "__main__":
    unittest.main()
