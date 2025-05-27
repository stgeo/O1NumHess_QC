import os
import shutil
from textwrap import dedent
import unittest
import numpy as np
from pathlib import Path
from O1NumHess_QC import O1NumHess_QC


class TestBDF(unittest.TestCase):
    def test_unitConvert(self):
        self.assertAlmostEqual(O1NumHess_QC.bohr2angstrom * O1NumHess_QC.angstrom2bohr, 1)

    def test_readAndWriteXYZ(self):
        path = Path("testR&W.xyz")
        xyz_bohr = np.random.random((12, 3))
        atoms = ["C", "C", "C", "C", "C", "C", "H", "H", "H", "H", "H", "H"]
        # write in Bohr, read in Angstrom and been converted to bohr, manually convert unit
        O1NumHess_QC._writeXYZ(xyz_bohr, atoms, path, useBohr=True)
        _, xyz_bohr_, _atoms = O1NumHess_QC._readXYZ(path, unit="angstrom")
        np.testing.assert_array_almost_equal(xyz_bohr_ * O1NumHess_QC.bohr2angstrom, xyz_bohr)
        self.assertListEqual(atoms, list(_atoms))
        os.remove(path)

    def test_BDF(self):
        cur_dir = Path(".").absolute()
        test_dir = Path("./test_BDF").absolute()
        os.makedirs(test_dir, exist_ok=True)

        test_xyz = Path("test.xyz").absolute()
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
        test_inp = Path("test.inp").absolute()
        test_inp.write_text(dedent(f"""
            $COMPASS
            Title
            benzene grad
            Basis
            cc-pvdz
            Geometry
            file={test_xyz.name}
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

if __name__ == "__main__":
    unittest.main()
