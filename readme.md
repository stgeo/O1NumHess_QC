# O1NumHess_QC

## requirement

* python >= 3.6
* numpy
* O1NumHess

## usage

```bash
python3 setup.py install
```

this will create config files in `~/.O1NumHess_QC`, copy the files `xxx_config_example.py` to `xxx_config.py` and modify it with your own condition.

test files:

* `benzene.xyz`:

    ```plain
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
    ```

* `benzene.inp`:

    ```plain
    $COMPASS
    Title
    benzene grad
    Basis
    cc-pvdz
    Geometry
    file=benzene.xyz
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
    ```

* `testQC.py`:

    ```python
    import numpy as np
    from O1NumHess_QC import O1NumHess_QC

    # read XYZ from file
    qc = O1NumHess_QC("../benzene.xyz")
    # qc = O1NumHess_QC("../benzene.xyz", unit="angstrom")  # specify the unit of xyz file manually
    # print(qc.xyz_angstrom)

    # calculate gradient once
    # qc._calcGrad_BDF(qc.xyz_angstrom, index=0, core=4, mem="4G", inp="../benzene.inp", tempdir = "~/tmp", task_name="abc")

    # parallel
    hessian: np.ndarray = qc.calcHessian_BDF(
        method = "single",
        delta = 1e-3,
        core = 4,
        mem = "4G",
        inp = "../benzene.inp",
        encoding = "utf-8",
        tempdir = "~/tmp",      # BDF tempdir, each calculate will generate a subfolder at there
        task_name = "ddd",      # all the output file will start with this string
        config_name = "",       # config name in your ~/.O1NumHess_QC folder config file
    )
    print(hessian)
    ```

it's recommended to run like this:

```plain
.
├── BDF_RUN/         # run in this folder
├── benzene.inp
├── benzene.xyz
└── testQC.py
```

run in `BDF_RUN` folder with `python3 ../testQC.py`, so that all the output file will in `BDF_RUN`.

## unit test

run this command under project root folder:

```python
python3 -m unittest
```

or simply run files like `test_BDF.py` in the `test` folder.
