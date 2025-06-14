# O1NumHess_QC Documentation

**Attention:** Development is not yet complete, things may change.

In the following text, O1NumHess_QC is sometimes abbreviated as "QC".

## Installation

## Usage

除`O1NumHess`库外，QC仅依赖`numpy`一个第三方库进行数学运算，相关的分子坐标等数据均以`np.ndarray`格式进行存储和返回。

### config

配置文件的详细说明

<!-- TODO -->

### Initialization

Import O1NumHess_QC and supply a `.xyz` file which contains **only one** molecular coordinates for initialization. QC will read the file and store the coordinates in the instance.

* If more than one molecule is contained, an error will be raised.
* If not manually specified by the user, by default the `.xyz` file units will be treated as `angstrom` and read accordingly. Use the `unit` parameter to specify the units of the `.xyz` file. This parameter accepts either `angstrom` or `bohr` as valid string inputs (case-insensitive).

```Python
from O1NumHess_QC import O1NumHess_QC

qc = O1NumHess_QC("../benzene.xyz")
# or specify the unit of .xyz file manually
qc = O1NumHess_QC("../benzene.xyz", unit="bohr")
```

初始化之后，分子的坐标会被以`bohr`为单位存储在qc的实例内部，用户可以通过下面的两个属性分别获取到以`bohr`和`angstrom`做为单位的坐标。

```Python
qc.xyz_bohr
qc.xyz_angstrom
```

初始化完成之后，用户可以调用4个软件相关的函数完成和4个软件的交互

### BDF

如果正确配置了BDF的配置文件，可以使用下面的代码调用BDF计算Hessian矩阵。这里仅解释各个参数的含义，详细的工作原理请参见`Development`部分。

```Python
hessian = qc.calcHessian_BDF(
    method = "single",
    delta = 1e-3,
    core = 4,
    mem = "4G",
    inp = "../benzene.inp",
    encoding = "utf-8",
    tempdir = "~/tmp",
    task_name = "abc",
    config_name = "BDF",
)
```

* `method`：目前只能设置为`single`或`double`（忽略大小写），代表计算梯度时调用`O1NumHess`所使用的算法。
* `delta`：计算梯度时传递给`O1NumHess`的计算精度。
* `core`和`mem`：每个梯度计算时使用的核心数量和内存大小，对应运行BDF时的线程数和内存设置的环境变量`OMP_NUM_THREADS`和`OMP_STACKSIZE`，core为int类型，mem为字符串类型，参见[Installation and Operation](https://bdf-manual.readthedocs.io/en/latest/Installation.html#run-bdf-standalone-and-execute-the-job-with-a-shell-script)。
* `inp`：调用BDF计算的配置文件，QC会配合O1NumHess生成对分子坐标的扰动，并借助该文件多次计算梯度，最终求出Hessian矩阵
  * 该文件中，要计算的分子坐标应写成`file=xxx.xyz`这样的格式，参考`readme.md`中的例子或者官方文档[Input and output formats](https://bdf-manual.readthedocs.io/en/latest/Input%20and%20Output.html#read-the-molecular-coordinates-from-the-specified-file)。
  * **关于单位**：`.xyz`文件中的单位在初始化时指定，计算时，如果要采用bohr作为单位进行计算，直接在inp文件中配置即可，QC会自动把单位转换为对应的格式进行计算
    * 例如`.xyz`文件中的单位可以是angstrom，`.inp`文件中可以指定单位为bohr进行计算，QC会自动将坐标转换为bohr传给BDF进行计算
* `encoding`：（可选，默认为utf-8）inp输入文件文件的编码，用来确保inp文件中存在注释时可以被正确读取
* `task_name`：任务名称，计算多个梯度时的前缀名称，用来区分多次任务。
  * 例如当task_name为abc时，第1次计算梯度时生成的相关文件为`abc_001.xxx`
* `tempdir`：（可选，默认为`~/tmp`）BDF运行时的临时文件夹，对应运行BDF时的线程数和内存设置的环境变量`BDF_TMPDIR`，参见[Installation and Operation](https://bdf-manual.readthedocs.io/en/latest/Installation.html#run-bdf-standalone-and-execute-the-job-with-a-shell-script)
  * 请确保你对该文件夹拥有写入和删除的权限
  * 例如，当tempdir为`~/tmp`，task_name为abc时，计算梯度时会生成`~/tmp/abc_001/abc_001.xxx`，`~/tmp/abc_002/abc_002.xxx`等文件夹和文件
* `config_name`：BDF配置文件中的配置名称，配置文件中可以存在多个名称不同的配置。
  * 如果该参数为空，默认使用配置文件中找到的第一个配置
  * 该参数的用途是：如果用户存在多个版本或者多个运行配置，可在配置文件中写入多个配置，并指定每个任务使用的具体配置。

## Development

For detailed function explanations, please refer to the specific docstrings and related comments in the code files. Comments use separators to delineate major steps. Therefore, this section only explains the main logic and some important notes.

### `type: ignore`

The code is developed in VSCode using the `PyLance` extension for "standard" level [type checking](https://microsoft.github.io/pyright/#/configuration?id=type-check-diagnostics-settings). Since Python is dynamically typed, it is unrealistic to make all code pass strict type checking. Therefore, you may sometimes see `# type: ignore` comments in the code, which instructs the type checker to ignore type checking for that specific line.

### Concept

这里介绍QC调用O1NumHess配合计算Hessian的基本逻辑。

O1NumHess和量子化学无关，通过接受一个向量x和一个用户提供的梯度函数g来计算Hessian矩阵。O1NumHess会对输入的x的每个分量进行扰动，并调用函数g计算一次扰动后的梯度，最终根据多次扰动后计算得到的梯度求出Hessian。

QC负责和量子化学相关的软件交互，实现上述的求梯度的函数g，并调用O1NumHess求出Hessian。具体的步骤为：

* 用户需要求出某个分子的Hessian，调用QC并传递参数
* QC中封装了调用BDF等软件求出单个梯度的函数g
* QC把函数g做为参数传递给O1NumHess，调用它求出Hessian
* O1NumHess在运行过程中重复扰动并调用QC中的梯度函数g得到不同的梯度，根据不同的梯度求出Hessian
* O1NumHess将求出的梯度返回给QC，QC将得到的Hessian返回给用户

### BDF

这里介绍QC和BDF交互的具体逻辑。

QC中存在`calcHessian_BDF`和`_calcGrad_BDF`两个函数。

* `_calcGrad_BDF`为调用BDF求出梯度的函数g
* 用户调用`calcHessian_BDF`，由`calcHessian_BDF`函数接收相关参数后，将`_calcGrad_BDF`函数作为g做为参数初始化`O1NumHess`，并调用`O1NumHess`完成对Hessian的计算

