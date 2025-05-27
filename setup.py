from setuptools import setup, find_packages
from setuptools.command.install import install
from pathlib import Path
import os
from textwrap import dedent


class PostInstallCommand(install):
    """generate config files after installation"""

    config = {
        "BDF": dedent('''
            # copy and rename this file to `<program>_config.py` in the same folder,
            # modify and write your configs in it.
            from textwrap import dedent


            config = [
                # below is an example of config, modify it according to your condition of how to run BDF.
                # you can easily write more than one config if you have more than one config to run the program.
                {
                    "name": "BDF", # unique name between different configurations
                    "bash": dedent(
                        # put your bash command below for running BDF successfully
                        # make sure there is NO "BDF_TMPDIR", as it's used by several threads together and can't be shared
                        """
                        #!/bin/bash
                        export BDFHOME=/path/to/bdf-pkg-pro
                        export USE_LIBCINT=no
                        export LD_LIBRARY_PATH=~/intel/mkl/lib/intel64:~/intel/compilers_and_libraries_2019/linux/lib/intel64:$LD_LIBRARY_PATH
                        export LD_LIBRARY_PATH=/path/to/bdf-pkg-pro/extlibs:/path/to/bdf-pkg-pro/libso:$LD_LIBRARY_PATH
                        ulimit -s unlimited
                        ulimit -t unlimited
                        """
                    ).lstrip(), # use lstrip() to remove the first empty line before #!/bin/bash
                    "path": r"/path/to/bdf-pkg-pro/sbin/bdfdrv.py", # program path
                },
            ]
            '''
        ).lstrip(),
        # "ORCA": dedent('''
        #     '''
        # ).lstrip(),
    }

    def run(self):
        install.run(self)
        self._create_config_file()

    def _create_config_file(self):
        # generate config folder
        config_dir = Path("~/.O1NumHess_QC").expanduser().absolute()
        if not config_dir.exists():
            os.makedirs(config_dir, exist_ok=True)

        # ========== generate config files
        for program, conf in self.config.items():
            # use suffix "_example" to avid cover existing config
            path = Path(config_dir / f"{program}_config_example.py")
            path.write_text(conf, "utf-8")

        print(f"\nConfiguration written to {config_dir}.\n**you need to modify them in config folder!!!**\n")


setup(
    name="O1NumHess_QC",
    version="0.1.1",
    packages=find_packages(),
    cmdclass={
        # command run after installation
        'install': PostInstallCommand,
    },
    install_requires=["numpy"],
    python_requires=">=3.6",
)
