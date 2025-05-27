import importlib.util
import sys
from pathlib import Path
from typing import Dict



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
    print(getConfig("BDF"))
    print(getConfig("BDF", "BDf")) # error test
