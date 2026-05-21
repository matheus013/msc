import yaml
import os

def load_burger_from_yaml(path: str):
    """Load a burger/product definition from a YAML file and return a dict."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data

def load_all_burgers(directory: str):
    """Load all burger YAMLs em um diretório e retornar um dict id → definição."""
    burgers = {}
    for fname in os.listdir(directory):
        if fname.endswith(".yml") or fname.endswith(".yaml"):
            full = os.path.join(directory, fname)
            bdata = load_burger_from_yaml(full)
            bid = bdata["id"]
            burgers[bid] = bdata
    return burgers
