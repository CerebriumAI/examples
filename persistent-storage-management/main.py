""" 
A little utility to view or clear files from your persistent storage.
Use it to view or clear out folders
"""
from typing import Optional
from pydantic import BaseModel
import os


class Item(BaseModel):
    ls: Optional[str] = None
    tree: Optional[str] = None
    rm: Optional[str] = None


def make_tree(directory: str):
    """Makes a string/file containing a tree of the directory passed"""
    tree = f"{directory}:\n"
    tree += generate_tree(directory, "")
    return tree


def generate_tree(root, indent):
    tree = ""
    with os.scandir(root) as entries:
        for entry in entries:
            if entry.is_dir():
                tree += f"{indent}+-- {entry.name}/\n"
                tree += generate_tree(entry.path, indent + "    ")
            else:
                tree += f"{indent}+-- {entry.name}\n"
    return tree


def get_persistent_storage_path(path: str):
    """Returns the path to the persistent storage"""
    if path.startswith("/persistent-storage/"):
        return path
    elif path.startswith("/"):
        return os.path.join("/persistent-storage/", path[1:])
    elif path.startswith("./"):
        return os.path.join("/persistent-storage/", path[2:])
    else:
        return os.path.join("/persistent-storage/", path)


def predict(item, run_id, logger):
    params = Item(**item)
    results = {}
    if params.ls:
        params.ls = get_persistent_storage_path(params.ls)
        logger.info("Listing directory")
        results["ls"] = os.listdir(params.ls)

    if params.tree:
        params.tree = get_persistent_storage_path(params.tree)
        logger.info("Generating tree")
        results["tree"] = make_tree(params.tree)

    if params.rm:
        params.rm = "/persistent-storage/" + params.rm
        logger.info("Removing directory")
        if os.path.isdir(params.rm):
            removed = []
            for root, dirs, files in os.walk(params.rm, topdown=False):
                for name in files:
                    try:
                        os.remove(os.path.join(root, name))
                    except:
                        pass
                        removed.append(os.path.join(root, name))
                for name in dirs:
                    try:
                        os.rmdir(os.path.join(root, name))
                    except:
                        pass
                        removed.append(os.path.join(root, name))
        else:
            try:
                os.remove(params.rm)
            except:
                pass
            removed = params.rm
        results["rm"] = removed

    return results
