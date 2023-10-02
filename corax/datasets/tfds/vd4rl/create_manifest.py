import collections
import json
import pathlib

from absl import app
from absl import flags

_DATA_ROOT = flags.DEFINE_string("data_root", None, "Data root", required=True)
_MANIFEST_PATH = flags.DEFINE_string("manifest_path", "manifest.json", "Data root")


def nest_dict(dict1):
    result = {}
    for k, v in dict1.items():
        # for each key call method split_rec which
        # will split keys to form recursively
        # nested dictionary
        split_rec(k, v, result)
    return result


def split_rec(k, v, out):
    # splitting keys in dict
    # calling_recursively to break items on '_'
    k, *rest = k.split("/", 1)
    if rest:
        split_rec(rest[0], v, out.setdefault(k, {}))
    else:
        out[k] = v


def unflatten(dictionary):
    resultDict = {}
    for key, value in dictionary.items():
        parts = key.split("/")
        d = resultDict
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
    return resultDict


def main(_):
    data_root = pathlib.Path(_DATA_ROOT.value)

    paths = [str(p.relative_to(data_root)) for p in data_root.rglob("*") if p.is_file()]

    manifest = collections.defaultdict(list)
    for path in paths:
        parts = path.split("/")
        manifest["/".join(parts[:-1])].append(parts[-1])

    for k in manifest:
        manifest[k] = sorted(manifest[k])

    with open(_MANIFEST_PATH.value, "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    app.run(main)
