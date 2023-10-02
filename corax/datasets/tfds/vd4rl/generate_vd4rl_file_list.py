import json

from gdown.download_folder import download_and_parse_google_drive_link  # noqa
from gdown.download_folder import get_directory_structure  # noqa


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


def main():
    url = "https://drive.google.com/drive/folders/16CmKNZWIK907Cj5J-_zaEy4YuKOTjupK"
    return_code, gdrive_file = download_and_parse_google_drive_link(
        url, remaining_ok=True
    )
    assert return_code
    d = {
        folder: file_id for file_id, folder in get_directory_structure(gdrive_file, "")
    }
    d = {k: v for k, v in d.items() if v is not None and "64px" not in k}
    print(json.dumps(nest_dict(d), indent=4))


if __name__ == "__main__":
    main()
