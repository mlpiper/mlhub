import os
import tempfile


def tmp_filepath():
    tmp_file = tempfile.NamedTemporaryFile()
    filepath = tmp_file.name
    tmp_file.close()
    return filepath


def remove_file_safely(filepath):
    if os.path.isfile(filepath):
        os.unlink(filepath)

