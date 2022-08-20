###############################################################################
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2021-2022 ETH Zurich.
###############################################################################

import os
from os import access, R_OK, W_OK, sep, makedirs
from os.path import normpath, abspath, isfile, exists, dirname, splitext


def add_fname_suffix(fname, suffix, delete_ext=False) -> str:
    fname_core, fext = splitext(fname)
    if delete_ext:
        return fname_core + '_' + suffix
    else:
        return fname_core + '_' + suffix + fext


def ensure_dir_structure(dname: str):
    assert not isfile(dname), "argument must not be a path to an existing file"
    dir_list = normpath(dname).split(sep)
    if isfile(dname):
        new_dir = sep.join(dir_list[:-1])
    else:
        new_dir = sep.join(dir_list)
    makedirs(new_dir, exist_ok=True)


def canonize_fname(rel_fname):
    return abspath(rel_fname)


def assert_file_readable(fname):
    if not isfile(fname):
        raise Exception("File does not exist: " + fname)
    if not access(fname, R_OK):
        raise Exception("I do not have read access to file: " + fname)


def assert_file_writable(fname):
    if exists(fname):
        if isfile(fname):
            if not access(fname, W_OK):
                raise Exception("I cannot write to file: " + fname)
        else:
            raise Exception("The file is not writable as it is a directory: " + fname)
    else:
        pdir = dirname(fname)
        if not pdir: pdir = '.'
        if not access(pdir, W_OK):
            raise Exception("Cannot create file (" + fname +
                            ") as the parent directory is not writable.")


def delete_tmp_files(folder: str, recurse=False):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        if os.path.isdir(file_path) and recurse:
            delete_tmp_files(file_path, recurse)
    # the tmp folder is empty, delete it as well
    if not os.listdir(folder):
        os.rmdir(folder)
