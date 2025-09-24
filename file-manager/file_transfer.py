#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------------
# Module Documentation
# -----------------------------------------------------------------------------
"""
Module Overview
---------------
Purpose
    Provides utilities for moving or copying files and folders between 
    directories to automate repetitive file management tasks.

Context
    Useful for organizing projects, archiving, or performing routine 
    file operations without manually dragging and dropping.

Inputs
    - Source path (str): file or folder you want to move/copy.
    - Destination path (str): target file path or directory.
    - Action (str): either "move" or "copy".
    - Overwrite (bool): whether to overwrite existing files/folders.

Outputs
    - A moved or copied file/folder in the target location.
    - Logging information about the action performed.

Usage
-----
    python file_manager.py 
    --src <source_path> 
    --dst <destination_path>
    --action "move" 
    [--overwrite]

Arguments
---------
--src : str
    Path to the source file/folder.
--dst : str
    Path to the destination (file path or directory).
--action : str
    Either "move" or "copy" (default: "copy").
--overwrite : bool
    If set, overwrite existing destination. Otherwise, skip (default: False).

Dependencies
------------
- Standard Library: shutil, argparse, os, logging

Limitations
-----------
- Tested with Python 3.11.
- Assumes UTF-8 encoding.
"""

__author__  = "Aidan Calderhead"
__created__ = "2025-09-24"
__version__ = "1.0.0"
__license__ = "MIT"

# TODO: Check for disk write availability.
# TODO: Add a merge option for folders.

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import logging
import sys
import shutil
import os
import argparse
from typing import Literal

# -----------------------------------------------------------------------------
# Constants / Config
# -----------------------------------------------------------------------------
DEFAULT_ACTION:    Literal["copy", "move"] = "copy"
DEFAULT_OVERWRITE: bool = False

# -----------------------------------------------------------------------------
# Logging Setup
# -----------------------------------------------------------------------------
logging.basicConfig(
    level    = logging.INFO,
    format   = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers = [logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Functions: Validation
# -----------------------------------------------------------------------------
def is_writable(path: str) -> bool:
    """
    Check if a directory (or its parent) is writable and that the path can be 
    created.
    """
    directory = path if os.path.isdir(path) else os.path.dirname(path)
    directory = directory or "."  # fallback to current directory
    return os.access(directory, os.W_OK)


def validate_paths(src: str, dst: str, overwrite: bool) -> None:
    """
    Validate source and destination consistency and permissions.
    """
    # check source exists
    if not os.path.exists(src):
        raise FileNotFoundError(f"Source not found: {src}")

    # Check writability for destination parent
    if not is_writable(dst):
        raise PermissionError(
            f"Destination directory not writable or cannot be created: {dst}"
        )
    
    # check file/folder consistency
    if os.path.isfile(src) and os.path.isdir(dst):
        # allowed: file → directory (places file inside)
        pass
    elif os.path.isdir(src) and os.path.isfile(dst):
        raise ValueError(f"Cannot copy/move folder into a file: {dst}")

    # check overwrite on existing destination
    if not overwrite:
        logger.warning(f"Destination exists and overwrite not allowed: {dst}")
        raise FileExistsError(f"Destination exists: {dst}")
    else:
        logger.info(f"Overwriting existing destination: {dst}")

# -----------------------------------------------------------------------------
# Functions: File Operations
# -----------------------------------------------------------------------------
def remove_item(path: str) -> None:
    """
    Remove a file or folder at the given path.
    """
    if os.path.isdir(path):
        shutil.rmtree(path)
    else:
        os.remove(path)


def copy_item(src: str, dst: str, overwrite: bool = False) -> None:
    """
    Copy a file or folder to the destination, handling overwrite if needed.
    """
    if overwrite:
        remove_item(dst)

    if os.path.isdir(src):
        shutil.copytree(src, dst)
        logger.info(f"Copied folder from {src} to {dst}")
    else:
        if os.path.isdir(dst):
            dst = os.path.join(dst, os.path.basename(src))
        shutil.copy2(src, dst)
        logger.info(f"Copied file from {src} to {dst}")


def move_item(src: str, dst: str, overwrite: bool = False) -> None:
    """
    Move a file or folder to the destination, handling overwrite if needed.
    """
    if overwrite:
        remove_item(dst)

    shutil.move(src, dst)
    logger.info(f"Moved item from {src} to {dst}")

# -----------------------------------------------------------------------------
# Arg Parsing
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description = "Move or copy files and folders."
    )

    parser.add_argument(
        "--src", 
        type     = str, 
        required = True,
        help     = "Path to the source file/folder."
    )
    parser.add_argument(
        "--dst", 
        type     = str, 
        required = True,
        help     = "Path to the destination (file path or directory)."
    )
    parser.add_argument(
        "--action", 
        type    = str, 
        choices = ["copy", "move"], 
        default = DEFAULT_ACTION,
        help    = "Action to perform: copy or move (default: copy)."
    )
    parser.add_argument(
        "--overwrite",
        action  = "store_true",
        default = DEFAULT_OVERWRITE,
        help    = "Allow overwriting existing destination (default: False)."
    )

    return parser.parse_args()

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main(
    src:       str,
    dst:       str,
    action:    str  = DEFAULT_ACTION, 
    overwrite: bool = DEFAULT_OVERWRITE
) -> None:
    logger.info("Starting file transfer task")

    try:
        validate_paths(src, dst, overwrite)

        if action == "copy":
            copy_item(src, dst, overwrite=overwrite)
        elif action == "move":
            move_item(src, dst, overwrite=overwrite)
        else:
            raise ValueError(f"Unsupported action: {action}")
        
        logger.info("Task complete")
    except Exception as e:
        logger.error(f"File transfer failed: {e}")
        raise

# -----------------------------------------------------------------------------
# Entry Point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_args()
    main(
        src       = args.src,
        dst       = args.dst,
        action    = args.action,
        overwrite = args.overwrite
    )
