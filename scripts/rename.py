#!/usr/bin/env python3

import argparse
import os
import shutil
from pathlib import Path
import subprocess

def convert_to_underscore(name: str) -> str:
    """Convert hyphenated name to underscore format."""
    return name.replace("-", "_")


def find_and_replace_in_file(file_path: Path, old_hyphen: str, new_hyphen: str,
                              old_underscore: str, new_underscore: str) -> bool:
    """
    Replace occurrences in a single file.

    Returns True if any replacements were made, False otherwise.
    """
    try:
        # Skip binary files and certain directories
        if file_path.suffix in ['.pyc', '.so', '.dylib', '.png', '.jpg', '.jpeg', '.gif', '.ico', '.pdf']:
            return False

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except (UnicodeDecodeError, PermissionError):
        # Skip binary files or files we can't read
        return False

    original_content = content
    content = content.replace(old_hyphen, new_hyphen)
    content = content.replace(old_underscore, new_underscore)

    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True

    return False


def rename_library(new_name: str):
    """Rename the library from /Users/kmt/arg-dashboard to the new name."""
    old_hyphen = "munch-group-library"
    old_underscore = "munch_group_library"

    new_hyphen = new_name
    new_underscore = convert_to_underscore(new_name)

    # Get the project root (parent of scripts directory)
    project_root = Path(__file__).parent.parent

    print(f"Renaming library from '{old_hyphen}' to '{new_hyphen}'")
    print(f"Renaming library from '{old_underscore}' to '{new_underscore}'")
    print(f"Project root: {project_root}")
    print()

    # Track changes
    files_modified = []

    # Walk through all files in the project
    for root, dirs, files in os.walk(project_root):
        # Skip certain directories
        dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', '.pytest_cache',
                                                  'node_modules', '.venv', 'venv', 'dist',
                                                  'build', '*.egg-info', '.ipynb_checkpoints']]

        for file in files:
            if file == 'rename.py':
                continue
            file_path = Path(root) / file
            if find_and_replace_in_file(file_path, old_hyphen, new_hyphen,
                                        old_underscore, new_underscore):
                files_modified.append(file_path.relative_to(project_root))

    print(f"Modified {len(files_modified)} files:")
    for file_path in files_modified:
        print(f"  - {file_path}")
    print()

    # Rename the src directory
    old_src_dir = project_root / "src" / old_underscore
    new_src_dir = project_root / "src" / new_underscore
    # old_src_dir = Path(old_underscore).relative_to(project_root)
    # new_src_dir = Path(new_underscore).relative_to(project_root)

    if old_src_dir.exists():
        # print(f"Renaming directory: {old_src_dir.relative_to(project_root)} -> {new_src_dir.relative_to(project_root)}")
        print(f"Renaming directory: {old_src_dir} -> {new_src_dir}")
        shutil.move(str(old_src_dir), str(new_src_dir))
        print("✓ Directory renamed successfully")
    else:
        print(f"Warning: Directory {old_src_dir} not found")

    print()
    print("✓ Library rename complete!")
    print(f"\nNext steps:")
    print(f"  1. Review the changes with: git diff")
    print(f"  2. Test the renamed library")
    print(f"  3. Commit the changes")


def main():

    git_executable = shutil.which('git')

    cmd = 'git rev-parse --show-toplevel'.split()
    cmd[0] = shutil.which(cmd[0])
    if not cmd[0]:
        print('A git executable was not found', file=sys.stderr)
        sys.exit(1)
    try:
        new_name = subprocess.check_output(cmd)
    except CalledProcessError:
        print('Could not get repo name using git', file=sys.stderr)
        sys.exit(1)

    new_name = new_name.strip().decode('utf-8')
    if not new_name:
        print("Library name cannot be empty", file=sys.stderr)

    rename_library(new_name)

    # Confirm with user
    print(f"\nThis will rename '/Users/kmt/arg-dashboard' to '{new_name}' throughout the project.")
    response = input("Continue? [y/N]: ")
    if response.lower() not in ['y', 'yes']:
        print("Aborted.")
        return

    rename_library(new_name)


if __name__ == "__main__":
    main()
