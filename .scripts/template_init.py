#!/usr/bin/env python3
"""
Template initialization script for Python library projects.

This script:
1. Gets the repository name from git
2. Converts dashes to underscores to create a valid Python module name
3. Replaces all instances of "munch-group-template" with the module name
4. Renames the src/munch_group_template directory to src/{modulename}

Usage:
    python template_init.py [--dry-run]
"""


print("THIS SCRIPT USES CLAUDE AND IS NOT REPRODUCIBLE, IT DOES NOT WORK EITHER .... MAKE YOUR OWN")



import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


def get_repo_name():
    """Get the repository name from git remote origin URL."""
    try:
        # Get the remote origin URL
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            check=True
        )
        remote_url = result.stdout.strip()
        
        # Extract repo name from various URL formats
        # Handle both SSH and HTTPS URLs
        if remote_url.endswith('.git'):
            repo_name = remote_url.split('/')[-1][:-4]  # Remove .git
        else:
            repo_name = remote_url.split('/')[-1]
            
        return repo_name
    except subprocess.CalledProcessError:
        print("Error: Could not get repository name from git remote")
        sys.exit(1)


def create_module_name(repo_name):
    

    """Convert repository name to valid Python module name."""
    # Replace dashes with underscores and ensure it's a valid Python identifier

    module_name = repo_name.replace('-', '_')
    
    # Ensure it starts with a letter or underscore
    if not module_name[0].isalpha() and re.match(r'[^a-zA-Z0-9_]+', module_name):
        print('Invalid module name:', module_name)
        sys.exit(1)

    return module_name


def find_and_replace_in_file(file_path, old_text, new_text, dry_run=False):
    """Replace text in a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if old_text in content:
            if not dry_run:
                new_content = content.replace(old_text, new_text)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
            return True
    except (UnicodeDecodeError, PermissionError):
        # Skip binary files or files we can't read
        pass
    return False


def get_files_to_process():
    """Get list of files to process, excluding certain directories and file types."""
    exclude_dirs = {
        '.git', '.pixi', '__pycache__', '.pytest_cache', 
        'node_modules', '.venv', 'venv', 'env', '.scripts',
        'munch_group_template.egg-info'
    }
    exclude_extensions = {
        '.pyc', '.pyo', '.so', '.dylib', '.dll', '.exe',
        '.png', '.jpg', '.jpeg', '.gif', '.ico', '.pdf',
        '.zip', '.tar', '.gz', '.bz2', '.xz'
    }
    
    files_to_process = []
    
    for root, dirs, files in os.walk('.'):
        # Remove excluded directories from dirs list to avoid walking into them
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        
        for file in files:
            file_path = Path(root) / file
            
            # Skip files with excluded extensions
            if file_path.suffix in exclude_extensions:
                continue
                
            # Skip lock files and other generated files
            if file in {'pixi.lock', 'package-lock.json', 'yarn.lock'}:
                continue
                
            files_to_process.append(file_path)
    
    return files_to_process


def rename_directory(old_name, new_name, dry_run=False):
    """Rename the src/libraryname directory to src/{new_name}."""
    old_path = Path(f"src/{old_name}")
    new_path = Path(f"src/{new_name}")
    
    if old_path.exists():
        action = "Would rename" if dry_run else "Renaming"
        print(f"{action} directory: {old_path} → {new_path}")
        if not dry_run:
            shutil.move(str(old_path), str(new_path))
        return True
    else:
        print(f"Warning: Directory {old_path} not found")
        return False


def main():
    """Main function to initialize the template."""
    parser = argparse.ArgumentParser(description="Initialize Python library template")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Show what would be changed without making changes")
    args = parser.parse_args()
    
    mode = "🔍 Dry run:" if args.dry_run else "🚀 Initializing"
    print(f"{mode} Python library template...")
    
    # Get repository name and create module name
    repo_name = get_repo_name()
    module_name = create_module_name(repo_name)
    
    print(f"Repository name: {repo_name}")
    print(f"Module name: {module_name}")
    
    # if module_name == "munch_group_template":
    #     print("Module name is already 'munch_group_template', no changes needed.")
    #     return
    
    # Get all files to process
    files_to_process = get_files_to_process()
    
    action = "Would process" if args.dry_run else "Processing"
    print(f"\n📝 {action} {len(files_to_process)} files...")
    
    # Replace text in files
    files_changed = 0
    for file_path in files_to_process:
        if find_and_replace_in_file(file_path, "munch-group-template", repo_name, args.dry_run):
            files_changed += 1
            status = "Would update" if args.dry_run else "✓"
            print(f"  {status} {file_path}")
        if find_and_replace_in_file(file_path, "munch_group_template", module_name, args.dry_run):
            files_changed += 1
            status = "Would update" if args.dry_run else "✓"
            print(f"  {status} {file_path}")

    result = "Would update" if args.dry_run else "Updated"
    print(f"\n{result} {files_changed} files")
    
    # Rename the source directory
    if rename_directory("munch_group_template", module_name, args.dry_run):
        status = "Would rename" if args.dry_run else "✓ Renamed"
        print(f"{status} source directory")
    
    if args.dry_run:
        print(f"\nDry run complete! Run without --dry-run to apply changes.")
    else:
        print(f"\nTemplate initialization complete!")
        print(f"Your library is now configured as: {module_name}")
        print(f"\nNext steps:")
        print(f"1. Update the description in pyproject.toml")
        print(f"2. Update author information") 
        print(f"3. Start developing your library in src/{module_name}/")


if __name__ == "__main__":
    main()
