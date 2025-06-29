#!/usr/bin/env python
import re
import sys

import argparse

# major, minor, patch arguments
parser = argparse.ArgumentParser(description='Release tag script')
parser.add_argument('--major', action='store_true', help='Bump major version number')
parser.add_argument('--minor', action='store_true', help='Bump minor version number') 
parser.add_argument('--patch', action='store_true', help='Bump patch version number')
args = parser.parse_args()
major = int(args.major)
minor = int(args.minor)
patch = int(args.patch)

if sum([major, minor, patch]) != 1:
    print("Please specify exactly one of --major, --minor, or --patch")
    sys.exit(1)

# file, regex pairs
spec = {
    'pyproject.toml':  r'(version = ")(\d+)\.(\d+)\.(\d+)(")',
}

def bump(content, m):
    global major, minor, patch
    assert m is not None, "Version not found"
    prefix = m.group(1)
    _major = int(m.group(2))
    _minor = int(m.group(3))
    _patch = int(m.group(4))
    postfix = m.group(5)
    version = f'{_major}.{_minor}.{_patch}'

    if major or minor:
        patch = -_patch
    if major:
        minor = -_minor

    new_version = f'{_major+major}.{_minor+minor}.{_patch+patch}'
    match = f'{version}'
    repl = f'{new_version}'
    # match = f'{prefix}{version}{postfix}'
    # repl = f'{prefix}{new_version}{postfix}'
    new_content = content.replace(match, repl)
    assert new_content != content
    return new_content, new_version

new_contents = {}
new_versions = []
for file, regex in spec.items():
    with open(file, 'r') as f:
        content = f.read()
    m = re.search(regex, content)
    new_content, new_version = bump(content, m)
    new_contents[file] = new_content
    new_versions.append(new_version)

# all versions should be the same
assert len(new_versions) == len(spec)
assert len(set(new_versions)) == 1, new_versions
new_version = list(new_versions)[0]

_major, _minor, _patch = map(int, new_version.split('.'))
old_version = f'{_major-major}.{_minor-minor}.{_patch-patch}'

just = max([len(x) for x in spec.keys()])
print(f"Version bump:\n  {old_version} -> {new_version}\nFiles changed:")
for file, content in new_contents.items():
    print(f"  {file.ljust(just)}  (replaced {len(re.findall(new_version, content))} instance)")
    with open(file, 'w') as f:
        f.write(content)

