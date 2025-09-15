#!/usr/bin/env python
import re
import sys

import argparse

# major, minor, patch arguments
parser = argparse.ArgumentParser(description='Release tag script')
parser.add_argument('--major', action='store_true', help='Bump major version number')
parser.add_argument('--minor', action='store_true', help='Bump minor version number') 
parser.add_argument('--patch', action='store_true', help='Bump patch version number')
parser.add_argument('--pre', action='store_true', help='Add or bump prerelease rc suffix')
parser.add_argument('--release', action='store_true', help='Remove prerelease rc suffix')

args = parser.parse_args()
major = int(args.major)
minor = int(args.minor)
patch = int(args.patch)

pre = int(args.pre)
release = int(args.release)

# if sum([pre, release]) > 1:
#     print("Please specify at most one of --pre and --release")
#     sys.exit(1)

if not sum([major, minor, patch, pre, release]): 
    print("Please specify at least one flag")
    sys.exit(1)

# if pre and sum([major, minor, patch]) == 0:
#     print("When using --pre, give at most one of --major, --minor, or --patch")
#     sys.exit(1)
if release and sum([major, minor, patch]) > 0:
    print("When using --release, do not give any of --major, --minor, or --patch")
    sys.exit(1)
elif release and sum([major, minor, patch]) == 0:
    print("Please specify exactly one of --major, --minor, or --patch")
    sys.exit(1)


# file, regex pairs
spec = {
    'pyproject.toml':  r'(version = ")(\d+)\.(\d+)\.(\d+)(?:.rc(\d+))?(")',
}

def bump(content, regex):
    global major, minor, patch, post, pre, release
    m = re.search(regex, content)
    assert m is not None, "Version not found"
    prefix = m.group(1)
    _major = int(m.group(2))
    _minor = int(m.group(3))
    _patch = int(m.group(4))
    _pre = int(m.group(5)) if m.group(5) else 0
    postfix = m.group(6)
    if _pre:
        version = f'{_major}.{_minor}.{_patch}.rc{_pre}'
    else:
        version = f'{_major}.{_minor}.{_patch}'

    print(version)

    assert not (_pre == 0 and release), "Cannot release a version with a pre-release tag"

    # zero pre if other tag is bumped without --pre flag
    if (major or minor or patch) and not pre:
         pre = -_pre

    # zero lower tags if higher tag is bumped
    if major or minor:
        patch = -_patch
    if major:
        minor = -_minor

    if release or _pre+pre == 0:        
        new_version = f'{_major+major}.{_minor+minor}.{_patch+patch}'
    else:
        new_version = f'{_major+major}.{_minor+minor}.{_patch+patch}.rc{_pre+pre}'

    print(version, new_version)

    # match = f'{version}'
    repl = f'{new_version}'
    # match = f'{prefix}{version}{postfix}'
    # repl = f'{prefix}{new_version}{postfix}'
    new_content = re.sub(regex, f'version = "{new_version}"', content)
    assert new_content != content
    return new_content, new_version

new_contents = {}
new_versions = []
for file, regex in spec.items():
    with open(file, 'r') as f:
        content = f.read()
    new_content, new_version = bump(content, regex)
    new_contents[file] = new_content
    new_versions.append(new_version)

# all versions should be the same
assert len(new_versions) == len(spec)
assert len(set(new_versions)) == 1, new_versions
new_version = list(new_versions)[0]

# _major, _minor, _patch = map(int, new_version.split('.'))
# old_version = f'{_major-major}.{_minor-minor}.{_patch-patch}'

just = max([len(x) for x in spec.keys()])
# print(f"Version bump:\n  {old_version} -> {new_version}\nFiles changed:")
print(f"Version bump to:\n {new_version}\nFiles changed:")
for file, content in new_contents.items():
    print(f"  {file.ljust(just)}  (replaced {len(re.findall(new_version, content))} instance)")
    with open(file, 'w') as f:
        f.write(content)

