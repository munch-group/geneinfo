# GitHub Actions Workflows

## Workflows Overview

### 1. `build-docs-and-installers.yml` (Main Workflow)
This is the primary workflow that handles both documentation building and installer creation.

**Triggers:**
- **Push to main branch**: Builds and publishes documentation only (no installers)
- **Release published**: Builds installers AND documentation
- **Manual dispatch**: Can optionally build installers via `build_installers` input

**Key Features:**
- Builds native installers only for official releases (not prereleases)
- Uses macOS runner for building both macOS (.dmg) and Windows (.exe) installers
- Installs NSIS via Homebrew for Windows installer creation
- Deploys documentation to GitHub Pages
- Attaches installers to GitHub releases

### 2. `conda-release.yml`
Handles conda package releases for Franklin.

### 3. `quarto-publish.yml` (Deprecated)
Original Quarto documentation publishing workflow. Superseded by `build-docs-and-installers.yml`.

## Installer Build Process

### When Installers Are Built
Installers are **ONLY** built when:
1. A new release is published (not a prerelease)
2. Manual workflow dispatch with `build_installers: true`

### What Gets Built
- **macOS**: `Franklin-Installer-macOS.dmg` - Native macOS application
- **Windows**: `Franklin-Installer-Windows.exe` - NSIS-based installer
- **Cross-platform**: `franklin_installer_gui.py` - Python GUI installer

### Dependencies
- **NSIS**: Installed via `brew install makensis` on macOS runner
- **Python**: For dependency checking and cross-platform installer
- **Quarto**: For documentation rendering

## Documentation Deployment

### Regular Pushes to Main
- Only documentation is built and deployed
- Installer download pages show links to GitHub releases
- Fast build process (uses Ubuntu runner)

### Release Builds
- Full build with installers
- Installers are embedded in documentation
- Direct download links available from docs
- Slower build process (uses macOS runner)

## Usage

### Creating a New Release with Installers
1. Create a new release on GitHub (not a prerelease)
2. The workflow automatically triggers
3. Installers are built and attached to the release
4. Documentation is updated with direct download links

### Manual Build with Installers
```yaml
# Trigger workflow manually with installers
gh workflow run build-docs-and-installers.yml -f build_installers=true
```

### Testing Changes
Push to main branch to test documentation changes without building installers.

## File Locations

- **Installer build script**: `src/franklin_cli/dependencies/build_native_installers.sh`
- **Installer sources**: `src/franklin_cli/dependencies/`
- **Built installers**: `src/franklin_cli/dependencies/dist/`
- **Documentation**: `docs/`
- **Published site**: GitHub Pages at `https://[org].github.io/franklin/`