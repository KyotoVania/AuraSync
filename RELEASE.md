# Release Guide for AuraSync

This document explains how to create releases for AuraSync and trigger the automated deployment workflow.

---

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Release Process](#release-process)
- [Tag Naming Convention](#tag-naming-convention)
- [CI/CD Workflow](#cicd-workflow)
- [Manual Deployment](#manual-deployment)
- [Troubleshooting](#troubleshooting)

---

## Overview

AuraSync uses **Git tags** to trigger automated release builds. When you push a tag starting with `v` (e.g., `v1.0.0`), the CI/CD pipeline automatically:

1. Runs code quality checks (formatting, static analysis)
2. Builds the documentation and deploys to GitHub Pages
3. Runs tests on Linux, Windows, and macOS
4. Builds release binaries for all three platforms
5. Creates a GitHub Release with the compiled artifacts

---

## Prerequisites

Before creating a release, ensure:

- [ ] All code changes are committed and pushed to the repository
- [ ] All tests pass locally
- [ ] The code builds successfully on your platform
- [ ] The version number follows [Semantic Versioning](https://semver.org/)
- [ ] You have push access to the repository

---

## Release Process

### Step 1: Prepare Your Code

Make sure all changes are committed:

```bash
# Check the status of your repository
git status

# Commit any pending changes
git add .
git commit -m "Prepare for release v1.0.0"

# Push your changes to the main branch
git push origin main
```

### Step 2: Create a Tag

Create an annotated tag with a descriptive message:

```bash
# Annotated tag (recommended)
git tag -a v1.0.0 -m "Release version 1.0.0 - Initial stable release"

# View the tag details
git show v1.0.0
```

**Note:** Annotated tags are recommended because they include metadata like the tagger's name, email, and date.

### Step 3: Push the Tag

Push the tag to GitHub to trigger the deployment:

```bash
# Push a specific tag
git push origin v1.0.0

# Alternative: Push all local tags at once
git push --tags
```

### Step 4: Monitor the Deployment

1. Navigate to the [Actions tab](https://github.com/KyotoVania/AuraSync/actions) on GitHub
2. Find the **Cpp_CI** workflow run for your tag
3. Monitor the progress of:
   - **quality** - Code quality checks
   - **docs** - Documentation build
   - **test** - Multi-platform testing
   - **deploy** - Release artifact creation

### Step 5: Verify the Release

Once the workflow completes:

1. Go to the [Releases page](https://github.com/KyotoVania/AuraSync/releases)
2. Find your newly created release
3. Verify that all platform binaries are attached:
   - `ave_analysis-linux`
   - `ave_analysis-windows.exe`
   - `ave_analysis-macos`

---

## Tag Naming Convention

AuraSync follows [Semantic Versioning](https://semver.org/) for releases:

### Version Format

```
vMAJOR.MINOR.PATCH[-PRERELEASE][+BUILD]
```

### Examples

- `v1.0.0` - First stable release
- `v1.2.3` - Bug fixes and minor improvements
- `v2.0.0` - Major version with breaking changes
- `v1.0.0-alpha` - Alpha pre-release
- `v1.0.0-beta.1` - Beta pre-release
- `v1.0.0-rc.1` - Release candidate
- `v1.0.0+20240101` - Release with build metadata

### When to Increment

- **MAJOR** - Incompatible API changes
- **MINOR** - New functionality, backward compatible
- **PATCH** - Backward compatible bug fixes

---

## CI/CD Workflow

### Trigger Conditions

The **Deploy Release** job is triggered when:

1. A tag matching the pattern `v*` is pushed to GitHub
2. **OR** the workflow is manually triggered via `workflow_dispatch`

### Workflow Configuration

From `.github/workflows/cpp_ci.yml`:

```yaml
on:
  push:
    tags: [ "v*" ]
  workflow_dispatch:
```

### Build Matrix

The deployment builds for three platforms:

| Platform | Runner | Artifact Name |
|----------|--------|---------------|
| Linux | `ubuntu-latest` | `ave_analysis-linux` |
| Windows | `windows-latest` | `ave_analysis-windows.exe` |
| macOS | `macos-latest` | `ave_analysis-macos` |

---

## Manual Deployment

If you need to manually trigger a deployment without creating a tag:

### Using GitHub Web UI

1. Go to the [Actions tab](https://github.com/KyotoVania/AuraSync/actions)
2. Select the **Cpp_CI** workflow
3. Click "Run workflow"
4. Select the branch
5. Click "Run workflow"

### Using GitHub CLI

```bash
# Install GitHub CLI if not already installed
# https://cli.github.com/

# Trigger the workflow manually
gh workflow run cpp_ci.yml
```

**Note:** Manual deployments via `workflow_dispatch` will run the deploy job, but won't automatically create a GitHub Release. You'll need to create the release manually.

---

## Troubleshooting

### Tag Already Exists

If you try to create a tag that already exists:

```bash
# Delete the local tag
git tag -d v1.0.0

# Delete the remote tag
git push origin :refs/tags/v1.0.0

# Recreate and push the tag
git tag -a v1.0.0 -m "Release version 1.0.0"
git push origin v1.0.0
```

### Deployment Failed

If the CI/CD workflow fails:

1. Check the [Actions tab](https://github.com/KyotoVania/AuraSync/actions) for error logs
2. Common issues:
   - **Build failure** - Check compiler errors in the test job
   - **Test failure** - Review test output for failing tests
   - **Missing dependencies** - Ensure all dependencies are properly configured

### Tag Pushed But No Release Created

Possible causes:

1. **Tag format incorrect** - Must start with `v` (e.g., `v1.0.0`)
2. **Tests failed** - The deploy job requires all tests to pass
3. **Permissions issue** - Check GitHub Actions permissions

### Listing All Tags

```bash
# List all local tags
git tag

# List all remote tags
git ls-remote --tags origin

# List tags with messages
git tag -n
```

### Checking Tag Details

```bash
# Show tag information
git show v1.0.0

# Show tag commit
git rev-list -n 1 v1.0.0
```

---

## Best Practices

1. **Always test before tagging** - Run the full test suite locally
2. **Use annotated tags** - Include meaningful release notes in the tag message
3. **Follow semantic versioning** - Make version numbers meaningful
4. **Update CHANGELOG** - Document changes in each release
5. **Create release notes** - Use GitHub's release notes feature to describe changes
6. **Don't force-push tags** - Tags should be immutable; if needed, create a new version

---

## Example Release Workflow

Here's a complete example of creating a release:

```bash
# 1. Update version in code if needed
# (e.g., update version constants, package files)

# 2. Commit all changes
git add .
git commit -m "chore: bump version to 1.2.0"

# 3. Push changes
git push origin main

# 4. Create and push the tag
git tag -a v1.2.0 -m "Release v1.2.0

Features:
- Added new BPM detection algorithm
- Improved spectral analysis performance

Bug Fixes:
- Fixed memory leak in FFT processing
- Corrected audio buffer handling on Windows
"

git push origin v1.2.0

# 5. Monitor the Actions tab for deployment progress
# 6. Verify the release on GitHub
```

---

## Additional Resources

- [Semantic Versioning](https://semver.org/)
- [Git Tagging Documentation](https://git-scm.com/book/en/v2/Git-Basics-Tagging)
- [GitHub Releases](https://docs.github.com/en/repositories/releasing-projects-on-github)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)

---

For questions or issues with the release process, please open an issue on the [GitHub repository](https://github.com/KyotoVania/AuraSync/issues).
