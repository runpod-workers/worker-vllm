# Contributing to worker-vllm

## 🚀 Release Process

### Development Workflow

1. **Feature Development**

   ```bash
   git checkout -b feature/your-feature-name
   # Make your changes
   git push origin feature/your-feature-name
   ```

   - Creates pull request → triggers dev build: `runpod/worker-v1-vllm:dev-feature-your-feature-name`

2. **Main Branch**
   ```bash
   git checkout main
   git merge feature/your-feature-name
   git push origin main
   ```
   - No automatic builds on main (staging area)

### Creating Releases

**Method 1: GitHub UI (Recommended)**

1. Go to [Releases](https://github.com/runpod-workers/worker-vllm/releases)
2. Click **"Create a new release"**
3. **Tag version**: `v2.8.0` (with "v" prefix, semantic versioning)
4. **Target**: `main` branch
5. **Title**: `Release 2.8.0`
6. **Description**: Brief changelog
7. Click **"Publish release"**

**Method 2: Git CLI**

```bash
git checkout main
git tag v2.8.0
git push origin v2.8.0
```

### What Happens Automatically

✅ **GitHub Release** created (if using Method 1)  
✅ **Docker Image** built and pushed: `runpod/worker-v1-vllm:v2.8.0`  
✅ **Documentation** updated with new version references

## 📋 Version Format

- **Format**: `vMAJOR.MINOR.PATCH` (e.g., `v2.8.0`)
- **With "v" prefix**: Use `v2.8.0` for git tags
- **Semantic Versioning**: Follow [SemVer](https://semver.org/)

## 🐛 Development

### Running Tests

```bash
# Update test configuration in .runpod/tests.json
# Tests run automatically via RunPod platform
```

### Model Updates

- Update `MODEL_NAME` in `.runpod/tests.json` and `worker-config.json`
- Ensure model has vLLM support and chat template (for OpenAI compatibility)

### Environment Variables

See [README.md](../README.md) for full list of supported environment variables.

## 🔧 CI/CD Workflows

- **Dev builds**: All pull requests → `dev-<branch-name>` images
- **Release builds**: Git tags → versioned images + GitHub releases
- **Manual triggers**: Available in GitHub Actions for emergency releases
