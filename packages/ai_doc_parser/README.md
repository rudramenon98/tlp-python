## Getting Started

### Pulling Data with Git LFS

This repository uses Git LFS (Large File Storage) to manage large data files. To properly pull the data files:

1. **Install Git LFS** (if not already installed):
   ```bash
   # On Ubuntu/Debian
   sudo apt install git-lfs

   # On Fedora/RHEL
   sudo dnf install git-lfs

   # Or download from: https://git-lfs.github.io/
   ```


2. **Initialize Git LFS** in your local repository:
   ```bash
   git lfs install
   ```

3. **Clone the repository** (if cloning fresh):
   ```bash
   git clone <repository-url>
   cd ai-pdf-parser
   ```

4. **Pull all data files**:
   ```bash
   git lfs pull
   ```

5. **Verify data files are downloaded**:
   ```bash
   ls -la data/
   ```

**Note**: The `data/` directory contains large files managed by Git LFS. Without proper Git LFS setup, these files will appear as small pointer files instead of the actual data.

## Development Guide

1) How to generate training data (including PDFs and labeling)
2) How to run training and create the model

Give detailed steps (including commands and training folder/file names)
