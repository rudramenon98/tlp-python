# TLP-python

Private Algorithm Code (Python)

This repository contains a collection of Python packages. Each packages can be installed individually.

## Getting started.
Run the ./scripts/setup_enviornment.sh to create a venv and install each package and their dependencies.

## Package Hierarchy

The packages in this repository follow a strict dependency hierarchy. Each package can only import from packages that appear below it in this hierarchy:

```
TLP-Python
├── AI-Doc-Parsing (ai_doc_parser)
│   ├── Training
│   ├── Parsers
│   └── Inference
├── DictionaryExtraction
├── Indexer
├── Scraping (scraping)
├── Embedding
├── Database (database)
└── Common-Tools (common_tools)
```

### Import Dependency Rules

Packages must respect the hierarchy order when importing other packages:

- **AI-Doc-Parsing** can import from all packages below it (DictionaryExtraction, Indexer, Scraping, Embedding, Database, Common-Tools)
- **DictionaryExtraction** can import from Indexer, Scraping, Embedding, Database, Common-Tools
- **Indexer** can import from Scraping, Embedding, Database, Common-Tools
- **Scraping** can import from Embedding, Database, Common-Tools
- **Embedding** can import from Database, Common-Tools
- **Database** can import from Common-Tools only
- **Common-Tools** is the base layer and cannot import from any other TLP packages

This hierarchy ensures that:
- Lower-level packages remain independent and reusable
- Dependencies flow in one direction (top to bottom)
- Circular dependencies are prevented
- The architecture maintains clear separation of concerns

## Package Layout

All packages are located in the `packages/` directory. Most packages follow a standard Python package structure:

```
packages/
└── <package_name>/
    ├── src/
    │   └── <package_name>/
    │       ├── __init__.py
    │       └── ... (package modules)
    ├── pyproject.toml          # Package metadata and dependencies
    ├── .gitignore             # Git ignore patterns (optional)
    └── README.md              # Package-specific documentation (optional)
```

### Key Components

- **`src/` directory**: Contains the package source code organized in a subdirectory matching the package name. This follows the [src-layout](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/) pattern, which is the recommended approach for Python packages.

- **`pyproject.toml`**: Modern Python packaging configuration file that defines:
  - Package metadata (name, version, description, authors)
  - Python version requirements
  - Dependencies
  - Build system configuration (setuptools)
  - Package discovery settings (pointing to the `src/` directory)
  - Optional tool configurations (black, isort, mypy, pytest, etc.)

- **`.gitignore`**: Package-specific ignore patterns for build artifacts, cache files, and other generated content that shouldn't be committed.

## Installation

Each package can be installed individually in development (editable) mode. Navigate to the package directory and install it using pip:

```bash
cd packages/<package_name>
pip install -e .
```

Or install from the root directory:

```bash
pip install -e packages/<package_name>
```

The `-e` flag installs the package in "editable" mode, meaning changes to the source code are immediately reflected without reinstalling.


## Packages Overview

### `ai_doc_parser`

**Purpose**: AI-powered PDF parser for extracting structured information from documents.

**Description**: This package provides machine learning-based document parsing capabilities. It uses trained models (RandomForest and XGBoost classifiers) to classify and extract structured content from PDF documents. The package includes:

- PDF text extraction using PyMuPDF
- Feature computation for machine learning models
- Training pipelines for document classification
- Label extraction from various document formats (XML, LaTeX, DOCX)
- Inference APIs for parsing PDFs using trained models
- Support for various document types including regulatory documents

**Key Dependencies**: PyMuPDF, scikit-learn, xgboost, pandas, numpy, FastAPI, PyQt5

**Python Version**: >=3.8

### `common_tools`

**Purpose**: Common utilities and tools shared across TLP Python packages.

**Description**: A lightweight package providing shared functionality used by other packages in the repository. Currently includes:

- Logging configuration utilities with support for log4j-style level mapping
- Command-line argument parsing for logging setup

**Key Dependencies**: None (standard library only)

**Python Version**: >=3.11

### `database`

**Purpose**: Database service layer for TLP Python applications.

**Description**: Provides SQLAlchemy-based database access and entity management for the TLP system. The package includes:

- Database connection management via MySQLDriver
- Entity models for documents, definitions, repositories, scraping URLs, vectors, and more
- Service layer functions for common database operations:
  - Document management (insert, update, find, bulk operations)
  - Definition/dictionary lookups
  - Scraping URL tracking and logging
- Configuration management for database credentials

**Key Dependencies**: sqlalchemy

**Python Version**: >=3.8

### `enginius_dockers`

**Purpose**: Docker container definitions and orchestration scripts.

**Description**: This package contains Docker configurations and deployment scripts for various services. Unlike other packages, this one doesn't follow the standard Python package structure (no `src/` or `pyproject.toml`). Instead, it contains:

- Docker container definitions in the `dockers/` subdirectory:
  - `ai_parser/`: AI document parser service
  - `dictionary_docker/`: Dictionary service
  - `scraper/`: Web scraping service
- Build and run scripts (`build_dockers.sh`, `run_dockers.sh`)
- Docker Compose configurations for service orchestration

**Installation**: This package is not installed via pip. Instead, use the provided shell scripts or Docker Compose to build and run the containers.

### `scraping`

**Purpose**: Web scraping scripts for regulatory documents from various government and regulatory body websites.

**Description**: Specialized web scraping modules for extracting regulatory documents and guidance from official sources. Includes scrapers for:

- **FDA (U.S. Food and Drug Administration)**:
  - CFR (Code of Federal Regulations)
  - Guidance documents
  - 510(k) submissions
- **EU Regulatory Bodies**:
  - EASA (European Union Aviation Safety Agency)
  - MDR (Medical Device Regulation) and amendments
  - MDCG (Medical Device Coordination Group) guidance
- **Other Regulatory Sources**:
  - Health Canada regulations
  - Australian guidance and regulations
  - FAA (Federal Aviation Administration) CFR
  - NIST RMF (Risk Management Framework)

**Key Dependencies**: beautifulsoup4, lxml, pandas, selenium, requests, PyPDF2

**Python Version**: >=3.11

## Development

The root directory contains a `pyproject.toml` file for repository-wide development tooling configuration (black, isort) and a `requirements.local.txt` file for local development dependencies.

To install development dependencies:

```bash
pip install -e ".[dev]"
```

## Project Structure

```
tlp-python/
├── packages/              # All Python packages
│   ├── ai_doc_parser/
│   ├── common_tools/
│   ├── database/
│   ├── enginius_dockers/
│   └── scraping/
├── data/                  # Data files (if any)
├── output/                # Output files (if any)
├── scripts/               # Utility scripts
├── pyproject.toml         # Root project configuration
├── requirements.local.txt # Local development requirements
└── README.md             # This file
```
