# kaggle-otto2

## Code Structure
- [WIP] コード構成のポンチ絵を作成

## 3 Types of experiment environment
- [WIP] dev, cv, lbの説明をテーブルで作成

## Procedure

### 000. Setup poetry

```bash
# Install packages
poetry install

# Get into virtual env
poetry shell
```

### 001. Download datasets

```bash
# Download datasets
# ※ ~/kaggle/.kaggle.json with your Kaggle API Key is required
./bin/001_download.sh
```

### 002. Preprocess

```bash
./bin/002_preprocess.sh exp001_dev
```

### 003. Candidate Generation

```bash
./bin/003_preprocess.sh exp001_dev
```
