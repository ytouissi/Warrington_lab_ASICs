# ASIC5 Expression Analyses

This repository contains scripts for analyzing ASIC1, ASIC2, and ASIC5 expression in human brain tissue from the Aging, Dementia and Traumatic Brain Injury Study and the SEA-AD single-nucleus RNA-sequencing dataset.

The analyses include regional bulk expression, SEA-AD cell-type expression, SEA-AD pseudobulk expression, and cell-type comparisons.

## Repository structure

```text
ASICs Aging, TBI Cohort/
SEAAD ASIC5 CPS/
SEAAD ASICs Cell type Expression/
SEAAD ASICs Pseudobulk/
config.json
SEAAD_MTG_RNAseq_final-nuclei.2024-02-13.h5ad
```

## Requirements

Install Python 3.10 or newer and the required packages.

```bash
pip install numpy pandas scipy statsmodels patsy matplotlib anndata h5py xlsxwriter openpyxl
```

## Configuration

The configuration file is `config.json`.

Set the genes and cell types to be analyzed in this file.

```json
{
  "gene_list": [
    "ASIC1",
    "ASIC2",
    "ASIC5"
  ],
  "cell_type": [
    "Pax6",
    "Sncg",
    "Vip",
    "L2/3 IT"
  ]
}
```

## Aging, Dementia and TBI Study

The regional bulk-expression analysis uses log2-transformed expression files from the Allen Institute Aging, Dementia and Traumatic Brain Injury Study.

The data for `ASIC1`, `ASIC2`, and `ASIC5` have already been downloaded and are present in this GitHub repository.

The files are organized as follows.

```text
ASICs Aging, TBI Cohort/
│
├── ASIC1/
│   └── log2 expression data
│
├── ASIC2/
│   └── log2 expression data
│
├── ASIC5/
│   └── log2 expression data
│
└── bulk_regional.py
```

The folders are named exactly `ASIC1`, `ASIC2`, and `ASIC5` because the script uses the folder names to identify the genes.

The regional data include

- hippocampus
- parietal cortex
- temporal cortex
- forebrain white matter

## SEA-AD data

The SEA-AD analyses use the processed Middle Temporal Gyrus single-nucleus RNA-sequencing dataset.

Download the file from

```text
https://sea-ad-single-cell-profiling.s3.amazonaws.com/MTG/RNAseq/previous_objects/SEAAD_MTG_RNAseq_final-nuclei.2024-02-13.h5ad
```

Place the downloaded file in the main repository folder.

The file is located as follows.

```text
Warrington_Lab_ASICs/
│
├── ASICs Aging, TBI Cohort/
├── SEAAD ASIC5 CPS/
├── SEAAD ASICs Cell type Expression/
├── SEAAD ASICs Pseudobulk/
├── config.json
└── SEAAD_MTG_RNAseq_final-nuclei.2024-02-13.h5ad
```

## Running the analyses

Run the regional bulk-expression analysis from the `ASICs Aging, TBI Cohort` folder.

```bash
python bulk_regional.py
```

Run the SEA-AD cell-type expression analysis from the `SEAAD ASICs Cell type Expression` folder.

```bash
python cell_type_expression.py
```

Run the SEA-AD pseudobulk analysis from the `SEAAD ASICs Pseudobulk` folder.

```bash
python Pseudobulk.py
```