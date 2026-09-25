# Full public datasets

These folders contain the original tabular datasets, including normal and attack
records. They use the original schemas and splits, which differ from the coursework
files in `../nsl-kdd/` and `../unsw-nb15/`. Raw packet captures are not included.

## NSL-KDD

Directory: `nsl-kdd/`

| File | Records | Contents |
| --- | ---: | --- |
| `KDDTrain+.txt` | 125,973 | Full training split |
| `KDDTest+.txt` | 22,544 | Full test split |
| `KDDTrain+_20Percent.txt` | 25,192 | Training subset |
| `KDDTest-21.txt` | 11,850 | Test subset excluding difficulty 21 |

The `.txt` files are comma-separated, without headers. Each has 41 features,
an attack-type label (`normal` for normal traffic), and a difficulty value.
Matching `.arff` files contain attribute names and binary normal/anomaly labels.
The subset files and ARFF versions are alternative views, not additional records
to concatenate with the full training and test splits.

Original source: [University of New Brunswick](https://www.unb.ca/cic/datasets/nsl.html).
UNB states that its download is no longer available. Files were retrieved from
[Zenodo record 3768048](https://zenodo.org/records/3768048).

Dataset reference: M. Tavallaee, E. Bagheri, W. Lu, and A. A. Ghorbani,
"A Detailed Analysis of the KDD CUP 99 Data Set", IEEE CISDA, 2009.
UNB permits redistribution and requires citation of the dataset and paper.

## UNSW-NB15

Directory: `unsw-nb15/`

The complete table is split across `UNSW-NB15_1.csv` through
`UNSW-NB15_4.csv`. The first three files each contain 700,001 rows; the fourth
contains 440,044. The last row of each file is repeated as the first row of
the next file. Together they contain 2,540,047 rows, or the published
2,540,044 records after removing those three overlapping boundary rows.
The downloaded files preserve those overlaps. When combining them, keep all
rows of file 1 and skip the first row of files 2, 3, and 4.

These files have no header and contain 49 columns, including `attack_cat`
and the binary `Label`.
Blank attack categories occur in normal records; the binary label is `0` for
normal traffic and `1` for attacks.

`NUSW-NB15_features.csv` describes the column order, types, and meanings.
Its filename preserves the spelling in the mirror. Also included are the
event list and the smaller `UNSW_NB15_training-set.csv` and
`UNSW_NB15_testing-set.csv` files. These smaller files have headers and a
different schema; do not append them to the four full-data files.

| Downloaded subset filename | Records | Columns |
| --- | ---: | ---: |
| `UNSW_NB15_training-set.csv` | 82,332 | 45 |
| `UNSW_NB15_testing-set.csv` | 175,341 | 45 |

These are the verified counts in the downloaded files. Their training/testing
filenames are reversed relative to the sizes described on the UNSW webpage.
The filenames are preserved as provided by the mirror.

Original source: [UNSW](https://research.unsw.edu.au/projects/unsw-nb15-dataset).
Its download link redirected to a Microsoft login when accessed.
The four full-data files were retrieved from
[Zenodo record 10140548](https://zenodo.org/records/10140548).
The feature definitions, event list, and smaller splits came from a pinned
revision of the
[GitHub mirror](https://github.com/jamshaid120/UNSW_NB15-Complete-dataset).

Dataset reference: N. Moustafa and J. Slay, "UNSW-NB15: a comprehensive data
set for network intrusion detection systems (UNSW-NB15 network data set)",
IEEE MilCIS, 2015. UNSW grants free academic research use and requests the
citations listed on its dataset page; commercial use requires agreement
from the authors.

## Download records

`manifest.json` records the download URLs, source checksums, byte counts,
and local SHA-256 hashes. Downloads are checked against the published Zenodo
MD5 checksums or the GitHub LFS SHA-256 checksums. The two data directories
are excluded by the project's `.gitignore`.
