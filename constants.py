

DATDIR = "data/Dataset_A/CSV_Exported"

NCELLS = 7200

NORMALIZATION_CONSTANT = 10000  # FACS data is on the order of 10^4

GENES = ['TBX6', 'BRA', 'CDX2', 'SOX2', 'SOX1']  # required order to match gmms

GENE_TO_IDX = {g:i for g, i in zip(GENES, range(len(GENES)))}

CONDITIONS = {
    0  : "NO CHIR",
    1  : "CHIR 2-2.5",
    2  : "CHIR 2-3",
    3  : "CHIR 2-3.5",
    4  : "CHIR 2-4",
    5  : "CHIR 2-5",
    6  : "CHIR 2-5 FGF 2-3",
    7  : "CHIR 2-5 FGF 2-3.5",
    8  : "CHIR 2-5 FGF 2-4",
    9  : "CHIR 2-5 FGF 2-4.5",
    10 : "CHIR 2-5 FGF 2-5",
}

REFERENCE_CONDITIONS = [0, 2, 5]
TRAINING_CONDITIONS = [0, 2, 4, 5, 6, 8, 10]
VALIDATION_CONDITIONS = [1, 3, 7, 9]

ALL_CONDITIONS = list(range(len(CONDITIONS)))

CONDITION_SIGNALS = {  # CHIR, FGF, PD
    0  : [(-1, -1),     (0.0, 3.0),     (-1, -1)],
    1  : [(2.0, 2.5),   (0.0, 3.0),     (-1, -1)],
    2  : [(2.0, 3.0),   (0.0, 3.0),     (-1, -1)],
    3  : [(2.0, 3.5),   (0.0, 3.0),     (-1, -1)],
    4  : [(2.0, 4.0),   (0.0, 3.0),     (-1, -1)],
    5  : [(2.0, 5.0),   (0.0, 3.0),     (-1, -1)],
    6  : [(2.0, 5.0),   (0.0, 3.0),     (3.0, 5.0)],
    7  : [(2.0, 5.0),   (0.0, 3.5),     (3.5, 5.0)],
    8  : [(2.0, 5.0),   (0.0, 4.0),     (4.0, 5.0)],
    9  : [(2.0, 5.0),   (0.0, 4.5),     (4.5, 5.0)],
    10 : [(2.0, 5.0),   (0.0, 5.0),     (-1, -1)],
}

CONDITION_REFERENCE_FLAG = {
    0  : True ,
    1  : False,
    2  : True ,
    3  : False,
    4  : False,
    5  : True ,
    6  : False,
    7  : False,
    8  : False,
    9  : False,
    10 : False,
}

# Each experiment consists of 7 samples, each corresponding to a particular
# time point. Some experiments have the same signaling history up to a time t,
# so the same sample can appear in across different experiments. For example,
# all experiments had the same history at day 2, so that the sample at day 2
# is used as the first state across all conditions.
CONDITION_FILES = {
    0 : ["D2_A1_A01_002.fcs .csv",
         "D2.5_A2_A02_003.fcs .csv",
         "D3_A4_A04_005.fcs .csv",
         "D3.5_A7_A07_008.fcs .csv",
         "D4_A11_A11_012.fcs .csv",
         "D4.5_B4_B04_017.fcs .csv",
         "D5_B10_B10_023.fcs .csv",
        ],
    1 : ["D2_A1_A01_002.fcs .csv",
         "D2.5_C_A3_A03_004.fcs .csv",
         "D3_C2-2.5_A5_A05_006.fcs .csv",
         "D3.5_C2-2.5_A8_A08_009.fcs .csv",
         "D4_C2-2.5_A12_A12_013.fcs .csv",
         "D4.5_C2-2.5_B5_B05_018.fcs .csv",
         "D5_C2-2.5_B11_B11_024.fcs .csv",
        ],
    2 : ["D2_A1_A01_002.fcs .csv",
         "D2.5_C_A3_A03_004.fcs .csv",
         "D3_C2-3_A6_A06_007.fcs .csv",
         "D3.5_C2-3_A9_A09_010.fcs .csv",
         "D4_C2-3_B1_B01_014.fcs .csv",
         "D4.5_C2-3_B6_B06_019.fcs .csv",
         "D5_C2-3_B12_B12_025.fcs .csv",
        ],
    3 : ["D2_A1_A01_002.fcs .csv",
         "D2.5_C_A3_A03_004.fcs .csv",
         "D3_C2-3_A6_A06_007.fcs .csv",
         "D3.5_C2-3.5_A10_A10_011.fcs .csv",
         "D4_C2-3.5_B2_B02_015.fcs .csv",
         "D4.5_C2-3.5_B7_B07_020.fcs .csv",
         "D5_C2-3.5_C1_C01_026.fcs .csv",
        ],
    4 : ["D2_A1_A01_002.fcs .csv",
         "D2.5_C_A3_A03_004.fcs .csv",
         "D3_C2-3_A6_A06_007.fcs .csv",
         "D3.5_C2-3.5_A10_A10_011.fcs .csv",
         "D4_C2-4_B3_B03_016.fcs .csv",
         "D4.5_C2-4_B8_B08_021.fcs .csv",
         "D5_C2-4_C2_C02_027.fcs .csv",
        ],
    5 : ["D2_A1_A01_002.fcs .csv",
         "D2.5_C_A3_A03_004.fcs .csv",
         "D3_C2-3_A6_A06_007.fcs .csv",
         "D3.5_C2-3.5_A10_A10_011.fcs .csv",
         "D4_C2-4_B3_B03_016.fcs .csv",
         "D4.5_C2-4.5_B9_B09_022.fcs .csv",
         "D5_C2-5_C3_C03_028.fcs .csv",
        ],
    6 : ["D2_A1_A01_002.fcs .csv",
         "D2.5_C_A3_A03_004.fcs .csv",
         "D3_C2-3_A6_A06_007.fcs .csv",
         "D3.5_CF2-3_C4_C04_029.fcs .csv",
         "D4_CF2-3_C6_C06_031.fcs .csv",
         "D4.5_CF2-3_C9_C09_034.fcs .csv",
         "D5_CF2-3_D1_D01_038.fcs .csv",
        ],
    7 : ["D2_A1_A01_002.fcs .csv",
         "D2.5_C_A3_A03_004.fcs .csv",
         "D3_C2-3_A6_A06_007.fcs .csv",
         "D3.5_CF3.5_C5_C05_030.fcs .csv",
         "D4_CF2-3.5_C7_C07_032.fcs .csv",
         "D4.5_CF2-3.5_C10_C10_035.fcs .csv",
         "D5_CF2-3.5_D2_D02_039.fcs .csv",
        ],
    8 : ["D2_A1_A01_002.fcs .csv",
         "D2.5_C_A3_A03_004.fcs .csv",
         "D3_C2-3_A6_A06_007.fcs .csv",
         "D3.5_CF3.5_C5_C05_030.fcs .csv",
         "D4_CF2-4_C8_C08_033.fcs .csv",
         "D4.5_CF2-4_C11_C11_036.fcs .csv",
         "D5_CF2-4_D3_D03_040.fcs .csv",
        ],
    9 : ["D2_A1_A01_002.fcs .csv",
         "D2.5_C_A3_A03_004.fcs .csv",
         "D3_C2-3_A6_A06_007.fcs .csv",
         "D3.5_CF3.5_C5_C05_030.fcs .csv",
         "D4_CF2-4_C8_C08_033.fcs .csv",
         "D4.5_CF2-4.5_C12_C12_037.fcs .csv",
         "D5_CF2-4.5_D4_D04_041.fcs .csv",
        ],
    10 : ["D2_A1_A01_002.fcs .csv",
          "D2.5_C_A3_A03_004.fcs .csv",
          "D3_C2-3_A6_A06_007.fcs .csv",
          "D3.5_CF3.5_C5_C05_030.fcs .csv",
          "D4_CF2-4_C8_C08_033.fcs .csv",
          "D4.5_CF2-4.5_C12_C12_037.fcs .csv",
          "D5_CF2-5_D5_D05_042.fcs .csv",
        ],
}

# Build a sorted list of all of the experimental condition csv files
ALL_FILES = set()
for idx in CONDITION_FILES:
    for fname in CONDITION_FILES[idx]:
        ALL_FILES.add(fname)
ALL_FILES = list(ALL_FILES)
ALL_FILES.sort()

# Each file corresponds to an individual sample of 7200 cells.
# Define a 1-1 mapping to refer to each file by an index.
FNAME_TO_SAMPLEIDX = {
    fname: i for fname, i in zip(ALL_FILES, range(len(ALL_FILES)))
}
SAMPLEIDX_TO_FNAME = {
    i: fname for fname, i in zip(ALL_FILES, range(len(ALL_FILES)))
}

##############
##  Colors  ##
##############

# CHIR_COLOR = (0.57, 0.26, 0.98)  # As used in Sáez et al.
CHIR_COLOR = (0.18, 0.22, 0.73)  # Dark blue plasma for PLNN project
FGF_COLOR  = (0.70, 0.09, 0.32)  # As used in Sáez et al.
PD_COLOR   = (0.57, 0.74, 0.32)  # As used in Sáez et al.
LFGF_COLOR = (0.82, 0.50, 0.62)  # As used in Sáez et al.
WHITE      = (1.00, 1.00, 1.00)
