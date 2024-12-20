import numpy as np

# pima_diabetes, LiNGAM background knowledge
bk_pima_matrix = np.full((9, 9), -1)
bk_pima_matrix[5, 7] = 1
bk_pima_matrix[4, 7] = 1
bk_pima_matrix[2, 7] = 1
bk_pima_matrix[1, 7] = 1
bk_pima_matrix[8, 2] = 1
bk_pima_matrix[8, 5] = 1
bk_pima_matrix[8, 6] = 1
bk_pima_matrix[8, 7] = 1

#original_sachs_bk
bk_sachs_matrix = np.zeros((11, 11))
bk_sachs_matrix[4, 2] = 1 # pip3 <-- plc
bk_sachs_matrix[3, 2] = 1 # pip2 <-- plc
bk_sachs_matrix[8, 2] = 1 # pkc <-- plc
bk_sachs_matrix[3, 4] = 1 # pip2 <-- pip3
bk_sachs_matrix[6, 4] = 1 # akt <-- pip3
bk_sachs_matrix[8, 3] = 1 # pkc <-- pip2
bk_sachs_matrix[1, 8] = 1 # mek <-- pkc
bk_sachs_matrix[0, 8] = 1 # raf <-- pkc
bk_sachs_matrix[7, 8] = 1 # pka <-- pkc
bk_sachs_matrix[9, 8] = 1 # p38 <-- pkc
bk_sachs_matrix[10, 8] = 1 # jnk <-- pkc
bk_sachs_matrix[0, 7] = 1 # raf <-- pka
bk_sachs_matrix[1, 7] = 1 # mek <-- pka
bk_sachs_matrix[5, 7] = 1 # erk <-- pka
bk_sachs_matrix[6, 7] = 1 # akt <-- pka
bk_sachs_matrix[9, 7] = 1 # p38 <-- pka
bk_sachs_matrix[10, 7] = 1 # jnk <-- pka
bk_sachs_matrix[1, 0] = 1 # mek <-- raf
bk_sachs_matrix[5, 1] = 1 # erk <-- mek
bk_sachs_matrix[6, 5] = 1 # akt <-- erk

#original_sachs_pkc_bk
bk_sachs_pkc_matrix = np.zeros((11, 11))
bk_sachs_pkc_matrix[4, 2] = 1 # pip3 <-- plc
bk_sachs_pkc_matrix[3, 2] = 1 # pip2 <-- plc
bk_sachs_pkc_matrix[10, 2] = 1 # pkc <-- plc
bk_sachs_pkc_matrix[3, 4] = 1 # pip2 <-- pip3
bk_sachs_pkc_matrix[6, 4] = 1 # akt <-- pip3
bk_sachs_pkc_matrix[10, 3] = 1 # pkc <-- pip2
bk_sachs_pkc_matrix[1, 10] = 1 # mek <-- pkc
bk_sachs_pkc_matrix[0, 10] = 1 # raf <-- pkc
bk_sachs_pkc_matrix[7, 10] = 1 # pka <-- pkc
bk_sachs_pkc_matrix[8, 10] = 1 # p38 <-- pkc
bk_sachs_pkc_matrix[9, 10] = 1 # jnk <-- pkc
bk_sachs_pkc_matrix[0, 7] = 1 # raf <-- pka
bk_sachs_pkc_matrix[1, 7] = 1 # mek <-- pka
bk_sachs_pkc_matrix[5, 7] = 1 # erk <-- pka
bk_sachs_pkc_matrix[6, 7] = 1 # akt <-- pka
bk_sachs_pkc_matrix[8, 7] = 1 # p38 <-- pka
bk_sachs_pkc_matrix[9, 7] = 1 # jnk <-- pka
bk_sachs_pkc_matrix[1, 0] = 1 # mek <-- raf
bk_sachs_pkc_matrix[5, 1] = 1 # erk <-- mek
bk_sachs_pkc_matrix[6, 5] = 1 # akt <-- erk
