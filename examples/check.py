# Extract data from the reference model spreadsheet for comparison

import pandas as pd

ranges = {
    ('Stock', 8, 'M:CY'): [
        ((8, 98), ('t', 'tm'), 'stock_pcar', 1e6),
        ((302, 392), ('t', 'tm'), 'stock_pcar_removed', 1e6),
        ((400, 490), ('t', 'tm'), 'stock_pcar_nonsurviving', 1e6),
        ],
    ('Stock', 8, 'DF:GR'): [
        ((8, 98), ('t', 'tm'), 'stock_npcar', 1e6),
        ((302, 392), ('t', 'tm'), 'stock_npcar_removed', 1e6),
        ((400, 490), ('t', 'tm'), 'stock_npcar_nonsurviving', 1e6),
        ],
    ('Stock', 8, 'GX:KJ'): [
        ((8, 98), ('t', 'tm'), 'stock_truck', 1e6),
        ((302, 392), ('t', 'tm'), 'stock_truck_removed', 1e6),
        ((400, 490), ('t', 'tm'), 'stock_truck_nonsurviving', 1e6),
        ],
    }

coords = {
    't': range(1960, 2051),
    'tm': range(1960, 2051),
    }

for r, tables in ranges.items():
    sheet, row, cols = r
    df = pd.read_excel('akerlind.xlsx', sheetname=sheet, skiprows=row - 1,
                       parse_cols=cols, header=None)
    for t in tables:
        rows, dims, filename, k = t
        sl = slice(rows[0] - row, rows[1] - row)
        data = df.loc[sl, :] * k
        data.index = coords[dims[0]]
        data.index.name = dims[0]
        data.columns = coords[dims[1]]
        data.columns.name = dims[1]
        data.to_csv('check/{}.csv'.format(filename), sep='\t')
