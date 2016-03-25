import sys

import pandas as pd
import xarray as xr

from fleet import Model

# Reproduce the Akerlind China model
m = Model()


def read_tsv(filename, dims):
    da = pd.read_csv('examples/{}.csv'.format(filename), delimiter='\t',
                     index_col=0, na_values='.', comment='#')
    return xr.DataArray(da, dims=dims)


m['sales'] = m.align('sales', read_tsv('sales', ['t', 'class']))
m.fill('sales', 't', stop='t0')
m.fill('sales', 't', dir=-1)

gr = read_tsv('sales_growth', ['t', 'class'])
m['sales_growth'] = m.align('sales_growth', gr)
m.fill('sales_growth', 't')

m.new('pcar_frac', ('class', 't'))
m['pcar_frac'] = m.align('pcar_frac', read_tsv('pcar_frac', ['t', 'class']))
m.fill('pcar_frac', 't')
m.fill('pcar_frac', 't', dir=-1)
m['pcar_frac'].loc[:, 'Non-private car'] = 1 - \
    m['pcar_frac'].loc[:, 'Private car']

m.compute('sales_ratio')
m.compute('sales')

# Compute number of private, non-private cars, total vehicles
m.disaggregate('sales', 'Car', 'pcar_frac')
m.aggregate('sales', 'Total')

# Survival rate parameters for private car, non-private car, light truck
c = {'class': ['Private car', 'Non-private car', 'Light truck']}
m['B'] = xr.DataArray([4.7, 5.33, 5.58], coords=c, dims=['class'])
m['T'] = xr.DataArray([14.46, 13.11, 8.02], coords=c, dims=['class'])

sys.exit(0)  # Refactor complete to here

m.compute('stock')

# Output, for comparison to the original model
o = m['stock'].loc['Private car', :, :].to_dataframe().unstack('model_year')
o.to_csv('pcar.csv')

# TODO add data from Stock!G49:J58
# TODO add data from Stock!G143:J153
