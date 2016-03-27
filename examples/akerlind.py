from fleet import Model


def akerlind_data(m):
    """Read in data for the Akerlind China fleet model."""
    from pandas import read_csv
    from xarray import DataArray

    def read_tsv(filename, dims):
        """Read *filename* into a DataArray indexed by *dims*."""
        da = read_csv('examples/{}.csv'.format(filename), delimiter='\t',
                      index_col=0, na_values='.', comment='#')
        return DataArray(da, dims=dims)

    # Historical sales data
    m['sales'] = m.align('sales', read_tsv('sales', ['t', 'category']))
    # Fill gaps forward to the last historical period, then backward to the
    # earliest historical period
    m.xfill('sales', 't0')

    # Sales growth rates
    m['sales_growth'] = m.align('sales_growth',
                                read_tsv('sales_growth', ['t', 'category']))
    # Fill forward to the end of the forecast horizon
    m.ffill('sales_growth', 't')

    # This model projects sales of 'Cars'. In a new variable 'pcar_frac', store
    # the assumed shares of (non-) private cars in future sales of cars.
    m.new('pcar_frac', ('category', 't'))
    m['pcar_frac'] = m.align('pcar_frac',
                             read_tsv('pcar_frac', ['t', 'category']))
    m['pcar_frac'].loc[:, 'Non-private car'] = 1 - \
        m['pcar_frac'].loc[:, 'Private car']
    m.xfill('pcar_frac')

    # Survival rate parameters for private car, non-private car, light truck
    c = ['Private car', 'Non-private car', 'Light truck']
    m['B'].loc[c] = [4.7, 5.33, 5.58]
    m['T'].loc[c] = [14.46, 13.11, 8.02]

    # VDT per vehicle
    m['vdt_v'].loc[c, '2000', 0] = [22540, 25760, 28980]
    m['vdt_v_rate'].loc[c] = [4, 4, 4]

# Initialize the model
m = Model(akerlind_data)


def solve():
    """Solve the model."""
    m.compute('sales_ratio')
    m.compute('sales').xagg('sales', 'pcar_frac')
    m.compute('stock')
    m.compute('vdt_v')

solve()

# Output, for comparison to the original model
A = m['stock'].sel(category='Private car').to_dataframe() \
              .drop('category', axis=1).unstack('tm')

# TODO add data from Stock!G49:J58
# TODO add data from Stock!G143:J153
