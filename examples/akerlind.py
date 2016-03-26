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
    m['sales'] = m.align('sales', read_tsv('sales', ['t', 'class']))
    # Fill gaps forward to the last historical period
    m.ffill('sales', 't', 't0')
    # Fill backwards to the earliest historical period
    m.bfill('sales', 't')

    # Sales growth rates
    m['sales_growth'] = m.align('sales_growth',
                                read_tsv('sales_growth', ['t', 'class']))
    # Fill forward to the end of the forecast horizon
    m.ffill('sales_growth', 't')

    # This model projects sales of 'Cars'. In a new variable 'pcar_frac', store
    # the assumed shares of (non-) private cars in future sales of cars.
    m.new('pcar_frac', ('class', 't'))
    m['pcar_frac'] = m.align('pcar_frac',
                             read_tsv('pcar_frac', ['t', 'class']))
    m['pcar_frac'].loc[:, 'Non-private car'] = 1 - \
        m['pcar_frac'].loc[:, 'Private car']
    m.ffill('pcar_frac', 't')
    m.bfill('pcar_frac', 't')

    # Survival rate parameters for private car, non-private car, light truck
    c = {'class': ['Private car', 'Non-private car', 'Light truck']}
    m['B'] = DataArray([4.7, 5.33, 5.58], coords=c)
    m['T'] = DataArray([14.46, 13.11, 8.02], coords=c)


# Initialize the model
m = Model(akerlind_data)

def solve():
    """Solve the model."""
    m.compute('sales_ratio')

    # Sales
    m.compute('sales')
    # Compute number of private, non-private cars, total vehicles
    m.disaggregate('sales', 'Car', 'pcar_frac')
    m.aggregate('sales', 'Total')

    m.compute('stock')

solve()

# Output, for comparison to the original model
A = m['stock'].sel(**{'class': 'Private car'}).to_dataframe() \
              .drop('class', axis=1).unstack('tm')

# TODO add data from Stock!G49:J58
# TODO add data from Stock!G143:J153
