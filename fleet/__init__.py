import collections
import datetime
import io

import numpy as np
import pandas as pd
import xray


# Default attributes for the fleet model
DEFAULTS = {
    'y_min': 1960,  # Start year of the model
    'y_0': 2011,  # First year for projections
    'y_max': 2050,  # Last year for projections
    'classes': [  # Atomic vehicle classes
        'Private car',
        'Non-private car',
        'Light truck',
        ],
    # Categories or groups of vehicle classes, stored as an OrderedDict to
    # facilitate calculations (e.g. sums) that must be performed in sequence.
    # Keys are category names; values are tuples (i.e. ordered) containing the
    # members of the category, which may be vehicle classes OR other categories.
    # TODO extend for groups of powertrain types, groups of fuels, as needed
    'categories': collections.OrderedDict((
        ('Car', ('Private car', 'Non-private car')),
        ('Total', ('Car', 'Light truck')),
        )),
    'powertrains': [ # Powertrain types
        'ICE NA-SI', # Internal combustion engine, naturally-aspirated, spark
                     # ignition (i.e. gasoline fuelled)
        'Turbo-SI',  # Turbocharged SI ICE
        'Diesel',    # Diesel ICE
        'HE',        # Hybrid-electric
        'PHE',       # Plug-in hybrid electric
        'E',         # (Battery) electric
        'CNG',       # Compressed natural gas
        'FC',        # Fuel cell
        ],
    'fuels': [ # Fuel types
        'Gasoline',
        'E10',  # 10% ethanol, 90% gasoline
        'E85',  # 85% ethanol, 15% gasoline
        'M85',  # 85% methanol, 15% gasoline
        'Diesel',
        'Biodiesel',
        'Electricity',
        ],
    }


class MissingDataException(Exception):
    """Raised by FleetModel methods that find missing prerequisite data."""
    # TODO replace assertions below with this exception
    pass


def nans(*size):
    """Return an array of NaNs (for brevity)."""
    return np.nan * np.ones(size)


def inline(s):
    """Reading an inline text tables from string *s*.

    See below for a usage example."""
    return pd.read_csv(io.StringIO(s), delim_whitespace=True, index_col=0,
                       skip_blank_lines=True)


class FleetModel(xray.Dataset):
    """Fleet model class.

    Inherits xray.Dataset, so that the data (variables, coords, attrs)
    associated with the model may be accessed directly from instances.
    """
    def __init__(self):
        # Use default attributes
        attrs = DEFAULTS
        # TODO override defaults with constructor kwargs
        # Range of years
        year = pd.date_range(str(attrs['y_min']), str(attrs['y_max'] + 1),
                             freq='A')
        # Data arrays store vehicles classes *plus* categories
        classes = list(attrs['classes']) + list(attrs['categories'].keys())
        Ny = len(year)
        Nc = len(classes)
        # Initialize variables and coordinates
        # TODO add description attributes & units for variables; see 'T':
        xray.Dataset.__init__(self, {
            'sales': (['class', 'model_year'], nans(Nc, Ny)),
            'sales_growth': (['class', 'model_year'], nans(Nc, Ny)),
            'sales_ratio': (['class', 'model_year'], nans(Nc, Ny)),
            'B': ('class', nans(Nc)),
            'T': ('class', nans(Nc), {
                'description': 'Rate parameter for vehicle survival function',
                'units': 'years'
                }),
            'stock': (['class', 'cal_year', 'model_year'], nans(Nc, Ny, Ny)),
            'stock_total': (['class', 'cal_year'], nans(Nc, Ny)),
            'age_mean': (['class', 'cal_year'], nans(Nc, Ny)),
            },
            coords={
                'class': classes,
                'cal_year': year,
                'model_year': year,
                'powertrain': attrs['powertrains'],
                'fuel': attrs['fuels'],
            })
        # Store the initial attributes—must follow Dataset.__init__
        self.attrs.update(attrs)

    def _N(self, coord):
        """Length of any coordinate *coord*."""
        return len(self.coords[coord])

    def set_sales_growth(self, rates):
        """Fill the variable 'sales_growth' using *rates*.

        *rates* must be a pandas.DataFrame with (some) columns from 'class' and rows from 'model_year', containing growth rates for vehicle sales in percent.

        Growth rates are filled forward; e.g. if *rates* gives a value for year
        2020 and then for year 2030, the rate for year 2020 is used for years
        between 2020 and 2029 inclusive.

        'sales_ratio' is also populated.
        """
        y0 = str(self.attrs['y_0'])
        for (cl, series) in rates.iteritems():
            for (y, v) in series.iteritems():
                assert y >= self.attrs['y_0']
                self['sales_growth'].loc[cl,str(y):] = v
        self['sales_ratio'].loc[:,y0:] = ((self['sales_growth'].loc[:,y0:] *
                                          0.01) + 1).values.cumprod(axis=1)

    def project_sales(self):
        """Project future vehicle sales."""
        y0 = str(self.attrs['y_0'])
        ym1 = str(self.attrs['y_0'] - 1)
        # Product of sales in the year before y_0 and the ratio of future years'
        # sales
        self['sales'].loc[:,y0:] = self['sales'].loc[:,ym1].values * \
            self['sales_ratio'].loc[:,y0:]

    def project_stock(self):
        """Project vehicle stocks using a survival function."""
        # Vector of vehicle ages
        age = np.arange(1, 100)
        # Survival function parameters
        B = self['B'].values
        T = self['T'].values
        # Survival function
        s = xray.DataArray(np.exp(-((age[:,np.newaxis] / T) ** B)).T,
            dims=('class', 'year'), coords=(self.coords['class'], age))
        print(s)
        # Copy sales into stock variable and compute survival in same step
        Nc = self._N('class')
        Ny = self._N('model_year')
        for i in range(Ny):
            y = str(self.attrs['y_min'] + i)
            self['stock'].loc[:,y:,y] = np.expand_dims(np.floor(s[:,:(Ny-i)] *
                self['sales'].loc[:,y].values), axis=2)
        print(self['stock'].loc['Private car',:'1970',:'1970'])
        # Compute subtotals by class category Ɐ {model_year, cal_year}
        for cat in self.attrs['categories'].keys():
            self.aggregate('stock', cat)
        # Compute total stock (all model years) Ɐ {cal_year, class}
        self['stock_total'] = self['stock'].sum('model_year')
        # Compute average age, for atomic classes only
        cl = self.attrs['classes']
        for y in range(self.attrs['y_min'], self.attrs['y_max'] + 1):
            y_ = str(y)
            age = np.arange(y - self.attrs['y_min'] + 1, 0,
                            -1)[np.newaxis,:,np.newaxis]
            self['age_mean'].loc[cl,y_] = (age *
                self['stock'].loc[cl,:y_,y_]).sum('model_year') / \
                self['stock_total'].loc[cl,y_]
        # TODO compute removed vehicles
        # TODO compute removed as % of sales

    def disaggregate(self, var, cat, shares):
        """Disaggregate data for a category.

        In variable *var*, the values for *cat* are shared out to its members
        (according to attrs['categories'][cat]), using *shares*. *shares* must
        be N×1 or N×M, where N is the number of members of *cat*, and *M* is the dimension of *var*.
        """
        # TODO only tested for 'stock'; add argument checking
        assert len(shares) == len(self.attrs['categories'][cat])
        assert np.all(np.sum(shares, axis=0) == 1) # Check the shares
        for i in range(len(shares)):
            s = self.attrs['categories'][cat][i]
            m[var].loc[s,:] = m[var].loc[cat,:] * shares[i]

    def aggregate(self, var, cat, how='sum'):
        """Aggregate data within a category.

        In variable *var*, the values for *cat* are computed by summing its
        members (according to attrs['categories']).
        """
        # TODO only tested for 'stock'; add argument checking
        assert how == 'sum'
        m[var].loc[cat,:] = \
            m[var].loc[self.attrs['categories'][cat],:].sum('class')


# Reproduce the Akerlind China model
m = FleetModel()

# Historical sales
# …data, from China Automotive Industry Yearbook
m['sales'].loc[('Car','Light truck'),'1996':'2010'] = np.array([
    [  750815,  444426],
    [  879226,  442191],
    [  943151,  443255],
    [ 1044973,  521305],
    [ 1270085,  528875],
    [ 1485895,  508701],
    [ 2090014,  666714],
    [ 3106275,  819278],
    [ 3466302,  979559],
    [ 4149329, 1087026],
    [ 5368536, 1241991],
    [ 6528245, 1420307],
    [ 6973101, 1536743],
    [10556246, 2065288],
    [14042149, 2571892]]).T
# …assumed
m['sales'].loc['Car',:'1989',] = 3e5
m['sales'].loc['Car','1990':'1993'] = 4e5
m['sales'].loc['Car','1994':'1995'] = [5e5, 6e5]
m['sales'].loc['Light truck',:'1991'] = 4e5
m['sales'].loc['Light truck','1992':'1995'] = np.array([41, 42, 43, 44]) * 1e4

m.set_sales_growth(inline("""
t     Car  "Light truck"
2011  8    8
2015  4    5
2020  2    3
2030  1.5  1.5
2040  1    1
"""))

# Assumed share of private cars in cars
pf = pd.Series(index=m.coords['model_year'])
# Historical
pf[:'1984'] = 0.4
pf['1985':'1988'] = 0.45
pf['1989':'1995'] = 0.5
pf['1996':'1999'] = 0.55
pf['2000':'2007'] = [0.6, 0.65, 0.7, 0.75, 0.75, 0.8, 0.85, 0.85]
pf['2008':'2010'] = 0.9
# Future
pf['2011':'2017'] = [0.9, 0.91, 0.92, 0.92, 0.93, 0.93, 0.93]
pf['2018':'2021'] = 0.94
pf['2022':] = 0.95

m.merge({'pcar_frac': ('model_year', pf, {
    'description': 'Assumed share of "Private car" sales in "Car" sales',
    'unit': '0'})},
    inplace=True)

m.project_sales()

# Compute number of private, non-private cars, total vehicles
# TODO put this in a lambda method passed to FleetModel.project_sales()
m.disaggregate('sales', 'Car', [pf, 1 - pf])
m.aggregate('sales', 'Total')

# Survival rate parameters for private car, non-private car, light truck
m['B'].values = [ 4.7,   5.33, 5.58, np.nan, np.nan]
m['T'].values = [14.46, 13.11, 8.02, np.nan, np.nan]

m.project_stock()

# Output, for comparison to the original model
o = m['stock'].loc['Private car',:,:].to_dataframe().unstack('model_year')
o.to_csv('pcar.csv')

# TODO add data from Stock!G49:J58
# TODO add data from Stock!G143:J153
