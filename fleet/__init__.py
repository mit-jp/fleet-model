import datetime

import numpy as np
import pandas as pd
import xray

# Indices
y_min = 1960
y_0 = 2011
y_max = 2050

year = pd.date_range(str(y_min), str(y_max + 1), freq='A')
N_y = len(year)

veh_class = ['Total', 'Car', 'Private car', 'Non-private car', 'Light truck']
pt = ['ICE NA-SI', 'Turbo-SI', 'Diesel', 'HE', 'PHE', 'E', 'CNG', 'FC']
fuel = ['Gasoline', 'E10', 'E85', 'M85', 'Diesel', 'Biodiesel', 'Electricity']

vc = ('Private car', 'Non-private car', 'Light truck')
N_c = len(veh_class)

def nans(*size):
    """Return an array of NaNs (for brevity)."""
    return np.nan * np.ones(size)

# TODO add attributes to record long descriptions, units for existing variables
ds = xray.Dataset({
    'sales': (['model_year', 'class'], nans(N_y, N_c)),
    'stock': (['model_year', 'cal_year', 'class'], nans(N_y, N_y, N_c)),
    'stock_total': (['cal_year', 'class'], nans(N_y, N_c)),
    'age_mean': (['cal_year', 'class'], nans(N_y, N_c)),
    },
    coords={
        'class': veh_class,
        'cal_year': year,
        'model_year': year,
        'powertrain': pt,
        'fuel': fuel,
        # TODO add 'age', possibly?
        }
    )

# Historical sales
# …data, from China Automotive Industry Yearbook
ds['sales'].loc['1996':'2010',('Car','Light truck')] = [
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
    [14042149, 2571892]]
# …assumed
ds['sales'].loc[:'1989','Car'] = 3e5
ds['sales'].loc['1990':'1993','Car'] = 4e5
ds['sales'].loc['1994':'1995','Car'] = [5e5, 6e5]
ds['sales'].loc[:'1991','Light truck'] = 4e5
ds['sales'].loc['1992':'1995','Light truck'] = np.array([41, 42, 43, 44]) * 1e4

# Assumed annual sales growth [%]
sales_growth = pd.DataFrame(index=year, columns=['Car', 'Light truck'])
sales_growth.loc['2011',:] = [8, 8]
sales_growth.loc['2015',:] = [4, 5]
sales_growth.loc['2020',:] = [2, 3]
sales_growth.loc['2030',:] = [1.5, 1.5]
sales_growth.loc['2040',:] = [1, 1]
sales_growth.fillna(method='ffill', inplace=True)

# Ratio of sales to 2010 level
sales_ratio = (1 + (sales_growth * 0.01)).cumprod()

# Compute future sales
ds['sales'].loc['2011':,('Car', 'Light truck')] = ds['sales'].loc['2010',('Car', 'Light truck')].values * sales_ratio['2011':]

# Assumed share of private cars in cars
pcar_frac = pd.Series(index=year)
# Historical
pcar_frac[:'1984'] = 0.4
pcar_frac['1985':'1988'] = 0.45
pcar_frac['1989':'1995'] = 0.5
pcar_frac['1996':'1999'] = 0.55
pcar_frac['2000':'2007'] = [0.6, 0.65, 0.7, 0.75, 0.75, 0.8, 0.85, 0.85]
pcar_frac['2008':'2010'] = 0.9
# Future
pcar_frac['2011':'2017'] = [0.9, 0.91, 0.92, 0.92, 0.93, 0.93, 0.93]
pcar_frac['2018':'2021'] = 0.94
pcar_frac['2022':] = 0.95

ds.merge({'pcar_frac': ('model_year', pcar_frac)}, inplace=True)

# Compute number of private, non-private cars, total vehicles
ds['sales'].loc[:,'Private car'] = pcar_frac * ds['sales'].loc[:,'Car']
ds['sales'].loc[:,'Non-private car'] = (1 - pcar_frac) * ds['sales'].loc[:,'Car']
ds['sales'].loc[:,'Total'] = ds['sales'].loc[:,('Car', 'Light truck')].sum('class')

# Survival rate parameters
ds.update({   # Private car, non-private car, light truck
    'B': ('class', [np.nan, np.nan,  4.7,   5.33, 5.58]),
    'T': ('class', [np.nan, np.nan, 14.46, 13.11, 8.02]),
    })

# Survival function
age = np.arange(1, 100)
s = xray.DataArray(
    np.exp(-((age[:,np.newaxis] / ds['T'].values) ** ds['B'].values)),
    dims=('year', 'class'),
    coords=(age, veh_class),
    )

# Copy sales into stock variable, and compute survival in the same step
for i in range(N_y):
    y = str(y_min + i)
    ds['stock'].loc[y,y:,:] = np.floor(s[:(91 - i),:] * ds['sales'].loc[y,:].values)

# Compute subtotal of cars, total vehicles Ɐ {model_year, cal_year}
ds['stock'].loc[:,:,'Car'] = ds['stock'].loc[:,:,('Private car',
    'Non-private car')].sum('class')
ds['stock'].loc[:,:,'Total'] = ds['stock'].loc[:,:,vc].sum('class')

# Compute total stock (all model years) Ɐ {cal_year, class}
ds['stock_total'] = ds['stock'].sum('model_year')

# commented: output for comparison
#ds['stock'].loc[:,:,'Private car'].to_dataframe().unstack('model_year').to_csv('pcar.csv')
#ds['stock'].loc[:,:,'Car'].to_dataframe().unstack('model_year').to_csv('car.csv')

# Compute average age
for y in range(y_min, y_max + 1):
    y_ = str(y)
    age = np.arange(y - y_min + 1, 0, -1)[:,np.newaxis,np.newaxis]
    ds['age_mean'].loc[y_,vc] = (age *
        ds['stock'].loc[:y_,y_,vc]).sum('model_year') / \
        ds['stock_total'].loc[y_,vc]

# TODO compute removed vehicles, removed as % of sales
# TODO add Stock!G49:J58
# TODO add Stock!G143:J153
