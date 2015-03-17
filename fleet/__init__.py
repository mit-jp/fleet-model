import pandas as pd
import xray

h_years = pd.date_range('1960', '2011', freq='A')
veh_class = ['Car', 'Private car', 'Light truck']

# Historical sales
h_sales = xray.DataArray(
    np.nan * np.ones([len(h_years), len(veh_class)]),
    coords=(h_years, veh_class),
    dims=('year', 'class'))

# Historical sales data from China Automotive Industry Yearbook
h_sales.loc['1996':,('Car','Light truck')] = [
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
    [14042149, 2571892],
    ]

# Assumed historical sales
h_sales.loc[:'1989','Car'] = 3e5
h_sales.loc['1990':'1993','Car'] = 4e5
h_sales.loc['1994':'1995','Car'] = [5e5, 6e5]
h_sales.loc[:'1991','Light truck'] = 4e5
h_sales.loc['1992':'1995','Light truck'] = np.array([41, 42, 43, 44]) * 1e4

# Assumed historical share of private cars in cars:
p = pd.Series(index=h_years)
p[:'1984'] = 0.4
p['1985':'1988'] = 0.45
p['1989':'1995'] = 0.5
p['1996':'1999'] = 0.55
p['2000':'2007'] = [0.6, 0.65, 0.7, 0.75, 0.75, 0.8, 0.85, 0.85]
p['2008':] = 0.9

h_sales.loc[:,'Private car'] = p * h_sales.loc[:,'Car']

# Future sales
p_years = pd.date_range('2011', '2051', freq='A')

# Assumed annual sales growth, cars [%]
# TODO add assumptions for light trucks
g = pd.Series(index=p_years)
g['2011'] = 8   # Main!D33
g['2015'] = 4   # Main!D34
g['2020'] = 2   # Main!D35
g['2030'] = 1.5 # Main!D36
g['2040'] = 1   # Main!D37
g.fillna(method='ffill', inplace=True)

# Projected sales
sales = xray.DataArray(
    np.nan * np.ones([len(p_years), len(veh_class)]),
    coords=(p_years, veh_class),
    dims=('year', 'class'))

# TODO comment
sales.loc[:,'Car'] = int(h_sales.loc['2010','Car']) * (1 + (g * 0.01)).cumprod()
sales
