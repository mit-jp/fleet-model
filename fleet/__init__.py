import numpy as np
import pandas as pd
import xarray as xr

import fleet.default as default
import fleet.variables


class MissingDataException(Exception):
    """Raised by FleetModel methods that find missing prerequisite data."""
    # TODO replace assertions below with this exception
    pass


# Functions for querying the class tree
def nodes(root):
    if root is None:
        return
    for node, subtree in root.items():
        yield node
        for item in nodes(subtree):
            yield item


def leaves(root):
    for node, subtree in root.items():
        if subtree is None:
            yield node
        else:
            for item in leaves(subtree):
                yield item


def tree_find(root, label):
    """Return the subtree of *root* with *label* at its head."""
    for node, subtree in root.items():
        if node == label:
            return subtree
        elif subtree is not None:
            return tree_find(subtree, label)


def subcategories(root, label):
    return list(tree_find(root, label).keys())


class Model(xr.Dataset):
    """Fleet model class.

    Inherits xray.Dataset, so that the data (variables, coords, attrs)
    associated with the model may be accessed directly from instances.
    """
    def __init__(self):
        # Data arrays store vehicles classes *plus* categories
        class_tree = {'Total': default.classes}
        # atomic = list(leaves(class_tree))
        classes = list(nodes(class_tree))

        # Coordinates
        coords = {
            't': pd.period_range(default.time['min'], default.time['max'],
                                 freq=default.time['freq']),
            'class': classes,
            'powertrain': default.powertrains,
            'fuel': default.fuels,
            }

        # Variables
        variables = {}
        for v in filter(lambda x: not x.startswith('__'),
                        vars(fleet.variables)):
            attrs = getattr(fleet.variables, v)
            dims = attrs.pop('dims')
            variables[v] = (dims,
                            np.nan * np.ones([len(coords[d]) for d in dims]),
                            attrs)

        # Attributes
        attrs = {
            'class tree': class_tree,
            't0': pd.Period(default.time[0], freq=default.time['freq'])
            }

        xr.Dataset.__init__(self, variables, coords, attrs=attrs)

        # Other attributes that could not be precomputed
        self.attrs['t+'] = self.t.where(self.t >= self.attrs['t0']).dropna('t')

    def compute(self, var):
        if var == 'sales_ratio':
            self._sales_ratio()
        elif var == 'sales':
            self._sales()
        elif var == 'stock':
            self._stock()

    def _sales_ratio(self):
        tp = self.attrs['t+']
        x = self['sales_growth'].sel(t=tp)
        axis = x.get_axis_num('t')
        x.values = (1 + 0.01 * x.values).cumprod(axis=axis)
        self['sales_ratio'].loc[:, tp] = x.T

    def _sales(self):
        """Project future vehicle sales."""
        tp = self.attrs['t+']
        # Product of sales in the year before y_0 and the ratio of future
        # years' sales
        a, b = xr.align(self['sales'],
                        self['sales'].sel(t=self.attrs['t0'] - 1) *
                        self['sales_ratio'].loc[:, tp],
                        join='outer')
        self['sales'] = a.fillna(0) + b.fillna(0)

    def _stock(self):
        """Project vehicle stocks using a survival function."""
        # Vector of vehicle ages
        age = np.arange(1, 100)
        # Survival function parameters
        B = self['B'].values
        T = self['T'].values
        # Survival function
        s = xr.DataArray(np.exp(-((age[:, np.newaxis] / T) ** B)).T,
                         dims=('class', 'year'),
                         coords=(self.coords['class'], age))
        # Copy sales into stock variable and compute survival in same step
        Ny = len(self.t)
        for i in range(Ny):
            y = str(self.attrs['y_min'] + i)
            self['stock'].loc[:, y:, y] = np.expand_dims(np.floor(
                s[:, :(Ny-i)] * self['sales'].loc[:, y].values), axis=2)
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
                            -1)[np.newaxis, :, np.newaxis]
            self['age_mean'].loc[cl, y_] = \
                (age * self['stock'].loc[cl, :y_, y_]).sum('model_year') / \
                self['stock_total'].loc[cl, y_]
        # TODO compute removed vehicles
        # TODO compute removed as % of sales

    def disaggregate(self, var, cat, shares):
        """Disaggregate data for a category.

        In variable *var*, the values for *cat* are shared out to its members
        (according to attrs['categories'][cat]), using *shares*. *shares* must
        be N×1 or N×M, where N is the number of members of *cat*, and *M* is
        the dimension of *var*.
        """
        x = self[var].loc[:, cat] * self[shares]
        sub = x.coords['class'].where(np.any(np.isfinite(x.values), axis=0)) \
               .dropna('class')
        self[var].loc[:, sub] = x.sel(**{'class': sub})

    def aggregate(self, var, cat, how='sum', weights=None):
        """Aggregate data within a category.

        In variable *var*, the values for *cat* are computed by summing its
        members (according to attrs['categories']).
        """
        # TODO only tested for 'stock'; add argument checking
        assert how == 'sum' and weights is None
        sub = subcategories(self.attrs['class tree'], cat)
        self[var].loc[:, cat] = self[var].loc[:, sub].sum('class')

    def align(self, var, other):
        var = self[var]
        other = xr.DataArray(other)
        for i in range(len(var.dims)):
            d = var.dims[i]
            if d in ['t', 't_model']:
                freq = var.coords[d].values[0].freq
                other[d] = list(map(lambda t: pd.Period(str(t), freq=freq),
                                    other.coords[d].values))
        return other

    def new(self, name, dims):
        self[name] = (dims, np.nan * np.ones([len(self[d]) for d in dims]))

    def fill(self, var, dim, dir=1, stop=None):
        """Fill the variable *var* along dimension *dim*."""
        x = self[var].values
        axis = self[var].get_axis_num(dim)
        if stop == 't0':
            stop = self.attrs['t0']
        fill_values = x.take(0, axis)
        it = enumerate(self[dim])
        if dir == -1:
            it = reversed(list(it))
        for i, k in it:
            if k == stop:
                break
            slice = x.take(i, axis)
            fill_values = np.where(np.isfinite(slice), slice, fill_values)
            idx0 = list(np.nonzero(np.isnan(slice)))
            idx1 = idx0.copy()
            idx0.insert(axis, [i])
            x[np.ix_(*idx0)] = fill_values[np.ix_(*idx1)]
