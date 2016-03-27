import numpy as np
import pandas as pd
import xarray as xr

from fleet.category import Category
import fleet.default as default
import fleet.variables


class MissingDataException(Exception):
    """Raised by FleetModel methods that find missing prerequisite data."""
    # TODO replace assertions below with this exception
    pass


class Model(xr.Dataset):
    """Fleet model class.

    Inherits xray.Dataset, so that the data (variables, coords, attrs)
    associated with the model may be accessed directly from instances.
    """
    def __init__(self, init_data):
        # Data arrays store vehicles classes *plus* categories
        cat_tree = Category('Total', default.categories)
        # atomic = cat_tree.leaves()
        categories = cat_tree.nodes()

        # Coordinates
        coords = {
            't': pd.period_range(default.time['min'], default.time['max'],
                                 freq=default.time['freq']),
            'tm': pd.period_range(default.time['min'], default.time['max'],
                                  freq=default.time['freq']),
            'category': categories,
            'powertrain': default.powertrains,
            'fuel': default.fuels,
            }
        coords['age'] = range(len(coords['t']) + 1)

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
            'cat tree': cat_tree,
            't0': pd.Period(default.time[0], freq=default.time['freq']),
            }

        xr.Dataset.__init__(self, variables, coords, attrs=attrs)

        # Other attributes that could not be precomputed
        self.attrs['t+'] = self.t.where(self.t >= self.attrs['t0']).dropna('t')

        # Initialize data
        init_data(self)

    def compute(self, var):
        """Compute the contents of *var*."""
        if var not in ['sales', 'sales_ratio', 'stock', 'vdt_v']:
            raise ValueError(var)
        getattr(self, '_' + var)()
        return self

    def _sales_ratio(self):
        """Compute sales_ratio."""
        tp = self.attrs['t+']
        x = self['sales_growth'].sel(t=tp)
        axis = x.get_axis_num('t')
        x.values = (1 + 0.01 * x.values).cumprod(axis=axis)
        self['sales_ratio'].loc[:, tp] = x.T

    def _sales(self):
        """Compute sales."""
        tp = self.attrs['t+']
        # Product of sales in the year before y_0 and the ratio of future
        # years' sales
        x, _ = xr.align(self['sales'].sel(t=self.attrs['t0'] - 1) *
                        self['sales_ratio'].loc[:, tp],
                        self['sales'], join='right')
        self['sales'] += x.fillna(0)

    def _stock(self):
        """Compute vehicle stocks."""
        # Survival function
        age = self['age'].values[:, np.newaxis]
        B = self['B'].values
        T = self['T'].values
        self['__s'] = (('age', 'category'), np.exp(-((age / T) ** B)))
        # Copy sales into stock variable & compute survival
        for t in self.t.values:
            tp = self.t.where(self.t >= t).dropna('t')
            stock_t = self['sales'].sel(t=t) * self['__s']
            self['stock'].loc[:, t, tp] = np.floor(stock_t[:, :len(tp)])
        return  # Refactor complete to here
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

    def _vdt_v(self):
        """Compute VDT per vehicle."""
        pass

    def xagg(self, var, shares):
        dims = list(self[var].dims)
        dims.remove('category')
        ct = self.attrs['cat tree']
        check = self[var].isnull().all(dims)
        for c in check:
            missing = c.item()
            if not missing:
                continue
            cat = c.coords['category'].item()
            children = ct.find(cat).children()
            if len(children) and not check.loc[children].any():
                # Can aggregate this category from its children
                self[var].loc[dict(category=cat)] = \
                    self[var].loc[dict(category=children)].sum('category')
                continue
            parent = ct.find(cat).parent()
            if parent is not None and not check.loc[parent]:
                # Can disaggregate this category from its parent
                self[var].loc[dict(category=cat)] = \
                    self[var].sel(category=parent) * \
                    self[shares].sel(category=cat)

    def disaggregate(self, var, cat, shares):
        """Disaggregate data for a category.

        In variable *var*, the values for *cat* are shared out to its members
        (according to attrs['categories'][cat]), using *shares*. *shares* must
        be N×1 or N×M, where N is the number of members of *cat*, and *M* is
        the dimension of *var*.
        """
        x = self[var].loc[:, cat] * self[shares]
        sub = x.coords['category'] \
               .where(np.any(np.isfinite(x.values), axis=0)).dropna('category')
        self[var].loc[:, sub] = x.sel(category=sub)

    def aggregate(self, var, cat, how='sum', weights=None):
        """Aggregate data within a category.

        In variable *var*, the values for *cat* are computed by summing its
        members (according to attrs['categories']).
        """
        # TODO only tested for 'stock'; add argument checking
        assert how == 'sum' and weights is None
        sub = self.attrs['cat tree'].find(cat).children()
        self[var].loc[:, cat] = self[var].loc[:, sub].sum('category')

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

    def fill(self, var, dim, reverse=False, stop=None):
        """Fill the variable *var* along dimension *dim*."""
        x = self[var].values
        axis = self[var].get_axis_num(dim)
        if stop == 't0':
            stop = self.attrs['t0']
        fill_values = x.take(0, axis)
        it = enumerate(self[dim])
        if reverse:
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

    def ffill(self, var, dim, stop=None):
        self.fill(var, dim, stop)

    def bfill(self, var, dim, stop=None):
        self.fill(var, dim, reverse=True, stop=stop)

    def round(self, var):
        self[var] = self[var].round()

    def xfill(self, var, stop=None):
        self.fill(var, 't', stop)
        self.fill(var, 't', reverse=True)
