import pandas as pd
import xarray as xr

import fleet


def test_accessor():
    model = xr.Dataset()
    model.fleet


def test_init():
    model = xr.Dataset()
    model.fleet.init()


def test_load():
    model = xr.Dataset()
    model.fleet.init()
    model.fleet.load('sales_growth', '../../examples/sales_growth.csv')
