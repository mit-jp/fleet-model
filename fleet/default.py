# Default attributes for the fleet model

# Vehicle types or categories
#
# These are defined as a tree using nested Python dict()s; any key with a
# value of None is a leaf node of the tree, or an 'atomic' vehicle category.
categories = {
    'Car': {
        'Private car': None,
        'Non-private car': None,
        },
    'Light truck': None,
    }


# Fuel types
fuels = [
    'Gasoline',
    'E10',  # 10% ethanol, 90% gasoline
    'E85',  # 85% ethanol, 15% gasoline
    'M85',  # 85% methanol, 15% gasoline
    'Diesel',
    'Biodiesel',
    'Electricity',
    ]


# Powertrain types
powertrains = [
    'ICE NA-SI',  # Internal combustion engine, naturally-aspirated, spark
                  # ignition (i.e. gasoline fuelled)
    'Turbo-SI',   # Turbocharged SI ICE
    'Diesel',     # Diesel ICE
    'HE',         # Hybrid-electric
    'PHE',        # Plug-in hybrid electric
    'E',          # (Battery) electric
    'CNG',        # Compressed natural gas
    'FC',         # Fuel cell
    ]


# Time dimension for the model
#
# See http://pandas.pydata.org/pandas-docs/stable
#     /timeseries.html#time-span-representation
time = {
    'freq': 'A',  # Annual
    'min': 1960,  # Initial period of the model
    0: 2011,      # First period for projections
    'max': 2050,  # Last period for projections
    }
