# Variables for the fleet model.

sales = {
    'dims': ['class', 't'],
    'description': 'Number of vehicles sold',
    'units': 'vehicle',
    }

sales_growth = {
    'dims': ['class', 't'],
    'description': 'Period-on-period growth in *sales*',
    'units': 'percent',
    'required': False,
    }

sales_ratio = {
    'dims': ['class', 't'],
    'description': 'Ratio of *sales* in period tₙ to period t₋₁',
    'units': '0',
    }

B = {
    'dims': ['class'],
    'description': 'Parameter for vehicle survival function',
    'units': '0',
    }

T = {
    'dims': ['class'],
    'description': 'Rate parameter for vehicle survival function',
    'units': 'time',
    }

stock = {
    'dims': ['class', 'tm', 't'],
    'description': 'Surviving stock of t₁ vehicles at time t₂',
    'units': 'vehicle',
    }
