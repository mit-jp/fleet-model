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
    'description': 'Rate parameter for vehicle survival function',
    'units': '0',
    }

T = {
    'dims': ['class'],
    'description': 'Vehicle survival function—average vehicle life span',
    'units': 'time',
    }

stock = {
    'dims': ['class', 'tm', 't'],
    'description': 'Surviving stock of t₁ vehicles at time t₂',
    'units': 'vehicle',
    }

vdt_v = {
    'dims': ['class', 'tm', 'age'],
    'description': 'VDT per vehicle',
    'units': 'distance',
    }

vdt_v_rate = {
    'dims': ['class'],
    'description': 'Exponential rate of decrease in VDT per vehicle',
    'units': '1/time',
    }
