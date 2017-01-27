# Variables for the fleet model.

__all__ = [
    'sales_growth',
    ]


sales = {
    'dims': ['category', 't'],
    'description': 'Number of vehicles sold',
    'units': 'vehicle',
    }

sales_growth = {
    'dims': ['category', 't'],
    'description': 'Period-on-period growth in *sales*',
    'units': 'percent',
    'required': False,
    }

sales_ratio = {
    'dims': ['category', 't'],
    'description': 'Ratio of *sales* in period tₙ to period t₋₁',
    'units': '0',
    }

B = {
    'dims': ['category'],
    'description': 'Rate parameter for vehicle survival function',
    'units': '0',
    }

T = {
    'dims': ['category'],
    'description': 'Vehicle survival function—average vehicle life span',
    'units': 'time',
    }

stock = {
    'dims': ['category', 'tm', 't'],
    'description': 'Surviving stock of t₁ vehicles at time t₂',
    'units': 'vehicle',
    }

vdt_v = {
    'dims': ['category', 'tm', 'age'],
    'description': 'VDT per vehicle',
    'units': 'distance',
    }

vdt_v_rate = {
    'dims': ['category'],
    'description': 'Exponential rate of decrease in VDT per vehicle',
    'units': '1/time',
    }
