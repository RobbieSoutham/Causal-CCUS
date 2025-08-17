"""
causalccus.constants
====================
Frozen mappings used across modules.
"""

POLICY_DATES = {
    "Paris Agreement": 2015,
    "US 45Q Expansion": 2018,
    "India NCAP, EU CCS Directive, UK Net Zero Law": 2019,
    "US Inflation Reduction Act": 2022,
    "EU ETS Phase IV, China National ETS": 2021,
}

FIXED_EFFECTS = [
    'Country',
    'CCUS_sector',
    'Sector',
    'emissions_sector'
]

ANALYTIC_VARS = [
    'Demand_electricity',
    'Demand_heat',
    'Demand_nuclear',
    'Demand_renewables_and_waste',
    'Supply_nuclear',
    'renewable_to_fossil_supply_ratio',
    'energy_demand_fossil_fuels',
    'CPI_growth',
    'GDP_per_capita_PPP',
    #'d_log_total_capacity'
]

FIGURE_DIR = 'C:/Users/rjsou/Documents/PhD/Causal-CCUS/Notebooks/Figures/Final/'