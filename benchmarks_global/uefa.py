import cpmpy as cp
import numpy as np
from cpmpy import cpm_array
from pycona import ConstraintOracle
from pycona import ProblemInstance, absvar

def construct_uefa(teams_data, n_groups=8, teams_per_group=4):
    parameters = {
        "n_groups": n_groups,
        "teams_per_group": teams_per_group,
        "n_teams": len(teams_data)
    }

    group_assignments = {}
    for team in teams_data:
        group_assignments[team] = cp.intvar(0, n_groups-1, name=f"group_{team}")
 

    model = cp.Model()
    
    all_constraints = []
    global_constraints = []

    # 1. Group size global constraint
    for g in range(n_groups):
        group_size = cp.Count([group_assignments[team] for team in teams_data], g)
        constraint = (group_size == teams_per_group)
        all_constraints.append(constraint)
        # all_constraints.append(constraint.decompose())
        global_constraints.append(constraint) 

    countries = {}
    for team, data in teams_data.items():
        country = data['country']
        if country not in countries:
            countries[country] = []
        countries[country].append(team)

    for country, country_teams in countries.items():
        if 1 < len(country_teams) <= 4:  # AllDifferent for 2-4 teams from same country
            country_vars = [group_assignments[team] for team in country_teams]
            constraint = cp.AllDifferent(country_vars)
            all_constraints.append(constraint)
            # all_constraints.append(constraint.decompose())
            global_constraints.append(constraint) 

    # 3. teams are divided into pots based on coefficients
    teams_by_coefficient = sorted(teams_data.items(), key=lambda x: x[1]['coefficient'], reverse=True)
    for pot_idx in range(0, len(teams_data), n_groups):
        pot_end = min(pot_idx + n_groups, len(teams_data))
        pot_teams = teams_by_coefficient[pot_idx:pot_end]
        if len(pot_teams) > 1:
            pot_vars = [group_assignments[team[0]] for team in pot_teams]
            constraint = cp.AllDifferent(pot_vars)
            all_constraints.append(constraint)
            # all_constraints.append(constraint.decompose())

            global_constraints.append(constraint)  

    # # 4. consecutive rounds must be on different days
    # for r in range(5):  # 6 rounds total, so 5 transitions
    #     constraint = (match_days[r] != match_days[r + 1])
    #     all_constraints.append(constraint)
    #     # This is a binary constraint, not a global constraint

    # # 5. Match constraints
    # for r in range(6):
    #     for g in range(n_groups):
    #         # Teams can't play against themselves
    #         constraint1 = (match_variables[(r, g, 1)]['home'] != match_variables[(r, g, 1)]['away'])
    #         constraint2 = (match_variables[(r, g, 2)]['home'] != match_variables[(r, g, 2)]['away'])
    #         all_constraints.append(constraint1)
    #         all_constraints.append(constraint2)
    #         # These are binary constraints, not global constraints
            
    #         # Teams can't play in multiple matches in the same round
    #         constraint3 = (match_variables[(r, g, 1)]['home'] != match_variables[(r, g, 2)]['home'])
    #         constraint4 = (match_variables[(r, g, 1)]['away'] != match_variables[(r, g, 2)]['away'])
    #         constraint5 = (match_variables[(r, g, 1)]['home'] != match_variables[(r, g, 2)]['away'])
    #         constraint6 = (match_variables[(r, g, 1)]['away'] != match_variables[(r, g, 2)]['home'])
    #         all_constraints.append(constraint3)
    #         all_constraints.append(constraint4)
    #         all_constraints.append(constraint5)
    #         all_constraints.append(constraint6)
    #         # These are binary constraints, not global constraints

    # 6. each team must play exactly once per round (global constraint)
    # for r in range(6):
    #     for team_idx in range(len(teams_data)):
    #         # Count home matches
    #         home_matches = sum(
    #             match_variables[(r, g, m)]['home'] == team_idx 
    #             for g in range(n_groups) 
    #             for m in (1, 2)
    #         )
    #         # Count away matches
    #         away_matches = sum(
    #             match_variables[(r, g, m)]['away'] == team_idx 
    #             for g in range(n_groups) 
    #             for m in (1, 2)
    #         )
    #         # Each team plays exactly one match per round
    #         constraint = (home_matches + away_matches == 1)
    #         all_constraints.append(constraint)
    #         global_constraints.append(constraint)  # This is a global constraint (sum)

    # # # 7. each group must have exactly two matches per round (global constraint)
    # for r in range(6):
    #     for g in range(n_groups):
    #         # We have exactly 2 matches per group per round (fixed by design)
    #         # This is a constant constraint (2 == 2), so we don't need to add it
    #         # Instead, we'll add constraints to ensure teams in matches belong to the correct group
    #         for m in (1, 2):
    #             home_team = match_variables[(r, g, m)]['home']
    #             away_team = match_variables[(r, g, m)]['away']
                
    #             # For each team in teams_data, create a constraint that if they're playing in this match,
    #             # they must be assigned to this group
    #             for team_idx, team_name in enumerate(teams_data.keys()):
    #                 # If this team is the home team in this match, it must be in this group
    #                 # Use logical OR for implication: (not A) OR B
    #                 home_constraint = (~(home_team == team_idx) | (group_assignments[team_name] == g))
    #                 # If this team is the away team in this match, it must be in this group
    #                 away_constraint = (~(away_team == team_idx) | (group_assignments[team_name] == g))
                    
    #                 all_constraints.append(home_constraint)
    #                 all_constraints.append(away_constraint)
    #                 # These are binary constraints (implications), not global constraints

    # MOCK OVER-FITTED CONSTRAINTS (will be consistent with 5 examples but NOT generally valid)
    mock_constraints = []

    # Mock constraints must be feasible with n_groups
    # Strategy: Create AllDifferent constraints on subsets of size min(n_groups, 4)
    mock_size = min(n_groups, 4)

    if len(teams_data) >= mock_size * 2:
        # Mock 1: First mock_size teams must be in different groups (too restrictive)
        first_teams = sorted(teams_data.keys())[:mock_size]
        first_vars = [group_assignments[team] for team in first_teams]
        mock_c1 = cp.AllDifferent(first_vars)
        mock_constraints.append(mock_c1)
        all_constraints.append(mock_c1)
        global_constraints.append(mock_c1)

        # Mock 2: Last mock_size teams must be in different groups (too restrictive)
        last_teams = sorted(teams_data.keys())[-mock_size:]
        last_vars = [group_assignments[team] for team in last_teams]
        mock_c2 = cp.AllDifferent(last_vars)
        mock_constraints.append(mock_c2)
        all_constraints.append(mock_c2)
        global_constraints.append(mock_c2)

    for constraint in all_constraints:
        model += constraint

    AV = absvar(2)  # create abstract vars - as many as maximum arity

    # Create abstract relations using the abstract vars - only global constraints
    lang = [
        sum(AV) == AV[0],    
        sum(AV) <= AV[0],   
        sum(AV) >= AV[0],     
    ]

    variables = []
    for team in sorted(teams_data.keys()):
        variables.append(group_assignments[team])
    # variables.extend(match_days)
    # for r in range(6):
    #     for g in range(n_groups):
    #         variables.append(match_variables[(r, g, 1)]['home'])
    #         variables.append(match_variables[(r, g, 1)]['away'])
    #         variables.append(match_variables[(r, g, 2)]['home'])
    #         variables.append(match_variables[(r, g, 2)]['away'])

    instance = ProblemInstance(variables=cpm_array(variables), params=parameters, language=lang, name="uefa")
    oracle = ConstraintOracle(global_constraints) 

    return instance, oracle 

def generate_uefa_instance(instance_params=None):
  
    if instance_params is None:
        instance_params = {}
    
    n_groups = instance_params.get('n_groups', 8)
    teams_per_group = instance_params.get('teams_per_group', 4)
    n_teams = n_groups * teams_per_group
    
    teams_data = {}
    countries = ['England', 'Spain', 'Germany', 'Italy', 'France', 'Portugal', 'Netherlands', 'Russia']
    
    while len(countries) < n_teams // 4 + 1:
        countries.append(f"Country_{len(countries) + 1}")
    
    for i in range(n_teams):
        country_idx = i % len(countries)
        team_name = f"Team_{i+1}"
        teams_data[team_name] = {
            'country': countries[country_idx],
            'coefficient': 100 - i
        }
    
    return construct_uefa(teams_data, n_groups, teams_per_group)