import cpmpy as cp
import numpy as np
from cpmpy import cpm_array
from pycona import ConstraintOracle
from pycona import ProblemInstance, absvar

def construct_uefa(teams_data, n_groups=8, teams_per_group=4):
    """
    Construct a UEFA Champions League group stage problem instance and oracle.
    
    :param teams_data: Dictionary containing team information (name, country, coefficient)
    :param n_groups: Number of groups (default 8)
    :param teams_per_group: Number of teams per group (default 4)
    :return: a ProblemInstance object, along with a constraint-based oracle
    """
    # Create a dictionary with the parameters
    parameters = {
        "n_groups": n_groups,
        "teams_per_group": teams_per_group,
        "n_teams": len(teams_data)
    }

    # Create variables for group assignments
    group_assignments = {}
    for team in teams_data:
        group_assignments[team] = cp.intvar(0, n_groups-1, name=f"group{team}")
 

    model = cp.Model()
    
    # Store all constraints
    all_constraints = []
    # Store only global constraints
    global_constraints = []

    # 1. Group size global constraint - using sum and equality
    for g in range(n_groups):
        group_size = sum(group_assignments[team] == g for team in teams_data)
        constraint = (group_size == teams_per_group)
        all_constraints.append(constraint)
        global_constraints.append(constraint)  # This is a global constraint (sum)

    # 2. Teams from same country must be in different groups (using AllDifferent)
    countries = {}
    for team, data in teams_data.items():
        country = data['country']
        if country not in countries:
            countries[country] = []
        countries[country].append(team)

    for country, country_teams in countries.items():
        if 1 < len(country_teams) <= 4:  # Only apply AllDifferent for 2-4 teams from same country
            country_vars = [group_assignments[team] for team in country_teams]
            constraint = cp.AllDifferent(country_vars).decompose()
            all_constraints.append(constraint)
            global_constraints.append(constraint)  # This is a global constraint (AllDifferent)

    # 3. Coefficient-based seeding - teams are divided into pots based on coefficients (using AllDifferent)
    teams_by_coefficient = sorted(teams_data.items(), key=lambda x: x[1]['coefficient'], reverse=True)
    for pot_idx in range(0, len(teams_data), n_groups):
        pot_end = min(pot_idx + n_groups, len(teams_data))
        pot_teams = teams_by_coefficient[pot_idx:pot_end]
        if len(pot_teams) > 1:
            pot_vars = [group_assignments[team[0]] for team in pot_teams]
            constraint = cp.AllDifferent(pot_vars).decompose()
            all_constraints.append(constraint)
            global_constraints.append(constraint)  # This is a global constraint (AllDifferent)

    # 4. Binary constraints - rival teams preference
    # Add binary constraints between specific team pairs to avoid certain group combinations
    team_names = list(teams_data.keys())
    for i in range(len(team_names)):
        for j in range(i+1, len(team_names)):
            team1, team2 = team_names[i], team_names[j]
            
            # Teams from neighboring countries should ideally be in different groups
            if (teams_data[team1]['country'] in ['Spain', 'Portugal'] and 
                teams_data[team2]['country'] in ['Spain', 'Portugal']):
                constraint = (group_assignments[team1] != group_assignments[team2])
                all_constraints.append(constraint)
            
            # Top coefficient teams should be spread across groups
            if (teams_data[team1]['coefficient'] > 90 and 
                teams_data[team2]['coefficient'] > 90):
                constraint = (group_assignments[team1] != group_assignments[team2])
                all_constraints.append(constraint)
            
            # Teams with similar coefficients should be in different groups (binary constraint)
            coeff_diff = abs(teams_data[team1]['coefficient'] - teams_data[team2]['coefficient'])
            if coeff_diff < 5:  # Very similar coefficients
                constraint = (group_assignments[team1] != group_assignments[team2])
                all_constraints.append(constraint)

    # # 4. Match day alternation - consecutive rounds must be on different days
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

    # 6. Match count constraints - each team must play exactly once per round (global constraint)
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

    # # # 7. Group match constraints - each group must have exactly two matches per round (global constraint)
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

    # Add all constraints to the model
    for constraint in all_constraints:
        model += constraint

    # Create the language for the oracle:
    AV = absvar(2)  # create abstract vars - as many as maximum arity

    # Create abstract relations using the abstract vars - only global constraints
    lang = [
        sum(AV) == AV[0],     # Sum equal to
        sum(AV) <= AV[0],     # Sum less than or equal
        sum(AV) >= AV[0],     # Sum greater than or equal
    ]

    # Create variables list in the correct order
    variables = []
    # First add group assignment variables
    for team in sorted(teams_data.keys()):
        variables.append(group_assignments[team])
    # Then add match day variables
    # variables.extend(match_days)
    # # Then add match variables in order
    # for r in range(6):
    #     for g in range(n_groups):
    #         variables.append(match_variables[(r, g, 1)]['home'])
    #         variables.append(match_variables[(r, g, 1)]['away'])
    #         variables.append(match_variables[(r, g, 2)]['home'])
    #         variables.append(match_variables[(r, g, 2)]['away'])

    instance = ProblemInstance(variables=cpm_array(variables), params=parameters, language=lang, name="uefa")
    oracle = ConstraintOracle(global_constraints)  # Use only the global constraints

    return instance, oracle 