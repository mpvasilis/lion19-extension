import cpmpy as cp

roster = cp.intvar(1, 8, shape=(7, 3, 2), name='var')
model = cp.Model()

# Add constraints
for d in range(7):
    model += cp.AllDifferent(roster[d, ...])

for d in range(6):
    model += cp.AllDifferent(roster[d, 2], roster[d+1, 0])

for n in range(1, 9):
    model += cp.Count(roster, n) <= 5

print(f'Base nurse rostering SAT: {model.solve()}')
print(f'Number of constraints: {len(model.constraints)}')
