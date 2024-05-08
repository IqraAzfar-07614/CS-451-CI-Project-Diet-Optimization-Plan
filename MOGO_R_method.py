import random
import csv
import matplotlib.pyplot as plt
import numpy as np

# this code contains the MOGO R implementation inspired EA

# Define constants
POPULATION_SIZE = 200
MUTATION_RATE = 0.6
NUM_GENERATIONS = 2000
ETA_C = 1.0  # Distribution index for SBX
TARGET_PROTEIN = 0
TARGET_CARBS = 0
TARGET_FAT = 0
WEIGHTS = np.array([0.3, 0.3, 0.4])
preference = []
all_foods = []


# Load data
data_file = []
path = "CI project ingredients-dataset - ingredients-dataset.csv.csv"
with open(path, newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip header
    for row in reader:
        food_name = row[1]
        all_foods.append((row[0], food_name))  # Store food number and name
        data_file.append(row[1:])  # Exclude the first column
    print(all_foods)

data_file_name = "athletes_data.csv"

def load_data():
    data_athlete = []
    with open(data_file_name, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for row in reader:
            data_athlete.append(row)
    return data_athlete

def update_targets(athlete, gender, age_group):
    data_athlete = load_data()
    for row in data_athlete:
        if row[0] == athlete and row[1] == gender and row[2] == age_group:
            target_protein = int(row[3])
            target_carbs = int(row[4])
            target_fats = int(row[5])
            return target_protein, target_carbs, target_fats
    return None, None, None

def ask_user_for_info():
    print("Here are the preferences for ingredients:")
    for i, item in all_foods:
        print(f"{i}\t{item}")

    print("\nEnter the numbers corresponding to your preferred ingredients separated by commas (e.g., 1, 5, 10):")
    user_choices = input("Your choices: ").strip().split(',')
    preference = []
    for choice in user_choices:
        try:
            index = int(choice.strip())
            if 0 <= index < len(all_foods):
                preference.append(all_foods[index][1])
            else:
                print(f"Ignore invalid choice: {choice}")
        except ValueError:
            print(f"Ignore non-integer choice: {choice}")

    if not preference:
        print("No valid choices selected. Please try again.")
        return ask_user_for_info()

    print("Which of the following athletes are you? (1. Boxers, 2. Runners, 3. Swimmers, 4. Cyclists)")
    athlete_map = {"1": "Boxers", "2": "Runners", "3": "Swimmers", "4": "Cyclists"}
    athlete_choice = input("Enter the number corresponding to your choice: ").strip()
    athlete = athlete_map.get(athlete_choice)
    if not athlete:
        print("Invalid input. Please choose a valid number.")
        return ask_user_for_info()

    print("Choose your gender: (1. Male, 2. Female)")
    gender_map = {"1": "Male", "2": "Female"}
    gender_choice = input("Enter the number corresponding to your choice: ").strip()
    gender = gender_map.get(gender_choice)
    if not gender:
        print("Invalid input. Please choose a valid number.")
        return ask_user_for_info()

    print("Choose your age group: (1. 20-30, 2. 30-40, 3. 40-50, 4. 50+)")
    age_group_map = {"1": "20-30", "2": "30-40", "3": "40-50", "4": "50+"}
    age_group_choice = input("Enter the number corresponding to your choice: ").strip()
    age_group = age_group_map.get(age_group_choice)
    if not age_group:
        print("Invalid input. Please choose a valid number.")
        return ask_user_for_info()

    return preference, athlete, gender, age_group


preference, athlete, gender, age_group = ask_user_for_info()
TOTAL_INGREDIENTS = len(preference)  # Update TOTAL_INGREDIENTS after user input

# Load data related to the athlete
TARGET_PROTEIN, TARGET_CARBS, TARGET_FAT = update_targets(athlete, gender, age_group)

if TARGET_PROTEIN is None:
    print("Could not find matching athlete, gender, and age group combination.")
    exit()

print("Target Protein:", TARGET_PROTEIN)
print("Target Carbohydrates:", TARGET_CARBS)
print("Target Fats:", TARGET_FAT)

data = []
for i in data_file:
    if i[0] in preference:
        data.append(i)


# Calculate maximum units for each ingredient
def calculate_maximum_units(food_data, target_protein, target_fat, target_carbs):
    protein, fat, carbs = map(float, food_data[2:5])
    max_units = float('inf')
    if protein >= 0.01:
        max_units = min(max_units, target_protein / protein)
    if fat >= 0.01:
        max_units = min(max_units, target_fat / fat)
    if carbs >= 0.01:
        max_units = min(max_units, target_carbs / carbs)
    return int(max_units)


def calculate_food_intake(data_index_number, unit):

    # print(data[data_index_number][0]," is the name of the item")
    ingredient_info = data[data_index_number]
    protein, fat, carbs = map(float, ingredient_info[2:5])

    total_protein = unit * protein
    total_fat = unit*fat
    total_carbs = unit*carbs

    return total_protein, total_fat, total_carbs


max_limits = []
ingredietns = []
# print(data)
for item in data:
    if item[0] in preference:
        ingredietns.append(item[0])

    max_units = calculate_maximum_units(item, TARGET_PROTEIN, TARGET_FAT, TARGET_CARBS)
    max_limits.append(max_units)

print(ingredietns)
# print("Maximum units for each ingredient:", max_limits)


def initialize_population():
    # return [[random.randint(0, max_limits[i]) for i in range(len(data))] for _ in range(POPULATION_SIZE)]

    population = []

    for i  in range(POPULATION_SIZE):
        choromosome = [0] * len(data)
        protein = 0
        fats = 0
        carbs = 0
        max= 4

        numb_dishes = TOTAL_INGREDIENTS

        for _ in range(numb_dishes):
        # while True:
            random_num = random.randint(0,len(data)-1) # index number which was generated randomly ->ingredient
            # print(random_num)

            # max_limits[random_num] -> max unit we can take for the ingredient
            # divided by 2 so that we can have a more diverse set of ingredients to be taken for the required
            # levels of protein, fats and carbs
            # this would also allow a more effiecient crossover as the same unit has a better chance to be utlised
            # by an another ingredient thus allowing an explorative nature to it

            #  0 - max/2   any value can be taken within this range
            # unit = random.randint(0, int(max_limits[random_num]/3))
            unit = 1

            p,f,c = calculate_food_intake(random_num, unit)

            if protein+p > TARGET_PROTEIN or fats+f > TARGET_FAT or carbs+c > TARGET_CARBS:
                break

            protein += p
            fats += f
            carbs+=c

            choromosome[random_num] += unit  #updating the chromosome with the limit

        population.append(choromosome)

    return population

def sbx_crossover(parent1, parent2):
    child1, child2 = [], []
    for i in range(len(parent1)):
        if random.random() > 0.5:
            # Ensure children are within bounds
            p1, p2 = sorted([parent1[i], parent2[i]])
            beta = (2.0 * random.random())**(1.0/(ETA_C+1)) if random.random() < 0.5 else (1/(2.0 * random.random()))**(1.0/(ETA_C+1))
            child1.append(int(min(max_limits[i], max(0, 0.5*((1+beta)*p1 + (1-beta)*p2)))))
            child2.append(int(min(max_limits[i], max(0, 0.5*((1-beta)*p1 + (1+beta)*p2)))))
        else:
            child1.append(parent1[i])
            child2.append(parent2[i])
    return child1, child2


def two_point_crossover(parent1, parent2):
    # Ensure the chromosome length is more than 1 to perform crossover
    if len(parent1) > 1:
        # Generate two random crossover points
        cp1, cp2 = sorted(random.sample(range(len(parent1)), 2))
        # Create new offspring by swapping segments between cp1 and cp2
        child1 = parent1[:cp1] + parent2[cp1:cp2] + parent1[cp2:]
        child2 = parent2[:cp1] + parent1[cp1:cp2] + parent2[cp2:]
    else:
        # If chromosome length is 1, crossover does not change the chromosomes
        child1, child2 = parent1[:], parent2[:]

#     return [child1, child2]

def uniform_crossover(parent1, parent2):
    child1, child2 = [], []
    for i in range(len(parent1)):
        if random.random() > 0.5:
            child1.append(parent1[i])
            child2.append(parent2[i])
        else:
            child1.append(parent2[i])
            child2.append(parent1[i])
    return [child1, child2]


# this crossover is more suitable for instances when the selcted indexes are very large. This often causes a lot of ingredients to have low values (or even stay at 0)
# and not incrememnet even with various crossovers
# this method aims to tackle that by summing it up whcih would allow even the low value indexes to be incremented allowing us to be a little more
# exploitative as well with our working
def new_crossOver(parent1, parent2):
    child1, child2  = parent1[:], parent2[:]
    cp_child1 = random.randint(1,TOTAL_INGREDIENTS-1)
    cp_child2 = random.randint(1,TOTAL_INGREDIENTS-1)

    for i in range(cp_child1, TOTAL_INGREDIENTS):
        child1[i] += parent2[i]

    for i in range(cp_child2, TOTAL_INGREDIENTS):
        child2[i] += parent1[i]

    return child1, child2


# if the item at an index is to chosen to be mutated
# it generates a random number from 0 to the max limit that ingredient can have for the specific index
# could be changed and tested by doing it only once and not based on the number of ingredients that we have

def mutate(chromosome):
    for i in range(len(chromosome)):
        if random.random() < MUTATION_RATE:
            chromosome[i] = random.randint(0, max_limits[i])
    return chromosome


def fitness(chromosome):

    '''
    finding the deviation in the fitness between the choromosome and the required targets
    where infinity is assigned if the value is not reached
    
    '''

    TARGETS = np.array([TARGET_PROTEIN, TARGET_FAT, TARGET_CARBS])  # Targets for protein, fat, and carbs
    chromosomes_protein = 0
    chromosomes_fat = 0
    chromosomes_carbs = 0


        # Calculate the total protein, fat, and carbs for the current chromosome
    for i in range(len(data)):
        ingred_protein, ingred_fats, ingred_carbs = calculate_food_intake(i, chromosome[i])
        chromosomes_protein += ingred_protein
        chromosomes_fat += ingred_fats
        chromosomes_carbs += ingred_carbs

    if chromosomes_protein + 10 > TARGET_PROTEIN or chromosomes_fat+10 > TARGET_FAT or chromosomes_carbs+10 > TARGET_CARBS:
        chromosomes_protein, chromosomes_fat, chromosomes_carbs = float('inf'),float('inf'),float('inf')

    chromosomes_total = np.array([chromosomes_protein, chromosomes_fat, chromosomes_carbs])

     # Compute deviations and ranks
    deviations = np.abs(chromosomes_total - TARGETS)

    return deviations


def calculate_pareto_dominance(population_totals):
    '''
    This function calculates Pareto dominance scores for each individual in the population based on their fitness deviations.
    It iterates over each individual's fitness deviations (population_totals) and assigns a Pareto dominance score of 0 if the individual is penalized (i.e., if any of its fitness values are infinity).
    It then compares each individual to every other individual in the population to determine dominance relationships.
    If an individual is dominated by any other individual (i.e., there exists another individual that is better in at least one objective and not worse in any other objective), its Pareto dominance score is set to 0.
    The function returns an array of Pareto dominance scores for the population.
    '''
    pareto_scores = np.ones(len(population_totals))
    for i, totals_i in enumerate(population_totals):
        if np.isinf(totals_i).any():  # Check if current individual is penalized
            pareto_scores[i] = 0
            continue
        for j, totals_j in enumerate(population_totals):
            if i != j and not np.isinf(totals_j).any() and np.all(totals_j <= totals_i) and np.any(totals_j < totals_i):
                pareto_scores[i] = 0
                break
    return pareto_scores

def evaluate_population(population):

    '''
    This function evaluates the entire population by calculating composite scores for each individual, considering both their Pareto dominance and their deviation from the target values.
    It first calculates the fitness deviations for each individual in the population using the fitness function.
    Then, it computes the Pareto dominance scores for the population using the calculate_pareto_dominance function.
    Next, it calculates normalized ranks based on the fitness deviations to represent the relative positions of individuals in the population.
    Weighted scores are computed by multiplying the normalized ranks with weights, indicating the importance of each objective.
    Finally, it computes the composite scores by multiplying the weighted scores with the Pareto dominance scores, representing a combination of fitness and dominance.
    The function returns the population array along with the computed composite scores for each individual.
    '''
    deviations = np.array([fitness(individual) for individual in population])
    pareto_scores = calculate_pareto_dominance(deviations)
    normalized_ranks = 1 - (np.argsort(np.argsort(deviations, axis=0), axis=0) / float(len(population)))
    weighted_scores = np.dot(normalized_ranks, WEIGHTS)
    composite_scores = weighted_scores * pareto_scores
    return population, composite_scores


def fitness_proportional_selection(population, scores, size):
    # print(scores)
    total_score = np.sum(scores)
    if total_score == 0:
        selection_probs = np.ones(len(scores)) / len(scores)
    else:
        selection_probs = scores / total_score

    selected_indices = np.random.choice(len(population), size=size, replace=True, p=selection_probs)
    return [population[i] for i in selected_indices]

def binary_tournament_selection(population, scores):
    selected = []
    while len(selected) < 2:
        ind1, ind2 = random.sample(range(len(population)), 2)
        selected.append(population[ind1] if scores[ind1] > scores[ind2] else population[ind2])
    return selected[0], selected[1]

def truncation_selection(population, scores, size):
    sorted_indices = np.argsort(-scores)
    return [population[i] for i in sorted_indices[:size]]

# 1, 2, 5, 6, 9, 12, 53, 66, 24, 65, 90

# Genetic algorithm execution

population = initialize_population()

def genetic_algorithm():
    population = initialize_population()
    population, scores = evaluate_population(population)

    for _ in range(NUM_GENERATIONS):
        new_population = population[:]
        while len(new_population) < POPULATION_SIZE:
            ind1, ind2 = fitness_proportional_selection(population, scores, 2)
            child1, child2 = new_crossOver(ind1, ind2)
            new_population.extend([mutate(child1), mutate(child2)])

        population, scores = evaluate_population(new_population)
        population = truncation_selection(population, scores, POPULATION_SIZE)

    best_solution = population[0]
    return best_solution

a = genetic_algorithm()
    # print the chromosome
    # for i in population[:10]:
    #     print(i)

best_solution = a

for i in population[:10]:
    print(fitness(i), i)


for chromosome in population[:10]:
    print("Chromosome:", chromosome)
    print("Diet Plan:")
    chromosome_total_protein = 0
    chromosome_total_fat = 0
    chromosome_total_carbs = 0
    for index, unit in enumerate(chromosome):
        if unit > 0:
            food_item_name = data[index][0]
            total_protein, total_fat, total_carbs = calculate_food_intake(index, unit)
            print(f" - {food_item_name}: {unit} units, {total_protein}g protein, {total_fat}g fat, {total_carbs}g carbohydrates")
            # Accumulate total nutrients for each chromosome
            chromosome_total_protein += total_protein
            chromosome_total_fat += total_fat
            chromosome_total_carbs += total_carbs
    print("Total Protein for Chromosome:", chromosome_total_protein, "g")
    print("Total Fat for Chromosome:", chromosome_total_fat, "g")
    print("Total Carbohydrates for Chromosome:", chromosome_total_carbs, "g")
    print()

# Extracting data for the last printed chromosome
chromosome = population[0]

# Initialize lists to store daily intake of protein, fats, and carbs
daily_protein = []
daily_fats = []
daily_carbs = []

# Calculate daily intake for each nutrient
for index, unit in enumerate(chromosome):
    if unit > 0:
        protein, fats, carbs = calculate_food_intake(index, unit)
        daily_protein.append(protein)
        daily_fats.append(fats)
        daily_carbs.append(carbs)

# Convert lists to numpy arrays
daily_protein = np.array(daily_protein)
daily_fats = np.array(daily_fats)
daily_carbs = np.array(daily_carbs)

# Total required nutrients
total_protein_required = TARGET_PROTEIN
total_fats_required = TARGET_FAT
total_carbs_required = TARGET_CARBS

# Calculate total actual intake
total_actual_protein = np.sum(daily_protein)
total_actual_fats = np.sum(daily_fats)
total_actual_carbs = np.sum(daily_carbs)

# Plotting bar chart for target and actual intake
categories = ['Protein', 'Fats', 'Carbs']
targets = [total_protein_required, total_fats_required, total_carbs_required]
actual_intake = [total_actual_protein, total_actual_fats, total_actual_carbs]

x = np.arange(len(categories))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, targets, width, label='Target Intake')
rects2 = ax.bar(x + width/2, actual_intake, width, label='Actual Intake')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Intake (g)')
ax.set_title('Target vs Actual Nutrient Intake')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()

# Displaying y-axis values above the bars
for rect1, rect2 in zip(rects1, rects2):
    height1 = rect1.get_height()
    height2 = rect2.get_height()
    ax.text(rect1.get_x() + rect1.get_width() / 2, height1, f'{height1:.2f}', ha='center', va='bottom')
    ax.text(rect2.get_x() + rect2.get_width() / 2, height2, f'{height2:.2f}', ha='center', va='bottom')

plt.show()

# Calculate percentages of total required nutrients
protein_percentage = (np.sum(daily_protein) / total_protein_required) * 100
fats_percentage = (np.sum(daily_fats) / total_fats_required) * 100
carbs_percentage = (np.sum(daily_carbs) / total_carbs_required) * 100
# Creating a pie chart for nutrient distribution
labels = ['Protein', 'Fats', 'Carbs']
sizes = [protein_percentage, fats_percentage, carbs_percentage]
colors = ['#ff9999','#66b3ff','#99ff99']
explode = (0.1, 0, 0)  # explode 1st slice (Protein)

plt.figure(figsize=(8, 6))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('Nutrient Distribution')
plt.show()