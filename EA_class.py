import random
import csv
import numpy as np
import matplotlib.pyplot as plt

# this code contains the EA implementation using classes and this is being used in our dashboard

class EvolutionaryAlgorithm:
    def __init__(self, population_size, mutation_rate, num_generations, eta_c):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.num_generations = num_generations
        self.eta_c = eta_c

        self.preference = []
        self.number_preference = []

        self.all_foods = []
        self.data = []
        self.max_limits = []
        self.daily_protein = []
        self.daily_fats = []
        self.daily_carbs = []
        self.best_lst = []
        self.calories = 0
        self.target_protein = 0
        self.target_carbs = 0
        self.target_fat = 0
        self.max_days = 0

        self.population = []

    def load_athlete_data(self, path):
        data_athlete = []
        with open(path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header
            for row in reader:
                data_athlete.append(row)
        return data_athlete


    def load_food_data(self, path):
        print("path is ", path)
        data_file = []
        with open(path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header
            for row in reader:
                food_name = row[1]
                self.all_foods.append((row[0], food_name))  # Store food number and name
                data_file.append(row[1:])  # Exclude the first column
        return data_file

    def update_targets(self, athlete, gender, age_group, data_athlete):
        for row in data_athlete:
            if row[0] == athlete and row[1] == gender and row[2] == age_group:
                self.calories = int(row[6])
                self.target_protein = int(row[3])
                self.target_carbs = int(row[4])
                self.target_fat = int(row[5])
                return
        raise ValueError("Athlete not found")

    def calculate_maximum_units(self, food_data):
        protein, fat, carbs = map(float, food_data[2:5])
        max_units = float('inf')
        if protein >= 0.01:
            max_units = min(max_units, self.target_protein / protein)
        if fat >= 0.01:
            max_units = min(max_units, self.target_fat / fat)
        if carbs >= 0.01:
            max_units = min(max_units, self.target_carbs / carbs)
        return int(max_units)

    def calculate_food_intake(self, data_index_number, unit):
        ingredient_info = self.data[data_index_number]
        protein, fat, carbs = map(float, ingredient_info[2:5])

        total_protein = unit * protein
        total_fat = unit * fat
        total_carbs = unit * carbs

        return total_protein, total_fat, total_carbs

# changes done in intialize
    def initialize_population(self, athlete, gender, age_group, preference):
        path2 = "C:/Users/user/Desktop/spring 2024/CI/project CI frontend attempt 2/athletes_data.csv"
        data_athlete = self.load_athlete_data(path2)
        self.update_targets(athlete, gender, age_group, data_athlete)
        self.preference = preference[:]
        

        print(self.preference, "is the current self.preference")
        for i in self.data_file:
            if i[0] in self.preference:
                self.data.append(i)
        # print(self.data)

        for item in self.data:
            max_units = self.calculate_maximum_units(item)
            self.max_limits.append(max_units)

        global population
        population = []
        for _ in range(self.population_size):
            chromosome = [0] * len(self.data)
            protein = 0
            fats = 0
            carbs = 0
            numb_dishes = len(self.preference)

            for _ in range(numb_dishes):
                random_num = random.randint(0, len(self.data) - 1)
                unit = 1
                p, f, c = self.calculate_food_intake(random_num, unit)

                if protein + p > self.target_protein or fats + f > self.target_fat or carbs + c > self.target_carbs:
                    break

                protein += p
                fats += f
                carbs += c

                chromosome[random_num] += unit

            population.append(chromosome)
        # print(population)
        return population

    def sbx_crossover(self, parent1, parent2):
        child1, child2 = [], []
        for i in range(len(parent1)):
            if random.random() > 0.5:
                # Ensure children are within bounds
                p1, p2 = sorted([parent1[i], parent2[i]])
                beta = (2.0 * random.random())**(1.0/(self.eta_c+1)) if random.random() < 0.5 else (1/(2.0 * random.random()))**(1.0/(self.eta_c+1))
                child1.append(int(min(self.max_limits[i], max(0, 0.5*((1+beta)*p1 + (1-beta)*p2)))))
                child2.append(int(min(self.max_limits[i], max(0, 0.5*((1-beta)*p1 + (1+beta)*p2)))))
            else:
                child1.append(parent1[i])
                child2.append(parent2[i])
        return [child1, child2]

    def two_point_crossover(self, parent1, parent2):
        if len(parent1) > 1:
            cp1, cp2 = sorted(random.sample(range(len(parent1)), 2))
            child1 = parent1[:cp1] + parent2[cp1:cp2] + parent1[cp2:]
            child2 = parent2[:cp1] + parent1[cp1:cp2] + parent2[cp2:]
        else:
            child1, child2 = parent1[:], parent2[:]
        return [child1, child2]

    def uniform_crossover(self, parent1, parent2):
        child1, child2 = [], []
        for i in range(len(parent1)):
            if random.random() > 0.5:
                child1.append(parent1[i])
                child2.append(parent2[i])
            else:
                child1.append(parent2[i])
                child2.append(parent1[i])
        return [child1, child2]

    def new_crossOver(self, parent1, parent2):
        child1, child2  = parent1[:], parent2[:]
        TOTAL_INGREDIENTS = len(parent1) -1
        cp_child1 = random.randint(1,TOTAL_INGREDIENTS-1)
        cp_child2 = random.randint(1,TOTAL_INGREDIENTS-1)

        for i in range(cp_child1, TOTAL_INGREDIENTS):
            child1[i] += parent2[i]

        for i in range(cp_child2, TOTAL_INGREDIENTS):
            child2[i] += parent1[i]

        return [child1, child2]

    def mutate(self, chromosome):
        for i in range(len(chromosome)):
            if random.random() < self.mutation_rate:
                chromosome[i] = random.randint(0, self.max_limits[i])
        return chromosome

    def fitness(self, chromosome):
        chromosome_protein = 0
        chromosome_fat = 0
        chromosome_carbs = 0

        for i in range(len(self.data)):
            ingred_protein, ingred_fats, ingred_carbs = self.calculate_food_intake(i, chromosome[i])
            chromosome_protein += ingred_protein
            chromosome_fat += ingred_fats
            chromosome_carbs += ingred_carbs

        score = 100
        protein_threshold = self.target_protein * 1.1
        fat_threshold = self.target_fat * 1.1
        carbs_threshold = self.target_carbs * 1.1

        if chromosome_protein > protein_threshold or chromosome_fat > fat_threshold or chromosome_carbs > carbs_threshold:
            return 0
        else:
            score -= (abs(chromosome_protein - self.target_protein) / self.target_protein) * 25
            score -= (abs(chromosome_fat - self.target_fat) / self.target_fat) * 25
            score -= (abs(chromosome_carbs - self.target_carbs) / self.target_carbs) * 25

        return max(score, 0)

    def truncation_selection(self, population, size):
        sorted_population = sorted(population, key=self.fitness, reverse=True)
        return sorted_population[:size]

    def binary_tournament_selection(self, population, size):
        selected = []
        for _ in range(size):
            ind1, ind2 = random.sample(population, 2)
            if self.fitness(ind1) > self.fitness(ind2):
                selected.append(ind1)
            else:
                selected.append(ind2)
        return selected

    def fitness_proportional_selection(self, population, size):
        total_fitness = sum(self.fitness(chromo) for chromo in population)
        selection_probs = [self.fitness(chromo) / total_fitness for chromo in population]
        selected_indices = np.random.choice(range(len(population)), size=size, replace=True, p=selection_probs)
        return [population[i] for i in selected_indices]
    
    def preference_dictionary(self):
        
        results_data = []
        for index, unit in enumerate(self.data):
            if unit > 0:
                food_item_name = self.data[index][0]
                grams = float(self.data[index][1]) 
                protein = float(self.data[index][2])
                fat = float(self.data[index][3])
                carbs = float(self.data[index][4])
                results = {"food_item": food_item_name, "unit (g)": str(grams), "protein": str(protein), "fat": str(fat), "carbohydrates": str(carbs)}
                results_data.append(results)
        print(results_data)
        return results_data   
    
    def chromosome_dictionary(self, day):
        
        # to be corrected as only max days k hissaab sai schdeule showed
        if day > self.max_days:
            day = day%self.max_days

        chromosome = self.best_lst[day-1]


        
        results_data = []
        for index, unit in enumerate(chromosome):
            if unit > 0:
                food_item_name = self.data[index][0]
                grams = float(self.data[index][1]) * unit
                protein = float(self.data[index][2]) * unit
                fat = float(self.data[index][3])* unit
                carbs = float(self.data[index][4])* unit
                results = {"food_item": food_item_name, "unit (g)": str(round(grams,2)), "protein": str(round(protein,2)), "fat": str(round(fat,2)), "carbohydrates": str(round(carbs,2))}
                results_data.append(results)
        print(results_data)
        return results_data    


    def run(self, athlete, gender, age_group, food_choices):
        # path1 = "ingredients.csv"
        path1 = "C:/Users/user/Desktop/spring 2024/CI/project CI frontend attempt 2/CI project ingredients-dataset - ingredients-dataset.csv"
        self.data_file = self.load_food_data(path1)

        population = self.initialize_population(athlete, gender, age_group, food_choices)

        # print(self.data)
        
        for gen in range(self.num_generations):
            # Selection
            # changed to the fitness from binary tournament
            selected = self.fitness_proportional_selection(population, 10)

            for i in range(0, 10, 2):
                if len(self.data) < 20:
                    child = self.sbx_crossover(selected[i], selected[i+1])
                else:
                    child = self.new_crossOver(selected[i], selected[i+1])

                child[0] = self.mutate(child[0])
                child[1] = self.mutate(child[1])
                population += child
            population = self.truncation_selection(population, self.population_size)


        best_lst = []
        
        for i in population:
            if i not in best_lst:
                best_lst.append(i)

            if len(best_lst) == 10:
                break

        self.max_days = len(best_lst)
        

        # print("\n\n\n",best_lst)


        print(f" The required target are: \nPortein --> {self.target_protein}, Fats --> {self.target_fat}, Carbs --> {self.target_carbs}")
        for i in best_lst:
            print(f"chormosome -> {i}, fitness -> {self.fitness(i)}")
        
        self.best_lst = best_lst[:]
        return self.best_lst         
            

    def plotting_image(self, chromosome, day):
        # Extracting data for the last printed chromosome
        # chromosome = population[0]

        # Initialize lists to store daily intake of protein, fats, and carbs
        daily_protein = []
        daily_fats = []
        daily_carbs = []

        # Calculate daily intake for each nutrient
        for index, unit in enumerate(chromosome):
            if unit > 0:
                protein, fats, carbs = self.calculate_food_intake(index, unit)
                daily_protein.append(protein)
                daily_fats.append(fats)
                daily_carbs.append(carbs)

        # Convert lists to numpy arrays
        daily_protein = np.array(daily_protein)
        daily_fats = np.array(daily_fats)
        daily_carbs = np.array(daily_carbs)

        # Total required nutrients
        total_protein_required = self.target_protein
        total_fats_required = self.target_fat
        total_carbs_required = self.target_carbs

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

        # plt.show()
        plt.savefig(f'assets/day_{day}_barChart.png')

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
        # plt.show()
        plt.savefig(f'assets/day_{day}_pieChart.png')

# Example usage:
# EA = EvolutionaryAlgorithm(population_size=100, mutation_rate=0.6, num_generations=5000, eta_c=1.0)
# best_solution = EA.run(athlete="Boxers", gender="Male", age_group="30-40")
