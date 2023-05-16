import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt


class City:
    def __init__(self):
        self.x = np.random.uniform(0, 100)
        self.y = np.random.uniform(0, 100)

    def distance(self, city):
        dx = self.x - city.x
        dy = self.y - city.y
        return np.sqrt(dx**2 + dy**2)

    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"


def generate_cities_and_distances(N):
    cities = [City() for _ in range(N)]
    distances = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            distance = cities[i].distance(cities[j]) + 1e-6  # Add a small constant to prevent zero distances
            distances[i, j] = distances[j, i] = distance
    return cities, distances



def plot(cities, path):
    # Get the list of pairs from the path
    path_pairs = path[0]

    # Path is a list of tuples, convert it to list of indices
    path_indices = [i for i, _ in path_pairs]

    # Append the starting city at the end to complete the loop
    path_indices.append(path_indices[0])

    # Get the coordinates of the cities in the order they are visited
    x_coordinates = [cities[i].x for i in path_indices]
    y_coordinates = [cities[i].y for i in path_indices]

    # Create a scatter plot of all cities
    plt.scatter([city.x for city in cities], [city.y for city in cities])

    # Plot the path
    plt.plot(x_coordinates, y_coordinates, 'r')
    plt.show()


class AntColony:
    def __init__(self, distances, n_ants, n_best, n_iterations, decay, alpha, beta):
        """
        Args:
            distances (2D numpy.array): Square matrix of distances. Diagonal is assumed to be np.inf.
            n_ants (int): Number of ants running per iteration
            n_best (int): Number of best ants who deposit pheromone
            n_iterations (int): Number of iterations
            decay (float): Rate it which pheromone decays. The pheromone value is multiplied by decay, so 0.95 will lead to decay, 0.5 to much faster decay.
            alpha (int or float): exponenet on pheromone, higher alpha gives pheromone more weight. Default=1
            beta (int or float): exponent on distance, higher beta give distance more weight. Default=1
        """
        self.distances  = distances
        self.pheromone = np.ones(self.distances.shape) / len(distances)
        self.all_inds = range(len(distances))
        self.n_ants = n_ants
        self.n_best = n_best
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta

    def run(self):
        shortest_path = None
        all_time_shortest_path = ("placeholder", np.inf)
        for i in range(self.n_iterations):
            all_paths = self.gen_all_paths()
            shortest_path = min(all_paths, key=lambda x: x[1])
            self.spread_pheronome(all_paths, shortest_path)
            print(f"Ітерація {i + 1}, найкоротший шлях: {int(shortest_path[1])}")
            if shortest_path[1] < all_time_shortest_path[1]:
                all_time_shortest_path = shortest_path
            self.pheromone *= self.decay
        return all_time_shortest_path

    def spread_pheronome(self, all_paths, shortest_path):
        sorted_paths = sorted(all_paths, key=lambda x: x[1])
        for path, dist in sorted_paths[:self.n_best]:
            for move in path:
                self.pheromone[move] += 1.0 / self.distances[move]

    def gen_path_dist(self, path):
        total_dist = 0
        for ele in path:
            total_dist += self.distances[ele]
        return total_dist

    def gen_all_paths(self):
        all_paths = []
        for i in range(self.n_ants):
            path = self.gen_path(0)
            all_paths.append((path, self.gen_path_dist(path)))
        return all_paths

    def gen_path(self, start):
        path = []
        visited = set()
        visited.add(start)
        prev = start
        for i in range(len(self.distances) - 1):
            move = self.pick_move(self.pheromone[prev], self.distances[prev], visited)
            path.append((prev, move))
            prev = move
            visited.add(move)
        path.append((prev, start))
        return path

    def pick_move(self, pheromone, dist, visited):
        pheromone = np.copy(pheromone)
        pheromone[list(visited)] = 0

        row = pheromone ** self.alpha * ((1.0 / (dist + 1e-10)) ** self.beta)

        row[np.isinf(row)] = 1e100
        row[np.isnan(row)] = 1e-100

        zero_idxs = np.where(row == 0)[0]
        nonzero_idxs = np.where(row > 0)[0]

        if len(nonzero_idxs) == 0:
            choice = np.random.choice(zero_idxs, 1)[0]
        else:
            norm_row = row / row.sum()
            choice = np.random.choice(nonzero_idxs, 1, p=norm_row[nonzero_idxs])[0]

        return choice


# Генеруємо випадкову кількість міст у діапазоні 25-35
N = random.randint(25, 35)

# Генеруємо міста та відстані між ними
cities, distances = generate_cities_and_distances(N)

# Зберігаємо дані у файл
df = pd.DataFrame(distances)
df.to_csv('distances.csv', index=False)

distances = pd.read_csv('distances.csv').values
ant_colony = AntColony(distances, n_ants=10, n_best=2, n_iterations=100, decay=0.95, alpha=1, beta=2)
shortest_path = ant_colony.run()
print("\nРезультуючий найкоротший шлях: {}".format(int(shortest_path[1])))
plot(cities, shortest_path)
