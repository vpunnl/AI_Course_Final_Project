import random

def get_children(board, position):
    moves = ['up', 'down', 'left', 'right']
    children = []
    x, y = position
    for move in moves:
        new_pos = move_position(position, move)
        if is_valid_move(new_pos, len(board)):
            new_board = simulate_move(board, new_pos)
            children.append((new_board, new_pos, move))
    return children

def move_position(position, direction):
    x, y = position
    if direction == 'up':
        return (x - 1, y)
    elif direction == 'down':
        return (x + 1, y)
    elif direction == 'left':
        return (x, y - 1)
    elif direction == 'right':
        return (x, y + 1)

def is_valid_move(position, size):
    x, y = position
    return 0 <= x < size and 0 <= y < size

def random_move(notValidMove):
    pos = ['up','down','left','right']
    pos.remove(notValidMove)
    randmove = random.choice(pos)
    return randmove

def no_collision(move,moving_agent,another_agent):
    mvx,mvy = moving_agent.position
    if move == 'up':
        mvx -= 1
    elif move == 'down':
        mvx += 1
    elif move == 'left':
        mvy -= 1
    elif move == 'right':
        mvy += 1
    ax,ay = another_agent.position

    if mvx != ax and mvy != ay:
        return True
    else:
        return False

def simulate_move(board, new_pos):
    new_board = [row[:] for row in board]
    new_board[new_pos[0]][new_pos[1]] = 0 
    return new_board


def evaluate_board(board):
    score = 0
    for row in board:
        for cell in row:
            if cell == 1: 
                score += 10 
            elif cell == 2: 
                score += 2 
    return score

def is_terminal(board):
    return all(cell == 0 for row in board for cell in row)

# Probabilistic Minimax
def probabilistic_minimax(board, depth, alpha, beta, maximizingPlayer, position):
    if depth == 0 or is_terminal(board):
        return evaluate_board(board), None

    best_move = None
    if maximizingPlayer:
        max_eval = float('-inf')
        for child_board, new_pos, move in get_children(board, position):
            eval, _ = probabilistic_minimax(child_board, depth - 1, alpha, beta, False, new_pos)
            if eval > max_eval:
                max_eval = eval
                best_move = move
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = float('inf')
        for child_board, new_pos, move in get_children(board, position):
            eval, _ = probabilistic_minimax(child_board, depth - 1, alpha, beta, True, new_pos)
            if eval < min_eval:
                min_eval = eval
                best_move = move
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval, best_move

# Genetic Algorithm
class GeneticAlgorithm:
    def __init__(self, initial_population_size, board, moves_per_path=10):
        self.population_size = initial_population_size
        self.population = [self.generate_random_path(moves_per_path) for _ in range(initial_population_size)]
        self.board = board

    def generate_random_path(self, moves_per_path):
        return [random.choice(['up', 'down', 'left', 'right']) for _ in range(moves_per_path)]

    def fitness(self, path, start_position):
        x, y = start_position
        score = 0
        consecutive_coins = 0
        simulated_board = [row[:] for row in self.board.board]
        for move in path:
            if move == 'up' and x > 0:
                x -= 1
            elif move == 'down' and x < len(simulated_board) - 1:
                x += 1
            elif move == 'left' and y > 0:
                y -= 1
            elif move == 'right' and y < len(simulated_board[0]) - 1:
                y += 1

            if simulated_board[x][y] == 1 or simulated_board[x][y] == 2:
                score += 10
                consecutive_coins+=1
                if consecutive_coins == 3:
                    score += 90
                    consecutive_coins = 0
                simulated_board[x][y] = 0
            else:
                consecutive_coins = 0
        return score

    def select(self):
        sorted_population = sorted(self.population, key=lambda path: self.fitness(path, (0, 0)), reverse=True)
        return sorted_population[:int(0.5 * len(sorted_population))]

    def crossover(self, path1, path2):
        crossover_index = random.randint(1, len(path1) - 1)
        return path1[:crossover_index] + path2[crossover_index:]

    def mutate(self, path, mutation_rate=0.1):
        return [move if random.random() > mutation_rate else random.choice(['up', 'down', 'left', 'right']) for move in path]

    def run_generation(self):
        selected = self.select()
        next_generation = []
        while len(next_generation) < self.population_size:
            parent1, parent2 = random.sample(selected, 2)
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            next_generation.append(child)
        self.population = next_generation

    def calculate_path(self, start_position, num_generations=20):
        for _ in range(num_generations):
            self.run_generation()
        return self.best_path(start_position)
    
    def best_path(self,start_position):
        best = max(self.population, key=lambda path: self.fitness(path, start_position))
        return best

# Agent Setup
class Agent:
    def __init__(self, board, start_pos,name,ga_machine):
        self.board = board
        self.position = start_pos
        self.score = 0
        self.name = name
        self.move_stack = []
        self.ga_machine = ga_machine
        self.ga_path = []
        self.moves_without_coins = 0
        self.consecutive_coins = 0

    def move(self, direction):
        x, y = self.position
        moves = {
            'up': (x - 1, y),
            'down': (x + 1, y),
            'left': (x, y - 1),
            'right': (x, y + 1)
        }
        if direction in moves and 0 <= moves[direction][0] < self.board.size and 0 <= moves[direction][1] < self.board.size:
            self.position = moves[direction]
            self.collect_coins()
        else:
            randmove = random_move(direction)
            self.move(randmove)

    def collect_coins(self):
        x, y = self.position
        if self.board.collect_coin(x, y):
            print(f"{self.name} collected a coin at {self.position}, current score : {self.score}")
            self.consecutive_coins+=1
            if self.consecutive_coins == 3:
                self.score+=6
                self.consecutive_coins = 0
            else:
                self.score+=1
        else:
            self.consecutive_coins = 0
            self.moves_without_coins += 1

    def decide_move(self,turn):
        is_maximizing = turn % 2 == 0
        _, move = probabilistic_minimax(self.board.board, 7, float('-inf'), float('inf'), is_maximizing, self.position)
        return move

    def update_path(self, new_path):
        """Update the agent's path with a new set of moves."""
        self.ga_path = new_path

# Board Setup
class GameBoard:
    def __init__(self, size=8):
        self.size = size
        self.board = [[random.choice([0,1,2]) for _ in range(size)] for _ in range(size)]
        self.board[0][0] = 0
        self.board[7][7] = 0
    
    def toggle_visibility(self):
        """Randomly toggle visibility of coins."""
        for row in range(self.size):
            for col in range(self.size):
                if self.board[row][col] != 0:
                    self.board[row][col] = random.choice([1, 2])
                    pass
    def simulate_visibility_change(self):
        """Simulate a change in visibility without actually changing the board."""
        simulated_board = []
        for row in self.board:
            simulated_row = []
            for cell in row:
                if cell != 0:
                    simulated_row.append(random.choice([1, 2]))
                else:
                    simulated_row.append(0)
            simulated_board.append(simulated_row)
        return simulated_board

    def collect_coin(self, x, y):
        if self.board[x][y] == 1:
            self.board[x][y] = 0
            return True
        return False

    def display(self, agents):
        display_board = [[self.board[i][j] for j in range(self.size)] for i in range(self.size)]
        display_symbols = {0: ' ', 1: '●', 2: '○'}
        agent_symbols = ['A', 'B']

        for index, agent in enumerate(agents):
            x, y = agent.position
            display_board[x][y] = agent_symbols[index]

        print("  +" + "---+" * self.size)

        for row in display_board:
            row_display = " | ".join(display_symbols.get(cell, cell) for cell in row)
            print(f" | {row_display} |")
            print("  +" + "---+" * self.size)

# Genetic Algorithm
def game_loop(board, agents):
    turn = 0    
    agent01.ga_path = agent01.ga_machine.best_path(agent01.position)
    agent02.ga_path = agent02.ga_machine.best_path(agent02.position)

    while not is_terminal(board.board):
        current_agent = agents[turn % 2] 
        another_agent = agents[(turn + 1) % 2]

        if turn > 100: # use ga to decide
            # print(current_agent.ga_path)
            if len(current_agent.ga_path) == 1:
                current_agent.ga_path = current_agent.ga_machine.best_path(current_agent.position)
            if len(another_agent.ga_path) == 1:
                another_agent.ga_path = another_agent.ga_machine.best_path(another_agent.position)
   
            if turn % 10 == 0 or turn % 10 == 1: 
                new_path1 = current_agent.ga_machine.calculate_path(current_agent.position, 10)
                current_agent.update_path(new_path1)
                # print('Mutated')

            move = current_agent.ga_path.pop(0)
            if no_collision(move,current_agent,another_agent):
                current_agent.move(move)
                current_agent.collect_coins()
            else:
                new = random_move(move)
                current_agent.move(new)
                current_agent.collect_coins()
            # print('ga')


        else: # use minimax to move
            move = current_agent.decide_move(turn)
            current_agent.move(move)
            current_agent.collect_coins()

        if current_agent.moves_without_coins >= 5:
            new_path = current_agent.ga_machine.calculate_path(current_agent.position)
            current_agent.update_path(new_path)
            
        print(f'{current_agent.name} moved',move)
        print(f'{current_agent.name} is at, {current_agent.position}')
        board.toggle_visibility()
        board.display(agents)
        turn += 1

    score1 = agent01.score
    score2 = agent02.score
    print('---------------GAME OVER---------------')

    if score1 > score2 :
        print(f'{agent01.name} won with the score of {agent01.score}')
        print(f'while {agent02.name} has a score of {agent02.score}')
    
    if score1 < score2 :
        print(f'{agent02.name} won with the score of {agent02.score}')
        print(f'while {agent01.name} has a score of {agent01.score}')

# Game loop setup
game_board = GameBoard()

ga1 = GeneticAlgorithm(initial_population_size=50, board=game_board, moves_per_path=10)
ga2 = GeneticAlgorithm(initial_population_size=50, board=game_board, moves_per_path=10)

agent01 = Agent(game_board, (0, 0), 'Agent1', ga1)
agent02 = Agent(game_board, (7, 7), 'Agent2', ga2)
players = [agent01, agent02]

# Start the game
game_loop(game_board, players)