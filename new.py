import retro
import neat
import numpy as np
import cv2
import pickle
import os

# Configuration de NEAT et de l'émulateur
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, 'config-feedfoward')
env = retro.make(game="SuperMarioWorld-Snes", state='YoshiIsland1')
existing_model = False
best_genome = None  # Initialize best_genome

# Vérification de l'existence d'un modèle
if os.path.exists('winner.pkl'):
    existing_model = True
    with open('winner.pkl', 'rb') as input_file:
        best_genome = pickle.load(input_file)

# Fonction de l'évaluation des génomes
imgarray = []

def eval_genomes(genomes, config):
    global best_genome
    for genome_id, genome in genomes:
        # If there is a best genome, use it as the starting point
        if best_genome is not None:
            genome = best_genome
        total_reward = 0
        observation = env.reset()
        observation = observation[0]
        action = env.action_space.sample()
        inx, iny, inc = env.observation_space.shape  # Fix typo in '_shape'
        inx = int(inx / 8)
        iny = int(iny / 8)
        current_max_fitness = 0
        fitness_current = 0
        frame = 0
        counter = 0
        xpos = 0
        xpos_max = 0
        done = False
        win = False
        halfway_block = False

        while not done:
            env.render()
            frame += 1
            observation = cv2.resize(observation, (inx, iny))
            observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
            observation = np.reshape(observation, (inx, iny))
            for x in observation:
                for y in x:
                    imgarray.append(y)
            nnOutput = genome.activate(imgarray)
            observation, reward, terminated, truncated, info = env.step(nnOutput)
            xpos = info['xpos']
            if xpos > xpos_max:
                fitness_current += 1
                xpos_max = xpos
            if fitness_current > current_max_fitness:
                current_max_fitness = fitness_current
                counter = 0
            else:
                counter += 1
            if terminated or counter == 300:
                terminated = True
                print(genome_id, fitness_current)
            genome.fitness = fitness_current
            imgarray.clear()
            total_reward += reward
            done = terminated
            win = info['done']
            halfway = info['halfway']

            if halfway == 1 and not halfway_block:
                fitness_current += 5000
                halfway_block = True

            if win == 1:
                fitness_current += 100000

# Create a new population
p = neat.Population(config)

# Run the NEAT algorithm
p.run(eval_genomes)

# Save the best genome after training
best_genome = p.best_genome
with open('winner.pkl', 'wb') as output:
    pickle.dump(best_genome, output, 1)
