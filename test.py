import retro
import neat
import numpy as np
import cv2
import os

# Configuration de NEAT et de l'émulateur
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, 'config-feedfoward')
env = retro.make(game="SuperMarioWorld-Snes", state='YoshiIsland2')

# Les modèles en entrainement seront sauvegardés dans ce dossier
checkpoint_folder = 'neat_checkpoints'

# Fonction de l'évaluation des génomes
def eval_genomes(genomes, config, total_reward=0):
    global net
    global checkpoint_folder
    imgarray = []
    
    for genome_id, genome in genomes:
        observation = env.reset()[0]
        action = env.action_space.sample()
        inx, iny, inc = env.observation_space._shape
        inx = int(inx/8)
        iny = int(iny/8)
        net = neat.nn.RecurrentNetwork.create(genome, config)
        
        fitness_current, frame, score, score_max, counter, xpos, xpos_max = 0, 0, 0, 0, 0, 0, 0
        done, win, halfway_block = False, False, False
        
        while not done:
            observation = cv2.resize(observation, (inx, iny))
            observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
            observation = np.reshape(observation, (inx, iny))
            imgarray.extend([pixel_value / 255.0 for row in observation for pixel_value in row])
            
            nnOutput = net.activate(imgarray)
            observation, reward, terminated, truncated, info = env.step(nnOutput)
            
            xpos, carry, score, win, halfway = info['xpos'], info['carry'], info['score'], info['done'], info['halfway']
            frame += 1
            done = terminated

            if xpos <= xpos_max:
                counter += 1
                stopped = True           

            if xpos > xpos_max:
                fitness_current += 3
                xpos_max = xpos
                stopped = False

            if carry == 1:
                fitness_current += 1

            if score > score_max:
                fitness_current += 100
                score_max = score

            if halfway == 1 and not halfway_block:
                fitness_current += 50000
                halfway_block = True

            if win == 1:
                fitness_current += 100000

            if stopped == False:
                counter = 0

            genome.fitness = fitness_current

            if terminated or counter == 400:
                terminated = True
                done = True
                print('Génome: ' + str(genome_id), 'Efficacité: '+ str(fitness_current))
                print('')

            imgarray.clear()
        


# Création ou importation de la population
        
os.makedirs(checkpoint_folder, exist_ok=True)
checkpoint_files = os.listdir(checkpoint_folder)
if checkpoint_files:
    checkpoint_files.sort(key=lambda x: int(x), reverse=True)
    last_checkpoint_file = checkpoint_files[0]
    last_checkpoint_path = os.path.join(checkpoint_folder, last_checkpoint_file)
    print(f"Importation de la génération: {last_checkpoint_file}")
    p = neat.Checkpointer.restore_checkpoint(last_checkpoint_path)
    p.config = config
else:
    p = neat.Population(config)


# Initialisation des reporteurs de stats et checkpoints

p.add_reporter(neat.StdOutReporter(True))
p.add_reporter(neat.Checkpointer(10,time_interval_seconds=None, filename_prefix=os.path.join(checkpoint_folder, '')))



# Évaluation des génomes et sortie sans messages d'erreur en cas d'interruption
try:
    winner = p.run(eval_genomes)

except KeyboardInterrupt:
    exit(0)


