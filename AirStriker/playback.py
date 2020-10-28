import retro
import numpy as np
import cv2 
import neat
import pickle

env = retro.make('Airstriker-Genesis')

rom_path = retro.data.get_romfile_path('Airstriker-Genesis', retro.data.Integrations.STABLE)
system = retro.get_romfile_system(rom_path)
core = retro.get_system_info(system)
buttons = core['buttons']
print(buttons)

imgarray = []

xpos_end = 0



config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward')

p = neat.Population(config)


p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)


with open('winner.pkl', 'rb') as input_file:
    genome = pickle.load(input_file)


ob = env.reset()
ac = env.action_space.sample()

inx, iny, inc = env.observation_space.shape

inx = int(inx/6)
iny = int(iny/6)

net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

current_max_fitness = 0
fitness_current = 0
frame = 0
counter = 0
xpos = 0
xpos_max = 0

done = False

while not done:
    
    env.render()
    frame += 1

    ob = cv2.resize(ob, (inx, iny))
    imgarray = np.ndarray.flatten(ob)
    imgarray = np.interp(imgarray, (0, 254), (-1, +1))
    
    actions = net.activate(imgarray)
    print(actions)
    actions_padded = np.zeros(12)
    actions_padded[0] = actions[0]
    actions_padded[6] = actions[1]
    actions_padded[7] = actions[2]
    
    actions_padded = np.around(actions_padded);
    print(actions_padded)
    ob, rew, done, info = env.step(actions_padded)
    #imgarray.clear()
    
        

            
    



