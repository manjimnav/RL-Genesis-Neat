import retro        # pip install gym-retro
import numpy as np  # pip install numpy
import cv2          # pip install opencv-python
import neat         # pip install neat-python
import pickle       # pip install cloudpickle
from pureples.hyperneat.hyperneat import create_phenotype_network
from pureples.shared.visualize import draw_net
from pureples.shared.substrate import Substrate
from pureples.shared.gym_runner import run_hyper
from pureples.hyperneat.hyperneat import create_phenotype_network

class Worker(object):
    def __init__(self, genome, config):
        self.genome = genome
        self.config = config
        # Network input, hidden and output coordinates.
        input_coordinates = []
        for i in range(0,5883):
            input_coordinates.append((-1. +(2.*i/3.), -1.))
        hidden_coordinates = [[(-0.5, 0.5), (0.5, 0.5)], [(-0.5, -0.5), (0.5, -0.5)]]
        output_coordinates = [(-1., 1.), (1., 1.), (-1., 1.)]
        activations = len(hidden_coordinates) + 2

        sub = Substrate(input_coordinates, output_coordinates, hidden_coordinates)
        self.sub = sub
        activations = len(hidden_coordinates) + 2
        self.activations = activations
        
    def work(self):
        
        self.env = retro.make('Airstriker-Genesis')
        
        self.env.reset()
        
        ob, _, _, _ = self.env.step(self.env.action_space.sample())
        
        inx = int(ob.shape[0]/6)
        iny = int(ob.shape[1]/6)
        done = False
        #print(len(self.sub.input_coordinates))

        cpnn = neat.nn.FeedForwardNetwork.create(self.genome, self.config)
        net = create_phenotype_network(cpnn, self.sub, "sigmoid")
        #print(len(net.input_nodes))
        
        fitness = 0
        xpos_max = 0
        counter = 0
        last_lives = 0
        while not done:
            self.env.render()
            ob = cv2.resize(ob, (inx, iny))
            #ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            #ob = np.reshape(ob, (inx, iny))
            imgarray = np.ndarray.flatten(ob)
            imgarray = np.interp(imgarray, (0, 254), (-1, +1))
            #for k in range(self.activations):
            for k in range(self.activations):
                actions = net.activate(imgarray)
            actions_padded = np.zeros(12)
            actions_padded[0] = actions[0]
            actions_padded[6] = actions[1]
            actions_padded[7] = actions[2]
            ob, rew, done, info = self.env.step(actions_padded)
            
            xpos = info['score']

            if xpos > xpos_max:
                xpos_max = xpos
                #counter = 0
                print(xpos)
                fitness += xpos + 0.05
            else:
                counter += 1
                fitness += 0.05

            #fitness += xpos
            if info['lives'] < last_lives:
                done = True
                last_lives = info['lives']

        return fitness


def eval_genomes(genome, config):
    
    worky = Worker(genome, config)
    return worky.work()


if __name__ == '__main__':
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         'config_cppn_pole_balancing')

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(10))
    pe = neat.ParallelEvaluator(10, eval_genomes)
    #p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-17')
    winner = p.run(pe.evaluate, 10)

    with open('winner.pkl', 'wb') as output:
        pickle.dump(winner, output, 1)

