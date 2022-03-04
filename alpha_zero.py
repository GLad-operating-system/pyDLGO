from network import Network
from config import BOARD_SIZE, KOMI
from board import Board, PASS, RESIGN, BLACK, WHITE, EMPTY, INVLD
from mcts import Search
from train import DataSet, TrainingPipe, Chunk
from time_control import TimeControl

import numpy as np
import argparse, os

class AlphaZeroPipe:
    def __init__(self, args):
        self.args = args
        self.board = Board(BOARD_SIZE, KOMI)
        self.network = Network(BOARD_SIZE)
        self.data_set = DataSet()

        self.time_control = TimeControl() # ignore

        if self.args.load_weights != None:
            self.network.load_pt(self.args.load_weights)
       
    def running(self):
        i = 0
        if not os.path.isdir(args.dir):
            os.mkdir(args.dir)

        for e in range(args.epoches):
            for _ in range(args.games_per_epoch):
                self.selfplay()

                i+=1
                if i % 100 == 0:
                    print("Played {} games".format(i))

            weights_name = "{}/{}_{}".format(args.dir, e, self.args.weights_name)

            self.network.trainable(True)
            pipe = TrainingPipe(self.network, self.data_set)
            pipe.running(args.step, args.verbose_step, args.batch_size, args.learning_rate, True) # disable plot
            pipe.save_weights(weights_name) # Save new network every step.

    def selfplay(self):
        self.board.reset(self.board.board_size, self.board.komi)
        self.network.trainable(False)
        self.time_control.reset()
        winner = INVLD

        temp = []
        random_threshold =  (int)(1.5 * self.board.board_size)
        resign_threshold = self.args.resign_threshold
        resign_probability = self.args.resign_probability
        playouts = self.args.playouts

        if np.random.choice(2, 1,
                                p=[1-resign_probability, resign_probability])[0] == 0:
            # Disable the resign move if resign_threshold is 0.
            resign_threshold = 0

        # Self-playing...
        while winner == INVLD:
            chunk = Chunk()
            to_move = self.board.to_move

            search = Search(self.board, self.network, self.time_control)
            move, features, prob = search.selfplay(playouts, resign_threshold, random_threshold)
            self.board.play(move)

            if move == RESIGN:
                winner = int(to_move == 0)
            elif self.board.num_passes >= 2:
                score = self.board.final_score()
                if score > 0.1:
                    winner = BLACK
                elif score < -0.1:
                    winner = WHITE
                else:
                    winner = EMPTY

            chunk.inputs = features
            chunk.policy = prob
            chunk.to_move = to_move
            temp.append(chunk)

        # The game is over. Store the data to the data pool.
        for chunk in temp:
            if winner == EMPTY:
                chunk.value = 0
            elif winner == chunk.to_move:
                chunk.value = 1
            elif winner != chunk.to_move:
                chunk.value = -1

        self.data_set.buffer.extend(temp)
        self.data_set.resize(args.buffer_size)
        self.network.nn_cache.clear()

def valid_args(args):
    result = True

    if args.dir == None:
        print("Must to give the argument --dir <string>")
        result = False
    if args.weights_name == None:
        print("Must to give the argument --weights-name <string>")
        result = False
    if args.step == None:
        print("Must to give the argument --step <integer>")
        result = False
    if args.batch_size == None:
        print("Must to give the argument --batch-size <integer>")
        result = False
    if args.learning_rate == None:
        print("Must to give the argument --learning-rate <float>")
        result = False
    if args.games_per_epoch == None:
        print("Must to give the argument --games-per-epoch <int>")
        result = False
    if args.epoches == None:
        print("Must to give the argument --epoches <int>")
        result = False
    if args.buffer_size == None:
        print("Must to give the argument --buffer-size <int>")
        result = False

    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--playouts", metavar="<integer>",
                        help="The number of playouts", type=int, default=400)
    parser.add_argument("-s", "--step", metavar="<integer>",
                        help="The training steps per epoch", type=int)
    parser.add_argument("-v", "--verbose-step", metavar="<integer>",
                        help="Dump verbose on every X steps.", type=int, default=1000)
    parser.add_argument("-b", "--batch-size", metavar="<integer>",
                        help="The batch size number.", type=int)
    parser.add_argument("-w", "--weights-name", metavar="<string>",
                        help="The output weights name.", type=str)
    parser.add_argument("--load-weights", metavar="<string>",
                        help="The inputs weights name.", type=str)
    parser.add_argument("-d", "--dir", metavar="<string>",
                        help="The saving file directory", type=str, default="root")
    parser.add_argument("-r", "--resign-threshold", metavar="<float>",
                        help="Resign when winrate is less than x.", type=float, default=0.1)
    parser.add_argument("-rp", "--resign-probability", metavar="<float>",
                        help="Resign when winrate is less than x.", type=float, default=0.9)
    parser.add_argument("-l", "--learning-rate", metavar="<float>",
                        help="The learning rate.", type=float)
    parser.add_argument("-g", "--games-per-epoch", metavar="<integer>",
                        help="The number of games per epoch", type=int)
    parser.add_argument("-e", "--epoches", metavar="<integer>",
                        help="The training epoches.", type=int)
    parser.add_argument("-bs", "--buffer-size", metavar="<integer>",
                        help="The training buffer size ", type=int)
    args = parser.parse_args()

    if valid_args(args):
        pipe = AlphaZeroPipe(args)
        pipe.running()
