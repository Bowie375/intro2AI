import numpy as np
np.random.seed(3407)
import argparse
from typing import List
from loadMap import tryToLoad
from simuScene import Scene2D
from visualizer import SimpleViewer, TestEnv
from answerPDControl import calc_pd_force
from answerPlanning import *
from physicsWrapper import PhysicsInfo
import pdb

def readArgparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--test_idx', type=int, default=0)
    args = parser.parse_args()
    return args



class PlanningController:
    def __init__(self, scene: Scene2D, eat_seq: List[int]) -> None:
        self.planner = RRT(scene.walls)
        self.scene = scene
        self.eat_sequence = eat_seq
        self.num_food = len(eat_seq)
        self.num_eaten = 0
        self.physice_info = PhysicsInfo(scene)
        self.cnt = 0
        #print(scene.foods)
        #pdb.set_trace()
        self.kp = 10
        self.kd = 1
        
        next_food = np.array(scene.foods[self.eat_sequence[0]])
        self.planner.find_path(self.physice_info.pacman_position(), next_food)


    def change_traj(self):
        self.num_eaten += 1
        if self.num_eaten < self.num_food:
            # 还没有全部吃完
            next_food = np.array(scene.foods[self.eat_sequence[self.num_eaten]])
            self.planner.find_path(self.physice_info.pacman_position(), next_food)
        else:
            # 全部吃完了
            self.scene.run = False
            self.scene.use_time = self.cnt
            print(f"Pacman use {self.cnt} steps to eat all the food. Please Exit!")
    
    def apply_pd_force(self):
        if self.scene.run:
            target_pose = self.planner.get_target(self.physice_info.pacman_position(), self.physice_info.pacman_velocity())
            force = calc_pd_force(target_pose, self.physice_info.pacman_position(), 
                                self.physice_info.pacman_velocity(),
                                kp=self.kp, kd=self.kd)
            #if self.num_eaten==2:
            #    pdb.set_trace()
            self.scene.apply_force(force)
    
    def check_eat(self):
        #pdb.set_trace()
        if self.scene.run:
            self.cnt += 1
            if self.scene.check_eat(self.eat_sequence[self.num_eaten]):
                # remove food from the scene
                self.scene.foods[self.eat_sequence[self.num_eaten]][1] = 1000
                #pdb.set_trace()
                self.change_traj()
            else:
                if self.cnt > 5000:
                    self.scene.run = False
                    print(f"Failed! Only {self.num_eaten} food has been eaten!")


def task_planning_and_control(scene: Scene2D, eat_sequence: List[int]):
    controller = PlanningController(scene, eat_sequence)
    scene.pre_simulation_func = controller.apply_pd_force
    #pdb.set_trace()
    #print(PhysicsInfo(scene).pacman_position())
    scene.post_simulation_func = controller.check_eat


if __name__ == "__main__":
    args = readArgparse()
    if not args.test:
        seq_file = 'data_q3/seq0.txt' # 自己修改测试
        with open(seq_file, 'r') as file:
            lines = file.readlines()
            layout_name = 'layouts/' + lines[0].split()[0]
            numbers = lines[1].split()
            eat_seq = [int(num) for num in numbers]
        scene = Scene2D(tryToLoad(layout_name))
        #pdb.set_trace()
        task_planning_and_control(scene, eat_seq)
        viewer = SimpleViewer(True, scene)
        viewer.update_food_flag = True
        viewer.run()
    else:
        seq_file = f'data_q3/seq{args.test_idx}.txt'
        with open(seq_file, 'r') as file:
            lines = file.readlines()
            layout_name = 'layouts/' + lines[0].split()[0]
            numbers = lines[1].split()
            eat_seq = [int(num) for num in numbers]
        scene = Scene2D(tryToLoad(layout_name))
        task_planning_and_control(scene, eat_seq)
        testenv = TestEnv(scene)
        testenv.run()