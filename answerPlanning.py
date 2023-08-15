import numpy as np
from typing import List
from utils import TreeNode
from simuScene import PlanningMap
import pdb

### 定义一些你需要的变量和函数 ###
STEP_DISTANCE = 1
TARGET_THREHOLD = 0.25

### 定义一些你需要的变量和函数 ###
def L2_norm(x1:np.array,x0:np.array):
    return np.linalg.norm(x1-x0)

def decide_border(walls:list):
    right=-1
    left=-1
    top=-1
    bottom=-1
    for wall in walls:
        if right<0:
            right=wall[0]
        else:
            right=max(right,wall[0])
        if left<0:
            left=wall[0]
        else:
            left=min(left,wall[0])
        if top<0:
            top=wall[1]
        else:
            top=max(top,wall[1])
        if bottom<0:
            bottom=wall[1]
        else:
            bottom=min(bottom,wall[1])
    return top,bottom,right,left

class RRT:
    def __init__(self, walls) -> None:
        """
        输入包括地图信息，你需要按顺序吃掉的一列事物位置 
        注意：只有按顺序吃掉上一个食物之后才能吃下一个食物，在吃掉上一个食物之前Pacman经过之后的食物也不会被吃掉
        """
        self.map = PlanningMap(walls)
        self.walls = walls
        
        # 其他需要的变量
        ### 你的代码 ###      
        self.maxStepNum=5
        #pdb.set_trace()
        self.top,self.bottom,self.left,self.right=decide_border(self.walls)
        self.get_target_cnt=0
        self.now_target_idx=0

        ### 你的代码 ###
        
        # 如有必要，此行可删除
        self.path = None
        
        
    def find_path(self, current_position, next_food):
        """
        在程序初始化时，以及每当 pacman 吃到一个食物时，主程序会调用此函数
        current_position: pacman 当前的仿真位置
        next_food: 下一个食物的位置
        
        本函数的默认实现是调用 build_tree，并记录生成的 path 信息。你可以在此函数增加其他需要的功能
        """
        
        ### 你的代码 ###      
        self.path = self.build_tree(current_position, next_food)

        ### 你的代码 ###
        # 如有必要，此行可删除
        
        
    def get_target(self, current_position, current_velocity):
        """
        主程序将在每个仿真步内调用此函数，并用返回的位置计算 PD 控制力来驱动 pacman 移动
        current_position: pacman 当前的仿真位置
        current_velocity: pacman 当前的仿真速度
        一种可能的实现策略是，仅作为参考：
        （1）记录该函数的调用次数
        （2）假设当前 path 中每个节点需要作为目标 n 次
        （3）记录目前已经执行到当前 path 的第几个节点，以及它的执行次数，如果超过 n，则将目标改为下一节点
        
        你也可以考虑根据当前位置与 path 中节点位置的距离来决定如何选择 target
        
        同时需要注意，仿真中的 pacman 并不能准确到达 path 中的节点。你可能需要考虑在什么情况下重新规划 path
        """
        target_pose = np.zeros_like(current_position)
        ### 你的代码 ###
 
        if self.now_target_idx>=len(self.path):
            target_pose=self.path[-1]
            return target_pose

        self.get_target_cnt+=1
        target_pose=self.path[self.now_target_idx]
        
        while L2_norm(current_position,target_pose)<TARGET_THREHOLD:
            self.now_target_idx+=1
            self.get_target_cnt=0
            if self.now_target_idx>=len(self.path):
                target_pose=self.path[-1]
                break
            #self.get_target_cnt+=1
            target_pose=self.path[self.now_target_idx]

        for target_idx in range(len(self.path)-self.now_target_idx-1):
            has_obstacle=self.map.checkline(list(current_position)
                                            ,[self.path[-(target_idx+1)][0]+0.0,self.path[-(target_idx+1)][1]+0.0])[0]
            if not has_obstacle:
                self.now_target_idx=len(self.path)-target_idx-1
                self.get_target_cnt=0
                target_pose=self.path[-(target_idx+1)]
                return target_pose

        '''
        if L2_norm(current_position,target_pose)<TARGET_THREHOLD:
            self.now_target_idx+=1
            self.get_target_cnt=0
        '''
        if self.get_target_cnt==self.maxStepNum:
            if L2_norm(current_position,self.path[self.now_target_idx])>TARGET_THREHOLD:
                #self.get_target_cnt-=1
                self.path=self.build_tree(current_position,self.path[-1])
                target_pose=self.path[0]
            else:    
                self.now_target_idx+=1
                self.get_target_cnt=0
          
        ### 你的代码 ###
        return target_pose
        
    ### 以下是RRT中一些可能用到的函数框架，全部可以修改，当然你也可以自己实现 ###
    def build_tree(self, start, goal):
        """
        实现你的快速探索搜索树，输入为当前目标食物的编号，规划从 start 位置食物到 goal 位置的路径
        返回一个包含坐标的列表，为这条路径上的pd targets
        你可以调用find_nearest_point和connect_a_to_b两个函数
        另外self.map的checkoccupy和checkline也可能会需要，可以参考simuScene.py中的PlanningMap类查看它们的用法
        """
        path = []
        graph_forward: List[TreeNode] = []
        graph_inverse: List[TreeNode] = []
        graph_forward.append(TreeNode(-1, start[0], start[1]))
        graph_inverse.append(TreeNode(-1, goal[0]+0.0 , goal[1]+0.0))
        
        self.get_target_cnt=0
        self.now_target_idx=0
        ### 你的代码 ###
        #pdb.set_trace()
        while True:
            if L2_norm(start,goal)<TARGET_THREHOLD:
                path.append(goal)
                break
            
            has_obstacle=(self.map.checkline(list(start),[goal[0]+0.0,goal[1]+0.0]))[0]
            if not has_obstacle:
                '''
                vec=goal-start
                vec_length=np.linalg.norm(vec)
                vec=vec/(vec_length+1e-6)
                gap=np.floor(vec_length)/1
                gap=int(gap)s
                for i in range(gap-1):
                    path.append(start+vec*(i+1)*2)
                '''
                #path.append(start+(goal-start)/2)
                path.append(goal)
                break
            
            if len(graph_forward)<=len(graph_inverse):
                while True:
                    x0=self.right+np.random.rand()*(self.left-self.right)
                    y0=self.bottom+np.random.rand()*(self.top-self.bottom)
                    if self.map.checkoccupy(np.array([x0,y0])):
                        continue;
                    else:
                        random_point=np.array([x0,y0])
                        break
                    
                nearest_idx,nearest_distance=self.find_nearest_point(random_point,graph_forward)
                nearest_node=graph_forward[nearest_idx]

                safeStep,newpoint=self.connect_a_to_b(nearest_node.pos,random_point)

                if safeStep:
                    new_node=TreeNode(graph_forward.index(nearest_node),newpoint[0],newpoint[1])
                    graph_forward.append(new_node)

                    if L2_norm(newpoint,goal)<TARGET_THREHOLD or not self.map.checkline(newpoint,[goal[0]+0.0,goal[1]+0.0])[0]:
                        path.append(newpoint)
                        p_index=new_node.parent_idx
                        now_node=graph_forward[p_index]

                        while now_node.parent_idx>=0:
                            path.append(now_node.pos)
                            p_index=now_node.parent_idx
                            now_node=graph_forward[p_index]

                        path.reverse()
                        path.append(goal)

                        return path
            else:
                while True:
                    x0=self.right+np.random.rand()*(self.left-self.right)
                    y0=self.bottom+np.random.rand()*(self.top-self.bottom)
                    if self.map.checkoccupy(np.array([x0,y0])):
                        continue;
                    else:
                        random_point=np.array([x0,y0])
                        break
                    
                nearest_idx,nearest_distance=self.find_nearest_point(random_point,graph_inverse)
                nearest_node=graph_inverse[nearest_idx]

                safeStep,newpoint=self.connect_a_to_b(nearest_node.pos,random_point)

                if safeStep:
                    new_node=TreeNode(graph_inverse.index(nearest_node),newpoint[0],newpoint[1])
                    graph_inverse.append(new_node)

                    if L2_norm(newpoint,start)<TARGET_THREHOLD or not self.map.checkline(newpoint,start)[0]:
                        path.append(newpoint)
                        p_index=new_node.parent_idx
                        now_node=graph_inverse[p_index]

                        while now_node.parent_idx>=0:
                            path.append(now_node.pos)
                            p_index=now_node.parent_idx
                            now_node=graph_inverse[p_index]

                        path.append(goal)

                        return path
            
            for node in graph_forward:
                for node2 in graph_inverse:
                    has_obstacle=self.map.checkline(list(node.pos),list(node2.pos))[0]
                    if not has_obstacle:
                        path_tmp=[]
                        path_tmp.clear()
                        path_tmp.append(node.pos)
                        p_index=node.parent_idx
                        now_node=graph_forward[p_index]

                        while now_node.parent_idx>=0:
                            path_tmp.append(now_node.pos)
                            p_index=now_node.parent_idx
                            now_node=graph_forward[p_index]

                        path_tmp.reverse()

                        path_tmp.append(node2.pos)
                        p_index=node2.parent_idx
                        now_node=graph_inverse[p_index]

                        while now_node.parent_idx>=0:
                            path_tmp.append(now_node.pos)
                            p_index=now_node.parent_idx
                            now_node=graph_inverse[p_index]

                        path_tmp.append(goal)

                        if len(path)==0 or len(path)>len(path_tmp):
                            path.clear()
                            path=path_tmp.copy()

            if len(path)!=0:
                return path                       
        ### 你的代码 ###
        return path

    @staticmethod
    def find_nearest_point(point, graph):
        """
        找到图中离目标位置最近的节点，返回该节点的编号和到目标位置距离、
        输入：
        point：维度为(2,)的np.array, 目标位置
        graph: List[TreeNode]节点列表
        输出：
        nearest_idx, nearest_distance 离目标位置距离最近节点的编号和距离
        """
        nearest_idx = -1
        nearest_distance = 10000000.
        ### 你的代码 ###
        for node in graph:
            if L2_norm(point,node.pos)<nearest_distance:
                nearest_distance=L2_norm(point,node.pos)
                nearest_idx=graph.index(node)
        ### 你的代码 ###
        return nearest_idx, nearest_distance
    
    def connect_a_to_b(self, point_a, point_b):
        """
        以A点为起点，沿着A到B的方向前进STEP_DISTANCE的距离，并用self.map.checkline函数检查这段路程是否可以通过
        输入：
        point_a, point_b: 维度为(2,)的np.array，A点和B点位置，注意是从A向B方向前进
        输出：
        is_empty: bool，True表示从A出发前进STEP_DISTANCE这段距离上没有障碍物
        newpoint: 从A点出发向B点方向前进STEP_DISTANCE距离后的新位置，如果is_empty为真，之后的代码需要把这个新位置添加到图中
        """
        is_empty = False
        newpoint = np.zeros(2)
        ### 你的代码 ###
        #pdb.set_trace()
        vector_ab=point_b-point_a
        vector_ab_length=np.linalg.norm(vector_ab)
        vector_ab=vector_ab/vector_ab_length

        newpoint=point_a+vector_ab*STEP_DISTANCE

        is_empty= not self.map.checkline(list(point_a),list(newpoint))[0]
        ### 你的代码 ###
        return is_empty, newpoint
