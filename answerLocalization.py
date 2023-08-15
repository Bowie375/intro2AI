from typing import List
import numpy as np
from utils import Particle
import pdb
import math

### 可以在这里写下一些你需要的变量和函数 ###
COLLISION_DISTANCE = 0.25#1
MAX_ERROR = 50000
MAX_WEIGHT = 0
RESAMPLE_CNT = 0
FINDED = False
### 可以在这里写下一些你需要的变量和函数 ###
def L2_norm(x_hat:np.array,x:np.array):
    return np.sqrt(((x_hat-x)**2).sum())

def norm(x_hat:np.array,x:np.array):
    return np.sqrt(((x_hat-x)**2).sum())

def generate_uniform_particles(walls, N):
    """
    输入：
    walls: 维度为(xxx, 2)的np.array, 地图的墙壁信息，具体设定请看README关于地图的部分
    N: int, 采样点数量
    输出：
    particles: List[Particle], 返回在空地上均匀采样出的N个采样点的列表，每个点的权重都是1/N
    """
    all_particles: List[Particle] = []
    #for _ in range(N):
    #    all_particles.append(Particle(_/5.0*1000.0,5.0, 1.0, float(1/N)))
    ### 你的代码 ###
    walls=list(walls)
    bottom=-1.0
    top=-1.0
    left=-1.0
    right=-1.0
    for wall in walls:
        if bottom<0:
            bottom=wall[1]
        else:
            bottom=min(bottom,wall[1])
        if top<0:
            top=wall[1]
        else:
            top=max(top,wall[1])
        if right<0:
            right=wall[0]
        else:
            right=max(right,wall[0])
        if left<0:
            left=wall[0]
        else:
            left=min(left,wall[0])
    
    cnt=N
    while cnt>0:
        #x0=np.random.rand()*(right-left)+left;
        #y0=np.random.rand()*(top-bottom)+bottom;
        y0=np.random.rand()*(top-bottom)+bottom;
        x0=np.random.rand()*(right-left)+left;
        theta=(np.random.rand()-0.5)*np.pi*2
        flag=True
        for wall in walls:
            if L2_norm(wall,np.array([x0,y0]))<COLLISION_DISTANCE:
                flag=False
                break
        if flag:
            all_particles.append(Particle(x0,y0, theta, float(1.0/N)))
            cnt-=1
    ### 你的代码 ###
    return all_particles


def calculate_particle_weight(estimated, gt):
    """
    输入：
    estimated: np.array, 该采样点的距离传感器数据
    gt: np.array, Pacman实际位置的距离传感器数据
    输出：
    weight, float, 该采样点的权重
    """
    weight = 1.0
    k=1.0#0.13
    ### 你的代码 ###
    #pdb.set_trace()
    #pdb.set_trace()
    weight=np.exp(-k*L2_norm(estimated,gt))
    ### 你的代码 ###
    
    return weight


def resample_particles(walls, particles: List[Particle]):
    """
    输入：
    walls: 维度为(xxx, 2)的np.array, 地图的墙壁信息，具体设定请看README关于地图的部分
    particles: List[Particle], 上一次采样得到的粒子，注意是按权重从大到小排列的
    输出：
    particles: List[Particle], 返回重采样后的N个采样点的列表
    """
    resampled_particles: List[Particle] = []
    #for _ in range(len(particles)):
    #    resampled_particles.append(Particle(1.0, 1.0, 1.0, 0.0))
    ### 你的代码 ###
    global MAX_WEIGHT
    global RESAMPLE_CNT
    global FINDED
    RESAMPLE_CNT+=1

    weight_list=[]
    for particle in particles:
        weight_list.append(particle.weight)

    resample_num=len(particles)
    random_resample_num=len(particles)
    #pdb.set_trace()
    MAX_WEIGHT=max(weight_list)
    print(MAX_WEIGHT)
    """
    weight_list=np.array(weight_list)
    weight_list=weight_list.cumsum(axis=0)
    weight_list=list(weight_list)
    
    N=len(particles)

    cnt=N
    while cnt>0: 
        point=np.random.rand()
        for w in weight_list:
            if w>point:
                '''
                x0=particles[weight_list.index(w)].position[0]+(np.random.rand()-0.5)*0.3#0.3
                y0=particles[weight_list.index(w)].position[1]+(np.random.rand()-0.5)*0.3#0.3
                theta=particles[weight_list.index(w)].theta+(np.random.rand()-0.5)*0.1
                '''
                x0=particles[weight_list.index(w)].position[0]#+np.random.normal(0,0.1)#0.08
                y0=particles[weight_list.index(w)].position[1]#+np.random.normal(0,0.1)#0.08
                theta=particles[weight_list.index(w)].theta#+np.random.normal(0,0.8)#0.03
                #theta=(np.random.rand()-0.5)*2*np.pi
                '''
                flag=True
                for wall in walls:
                    if L2_norm(wall,np.array([x0,y0]))<COLLISION_DISTANCE: 
                        flag=False
                        break
                if flag:
                    resampled_particles.append(Particle(x0,y0,theta,float(1.0/N)))
                    cnt-=1
                '''
                resampled_particles.append(Particle(x0,y0,theta,float(1.0/N)))
                cnt-=1
                break
    """
    for particle in particles:
        num=math.floor(resample_num*particle.weight)
        random_resample_num-=num
        for _ in range(num):
            #pdb.set_trace()
            new_x=(particle.position[0]+np.random.normal(0,0.1))#0.08
            new_y=(particle.position[1]+np.random.normal(0,0.1))#0.08
            new_theta=particle.theta+np.random.normal(0,0.05)
            new_particle=Particle(new_x,new_y,new_theta,1.0/resample_num)
            resampled_particles.append(new_particle)
    ### 你的代码 ###
    
    if RESAMPLE_CNT>10 and MAX_WEIGHT>0.5:
        FINDED=True

    if random_resample_num!=0 :
        if not (RESAMPLE_CNT>10 and MAX_WEIGHT>0.5) and not FINDED:
            random_sampled_particles=generate_uniform_particles(walls,random_resample_num)
            for particle in random_sampled_particles:
                particle.weigtht=0.5/resample_num
                resampled_particles.append(particle)
        else:
            #pdb.set_trace()
            weight_list=np.array(weight_list)
            now_particle=particles[weight_list.argmax()]
            now_particle.weight=1.0/resample_num
            for i in range(random_resample_num):
                resampled_particles.append(Particle(now_particle.position[0]+np.random.normal(0,0.1),
                                                    now_particle.position[1]+np.random.normal(0,0.1),
                                                    now_particle.theta+np.random.normal(0,0.05),
                                                    now_particle.weight))
    '''
    if random_resample_num!=0 :
        #if not (RESAMPLE_CNT>10 and MAX_WEIGHT>0.5) and not FINDED:
        random_sampled_particles=generate_uniform_particles(walls,random_resample_num)
        for particle in random_sampled_particles:
            particle.weigtht=0.5/resample_num
            resampled_particles.append(particle)
    '''    
    
    return resampled_particles

def apply_state_transition(p: Particle, traveled_distance, dtheta):
    """
    输入：
    p: 采样的粒子
    traveled_distance, dtheta: ground truth的Pacman这一步相对于上一步运动方向改变了dtheta，并移动了traveled_distance的距离
    particle: 按照相同方式进行移动后的粒子
    """
    ### 你的代码 ###
    global FINDED
    """
    #pdb.set_trace()
    if (MAX_WEIGHT>0.5 and RESAMPLE_CNT>=20) or FINDED:
        FINDED= True
        p.theta+=(dtheta+np.random.normal(0,0.2))#0.08
    
        while p.theta>np.pi or p.theta<-np.pi:
            if p.theta>np.pi:
                p.theta-=2*np.pi
            elif p.theta<-np.pi:
                p.theta+=2*np.pi 
        p.position[0]+=(traveled_distance*np.cos(p.theta)+np.random.normal(0,0.2))#0.11
        p.position[1]+=(traveled_distance*np.sin(p.theta)+np.random.normal(0,0.2))#0.11
    else:
        p.theta+=(dtheta+np.random.normal(0,1.0))#0.08
    
        while p.theta>np.pi or p.theta<-np.pi:
            if p.theta>np.pi:
                p.theta-=2*np.pi
            elif p.theta<-np.pi:
                p.theta+=2*np.pi 
        p.position[0]+=(traveled_distance*np.cos(p.theta)+np.random.normal(0,2.5))#0.11
        p.position[1]+=(traveled_distance*np.sin(p.theta)+np.random.normal(0,2.5))#0.11
    """
    p.theta+=(dtheta)#+np.random.normal(0,0.5))#0.08
    
    while p.theta>np.pi or p.theta<-np.pi:
        if p.theta>np.pi:
            p.theta-=2*np.pi
        elif p.theta<-np.pi:
            p.theta+=2*np.pi 
    p.position[0]+=(traveled_distance*np.cos(p.theta))#+np.random.normal(0,0.5))#0.11
    p.position[1]+=(traveled_distance*np.sin(p.theta))#+np.random.normal(0,0.5))#0.11
 
    ### 你的代码 ###
    return p

def get_estimate_result(particles: List[Particle]):
    """
    输入：
    particles: List[Particle], 全部采样粒子
    输出：
    final_result: Particle, 最终的猜测结果
    """
    #final_result = Particle()
    ### 你的代码 ###
    pos_list=[]
    theta_list=[]
    weight_list=[]
    for particle in particles:
        pos_list.append(particle.position)
        theta_list.append(particle.theta)
        weight_list.append(particle.weight)

    N=len(particles)
    #pdb.set_trace()
    '''
    weight_list=np.array(weight_list)
    pos_list=np.array(pos_list)
    theta_list=np.array(theta_list)
    
    pos=pos_list.sum(axis=0)/N
    theta=theta_list.sum(axis=0)/N
    '''
    _weight_list=weight_list.copy()
    _weight_list.sort(reverse=True)

    final_x=0
    final_y=0
    final_theta=0

    '''
    for i in range(3):
        #pdb.set_trace()
        pos=pos_list[weight_list.index(_weight_list[i])]
        theta=theta_list[weight_list.index(_weight_list[i])]
        final_x+=pos[0]
        final_y+=pos[1]
        final_theta+=theta

    final_x/=3
    final_y/=3
    final_theta/=3
    '''
    weight_list=np.array(weight_list)
    final_x=pos_list[weight_list.argmax()][0]
    final_y=pos_list[weight_list.argmax()][1]
    final_theta=theta_list[weight_list.argmax()]

    final_result=Particle(final_x,final_y,final_theta,1)
    ### 你的代码 ###
    return final_result