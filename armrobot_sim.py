import numpy as np
import matplotlib.pyplot as plt




class ArmRobot2D:



    def __init__(self, link_num:int, link_len:list, theta_init:np.ndarray, timestep:float=0.0167, color:str='black') -> None:
        # Variables
        self.link_num   = link_num
        self.link_len   = link_len
        self.theta      = theta_init
        self.theta_dot  = np.zeros((self.link_num, 1))
        self.dt         = timestep
        self.time       = 0
        self.color      = color

        # Additional variables
        self.theta_his      = [theta_init]
        self.theta_dot_his  = []
        self.state_x_his    = []
        self.state_y_his    = []
        self.max_len        = 0
        for length in self.link_len: 
            self.max_len += length

        # Throw errors if the value does not match
        if len(self.link_len) != self.link_num:
            raise Exception("The length of the link length list must equal to the number of links declared")
        if len(self.theta) != self.link_num:
            raise Exception("The number of joints (theta) must equal to the number of links declared")



    def forwardKinematics(self, theta:np.ndarray) -> np.ndarray:
        # Initial variables
        link_x          = np.zeros((self.link_num, 2))
        theta_part_sum  = [0 for i in range(self.link_num)]

        # Iterate through all links
        for link_iter in range(self.link_num):

            # Calculate theta_part_sum, which is the partial consecutive sum of joint value
            if link_iter == 0:
                theta_part_sum[link_iter] = theta[link_iter].item()
            else:
                theta_part_sum[link_iter] = theta_part_sum[link_iter - 1] + theta[link_iter].item()
            
            # Calculate the forward kinematics for each link
            link_x[link_iter][0] = self.link_len[link_iter]*np.cos(theta_part_sum[link_iter])
            link_x[link_iter][1] = self.link_len[link_iter]*np.sin(theta_part_sum[link_iter])
            if link_iter != 0:
                link_x[link_iter][0] += link_x[link_iter - 1][0]
                link_x[link_iter][1] += link_x[link_iter - 1][1]

        # Return link_x
        return link_x
    


    def jacobianMat(self, theta:np.ndarray) -> np.ndarray:
        # Initial variables
        jacobian_mat    = np.zeros((2, self.link_num))
        theta_part_sum  = [0 for i in range(self.link_num)]

        # Iterate through all links once more
        for link_iter in range(self.link_num):
            rev_link_iter = self.link_num - (link_iter + 1)

            # Calculate theta_part_sum
            if link_iter == 0:
                for theta_iter in range(rev_link_iter + 1):
                    theta_part_sum[rev_link_iter] += theta[theta_iter].item()
            else:
                theta_part_sum[rev_link_iter] = theta_part_sum[rev_link_iter + 1] - theta[rev_link_iter + 1].item()

            # Calculate the 'derivative' forward kinematics
            jacobian_mat[0][rev_link_iter] = -self.link_len[rev_link_iter]*np.sin(theta_part_sum[rev_link_iter])
            jacobian_mat[1][rev_link_iter] = self.link_len[rev_link_iter]*np.cos(theta_part_sum[rev_link_iter])
            if link_iter != 0:
                jacobian_mat[0][rev_link_iter] += jacobian_mat[0][rev_link_iter + 1]
                jacobian_mat[1][rev_link_iter] += jacobian_mat[1][rev_link_iter + 1]

        # Return the jacobian
        return jacobian_mat



    def velocityInput(self, theta_dot:np.ndarray) -> None:
        self.theta_dot = theta_dot



    def stepSimulation(self) -> None:
        # Calculate the next theta
        self.theta += self.theta_dot*self.dt

        # Calculate the forward kinematics
        fkx = self.forwardKinematics(self.theta)

        # Save the state
        self.theta_his.append(self.theta)
        self.theta_dot_his.append(self.theta_dot)
        self.state_x_his.append([0.0] + list(fkx.T[0]))
        self.state_y_his.append([0.0] + list(fkx.T[1]))



    def beginRender(self) -> None:
        # Prepare the plot
        self.fig, self.ax   = plt.subplots(1, 1)
        self.render_step    = 0



    def stepPlay(self) -> None:
        # Clear the axis
        self.ax.clear()

        # Define the joints
        plt.title('2-Links Arm Kinematics Simulation')
        self.ax.set_xlim(-self.max_len, self.max_len, auto=False)
        self.ax.set_ylim(-self.max_len, self.max_len, auto=False)
        self.ax.set_xlabel('x-axis')
        self.ax.set_ylabel('y-axis')
        self.ax.set_aspect('equal')
        self.ax.plot(self.state_x_his[self.render_step], self.state_y_his[self.render_step], lw=4, c=self.color)
        self.ax.scatter(self.state_x_his[self.render_step], self.state_y_his[self.render_step], s=150, c=self.color)

        # Increment the render step
        self.render_step += 1




class PointTrajectory:



    def __init__(self, point1:tuple, point2:tuple, velocity:float) -> None:
        # Variables
        self.pnt1   = point1
        self.pnt2   = point2
        self.vel    = velocity
        self.dist   = np.sqrt((self.pnt1[0] - self.pnt2[0])**2. + (self.pnt1[1] - self.pnt2[1])**2.)
        self.tot_t  = self.dist/self.vel
        
        # Latching variables
        self.IS_DONE    = False
        self.IS_NEW     = True
        self.start_t    = 0

    

    def getVal(self, time:float) -> tuple:
        # Latch the call
        if self.IS_NEW: 
            self.start_t = time
            self.IS_NEW  = False

        if not self.IS_DONE:
            # Shifted time
            t = time - self.start_t

            # If t is bigger than total time
            if t > self.tot_t:
                self.IS_DONE = True
                return (0, 0)
            else:
                val_x = (1. - (t/self.tot_t))*self.pnt1[0] + (t/self.tot_t)*self.pnt2[0]
                val_y = (1. - (t/self.tot_t))*self.pnt1[1] + (t/self.tot_t)*self.pnt2[1]
                return (val_x, val_y)
            
    

    def reset(self) -> None:
        self.IS_DONE = False
        self.IS_NEW  = True




class Sequencer:



    def __init__(self, point_trajectories:list, loop:bool) -> None:
        # Variables
        self.pnt_trj    = point_trajectories
        self.seg_num    = len(point_trajectories)
        self.counter    = 0
        self.IS_DONE    = False
        self.IS_LOOP    = loop



    def getVal(self, time:float) -> tuple:
        # Initiates variables
        loop = True
        res  = (0, 0)

        # Check for available segments
        while loop and not self.IS_DONE:
            res = self.pnt_trj[self.counter].getVal(time)

            if self.pnt_trj[self.counter].IS_DONE:
                self.counter += 1

                if self.counter == self.seg_num:

                    if self.IS_LOOP:
                        self.counter = 0
                        for seg in self.pnt_trj: seg.reset()

                    else:
                        loop = False
                        self.IS_DONE = True

            else:
                loop = False

        return res