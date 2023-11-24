import numpy as np
import matplotlib.pyplot as plt



class Torque2LinksArmRobot:


    def __init__(self, l1:float, l2:float, lc1:float, lc2:float, m1:float, m2:float, theta_init:np.ndarray, theta_dot_init:np.ndarray, timestep:float=0.0167, color:str='black') -> None:
        # Retrieve the constructor arguments
        self.l1         = l1
        self.l2         = l2
        self.lc1        = lc1
        self.lc2        = lc2
        self.m1         = m1
        self.m2         = m2
        self.dt         = timestep
        self.color      = color
        self.max_length = l1 + l2
        self.g          = 9.80665
        self.b1         = 0.09
        self.b2         = 0.07

        # Calculate the approximate inertial moment I1 and I2 based on lc1 and lc2
        self.I1 = (0.667)*(m1*lc1*l1)
        self.I2 = (0.667)*(m2*lc2*l2)

        # Time and state variables
        self.iter       = 0
        self.theta      = [-theta_init + np.array([[np.pi], [0.0]])]
        self.theta_dot  = [-theta_dot_init]
        self.theta_ddot = []
        self.tau        = []

        # Calculate constant values inside matrices
        # M matrix
        self.M11_const_a = m1*lc1**2 + m2*(l1**2 + lc2**2) + self.I1 + self.I2
        self.M11_const_b = 2.0*m2*l1*lc2
        self.M12_const_a = m2*lc2**2 + self.I2
        self.M12_const_b = m2*l1*lc2
        self.M21_const_a = self.M12_const_a
        self.M21_const_b = self.M12_const_b
        self.M22_const_a = m2*lc2**2 + self.I2
        self.M = lambda theta: np.array([
            [self.M11_const_a + self.M11_const_b*np.cos(theta[1].item()), self.M12_const_a + self.M12_const_b*np.cos(theta[1].item())],
            [self.M21_const_a + self.M21_const_b*np.cos(theta[1].item()), self.M22_const_a]
        ])

        # C matrix
        self.C_const = m2*l1*lc2
        self.C = lambda theta, theta_dot: np.array([
            [-2.0*self.C_const*np.sin(theta[1].item())*theta_dot[1].item(), -self.C_const*np.sin(theta[1].item())*theta_dot[1].item()],
            [self.C_const*np.sin(theta[1].item())*theta_dot[0].item(), 0]
        ])

        # b matrix
        self.b = np.array([
            [self.b1, 0],
            [0, self.b2]
        ])

        # G matrix
        self.G_const_a = m1*self.g*lc1 + m2*self.g*l1
        self.G_const_b = m2*self.g*lc2
        self.G = lambda theta: np.array([
            [self.G_const_a*np.sin(theta[0].item()) + self.G_const_b*np.sin(theta[0].item() + theta[1].item())],
            [self.G_const_b*np.sin(theta[0].item() + theta[1].item())]
        ])

        # Jacobian
        self.J = lambda theta : np.array([
            [-self.l1*np.cos(theta[0].item())-self.l2*np.cos(theta[0].item()+theta[1].item()), -self.l2*np.cos(theta[0].item() + theta[1].item())],
            [self.l1*np.sin(theta[0].item())+self.l2*np.sin(theta[0].item()+theta[1].item()), self.l2*np.sin(theta[0].item() + theta[1].item())]
        ])

        # Forward kinematics
        self.fk_x1  = lambda theta: np.array([
            [-self.l1*np.sin(theta[0].item())], 
            [-self.l1*np.cos(theta[0].item())]
        ])
        self.fk_x2  = lambda theta: self.fk_x1(theta) + np.array([
            [-self.l2*np.sin(theta[0].item() + theta[1].item())], 
            [-self.l2*np.cos(theta[0].item() + theta[1].item())]
        ])



    def torqueInput(self, torques:np.ndarray) -> None:
        self.tau.append(torques)



    def stepSimulation(self) -> None:
        # Calculate theta_ddot
        theta       = self.theta[self.iter]
        theta_dot   = self.theta_dot[self.iter]
        tau         = self.tau[self.iter]
        theta_ddot  = np.linalg.inv(self.M(theta))@(tau - (self.C(theta, theta_dot) + self.b)@theta_dot - self.G(theta))
        self.theta_ddot.append(theta_ddot)

        # Compute theta and theta_dot
        new_theta_dot = theta_dot + theta_ddot*self.dt
        new_theta     = theta + new_theta_dot*self.dt
        self.theta_dot.append(new_theta_dot)
        self.theta.append(new_theta)

        # Add iteration
        self.iter += 1



    def render(self) -> None:
        # Prepare the plot
        self.render_step    = 0
        self.fig, self.ax   = plt.subplots(1, 1)

        # Reshape the theta
        self.theta_1        = [theta[0].item() for theta in self.theta]
        self.theta_2        = [theta[1].item() for theta in self.theta]

        # Calculate the point of each joints
        self.joint1_x = [-self.l1*np.sin(theta_1) for theta_1 in self.theta_1]
        self.joint1_y = [-self.l1*np.cos(theta_1) for theta_1 in self.theta_1]
        self.joint2_x = []
        self.joint2_y = []
        for i in range(self.iter):
            self.joint2_x.append(self.joint1_x[i] - self.l2*np.sin(self.theta_1[i] + self.theta_2[i]))
            self.joint2_y.append(self.joint1_y[i] - self.l2*np.cos(self.theta_1[i] + self.theta_2[i]))



    def stepPlay(self) -> None:
        if self.render_step < self.iter:
            # Clear the axis
            self.ax.clear()

            # Define the joints
            plt.title('2-Links Arm Dynamics Simulation')
            self.ax.set_xlim(-self.max_length, self.max_length, auto=False)
            self.ax.set_ylim(-self.max_length, self.max_length, auto=False)
            self.ax.set_xlabel('x-axis')
            self.ax.set_ylabel('y-axis')
            self.ax.set_aspect('equal')
            self.ax.plot([0.0, self.joint1_x[self.render_step]], [0.0, self.joint1_y[self.render_step]], lw=6, c=self.color)
            self.ax.scatter([0.0, self.joint1_x[self.render_step]], [0.0, self.joint1_y[self.render_step]], s=200, c=self.color)
            self.ax.plot([self.joint1_x[self.render_step], self.joint2_x[self.render_step]], [self.joint1_y[self.render_step], self.joint2_y[self.render_step]], lw=6, c=self.color)
            self.ax.scatter([self.joint1_x[self.render_step], self.joint2_x[self.render_step]], [self.joint1_y[self.render_step], self.joint2_y[self.render_step]], s=200, c=self.color)

            # Increment the render step
            self.render_step += 1



class Velocity2LinksArmRobot:


    def __init__(self, l1:float, l2:float, theta_init:np.ndarray, timestep:float=0.0167, color:str='black') -> None:
        # Retrieve the constructor arguments
        self.l1         = l1
        self.l2         = l2
        self.theta      = [theta_init]
        self.theta_dot  = []
        self.dt         = timestep
        self.color      = color
        self.max_length = l1 + l2

        # Iteration
        self.iter = 0

        # Initial conditions
        self.fk_x1  = lambda theta: np.array([[-self.l1*np.sin(theta[0].item())], [self.l1*np.cos(theta[0].item())]])
        self.fk_x2  = lambda theta: self.fk_x1(theta) + np.array([[-self.l2*np.sin(theta[0].item() + theta[1].item())], [self.l2*np.cos(theta[0].item() + theta[1].item())]])
        self.x1     = [self.fk_x1(self.theta[0])]
        self.x2     = [self.fk_x2(self.theta[0])]

        # Jacobian
        self.J = lambda theta : np.array([
            [-self.l1*np.cos(theta[0].item())-self.l2*np.cos(theta[0].item()+theta[1].item()), -self.l2*np.cos(theta[0].item() + theta[1].item())],
            [-self.l1*np.sin(theta[0].item())-self.l2*np.sin(theta[0].item()+theta[1].item()), -self.l2*np.sin(theta[0].item() + theta[1].item())]
        ])



    def velocityInput(self, theta_dot:np.ndarray) -> None:
        self.theta_dot.append(theta_dot)



    def stepSimulation(self) -> None:
        # Calculate the next theta
        self.theta.append(self.theta[self.iter] + self.theta_dot[self.iter]*self.dt)

        # Add iteration
        self.iter += 1

        # Compute next task space
        self.x1.append(self.fk_x1(self.theta[self.iter]))
        self.x2.append(self.fk_x2(self.theta[self.iter]))



    def render(self) -> None:
        # Prepare the plot
        self.render_step    = 0
        self.fig, self.ax   = plt.subplots(1, 1)

        # Reshape the theta
        self.theta_1  = [theta[0].item() for theta in self.theta]
        self.theta_2  = [theta[1].item() for theta in self.theta]

        # Reshape the task spaces
        self.joint1_x = [x1[0].item() for x1 in self.x1]
        self.joint1_y = [x1[1].item() for x1 in self.x1]
        self.joint2_x = [x2[0].item() for x2 in self.x2]
        self.joint2_y = [x2[1].item() for x2 in self.x2]



    def stepPlay(self) -> None:
        if self.render_step < self.iter:
            # Clear the axis
            self.ax.clear()

            # Define the joints
            plt.title('2-Links Arm Kinematics Simulation')
            self.ax.set_xlim(-self.max_length, self.max_length, auto=False)
            self.ax.set_ylim(-self.max_length, self.max_length, auto=False)
            self.ax.set_xlabel('x-axis')
            self.ax.set_ylabel('y-axis')
            self.ax.set_aspect('equal')
            self.ax.plot([0.0, self.joint1_x[self.render_step]], [0.0, self.joint1_y[self.render_step]], lw=6, c=self.color)
            self.ax.scatter([0.0, self.joint1_x[self.render_step]], [0.0, self.joint1_y[self.render_step]], s=200, c=self.color)
            self.ax.plot([self.joint1_x[self.render_step], self.joint2_x[self.render_step]], [self.joint1_y[self.render_step], self.joint2_y[self.render_step]], lw=6, c=self.color)
            self.ax.scatter([self.joint1_x[self.render_step], self.joint2_x[self.render_step]], [self.joint1_y[self.render_step], self.joint2_y[self.render_step]], s=200, c=self.color)

            # Increment the render step
            self.render_step += 1