import numpy as np
import cvxpy as cp
class MPCAgent(object):

    def __init__(self, environment, gamma, safety_margin=0.9,
                 planning_steps=1):
        self.gamma = gamma
        self.safety_margin = safety_margin
        self.planning_steps=planning_steps
        self.action_space=environment.action_space
        self.num_of_state=environment.roomnum
        self.num_of_action=environment.action_space.shape[0]
        self.A_d=environment.A_d
        self.B_d=environment.B_d
        self.temp=environment.OutTemp[environment.epochs]
        self.acmap=environment.acmap

        self.GroundTemp=environment.GroundTemp[environment.epochs]
        # self.Occupancy=environment.Occupancy[environment.epochs]
        self.Occupancy=environment.Occupower
        self.ghi=environment.ghi[environment.epochs]
        self.target=environment.target

        self.problem=None

    def predict(self, environment):
        #current_state = environment.simulator.state
        self.A_d=environment.A_d
        self.B_d=environment.B_d
        self.temp=environment.OutTemp[environment.epochs]
        self.GroundTemp=environment.GroundTemp[environment.epochs]
        # self.Occupancy=environment.Occupancy[environment.epochs]

        self.Occupancy=environment.Occupower
        self.ghi=environment.ghi[environment.epochs]
        # print('OC',self.Occupancy)

        action=np.zeros((self.num_of_action))

        x0 = cp.Parameter(self.num_of_state, name='x0')
        u_max = cp.Parameter(self.num_of_action, name='u_max')
        u_min = cp.Parameter(self.num_of_action, name='u_max')

        x = cp.Variable((self.num_of_state, self.planning_steps + 1), name='x')
        u = cp.Variable((self.num_of_action, self.planning_steps), name='u')

        x0.value = environment.state[:self.num_of_state]
        u_max.value = 1.0*np.ones((self.num_of_action,))
        u_min.value = -1.0 * np.ones((self.num_of_action,))

        # x_desired=22*np.ones((self.num_of_state))
        x_desired=self.target

        obj = 0
        constr = [x[:, 0] == x0]
        # print(np.shape(self.A_d))
        # print(self.num_of_state)

        a=self.B_d[:,3:-1]@ u[:, 0]
        for t in range(self.planning_steps):
            constr += [x[:, t + 1] == self.A_d@ x[:, t].T + self.B_d[:,3:-1]@ u[:, t]+ self.B_d[:,2]*self.temp+self.B_d[:,1]*self.GroundTemp+self.B_d[:,0]*self.Occupancy+self.B_d[:,-1]*self.ghi,
                       u[:, t] <= u_max,
                       u[:, t] >= u_min,
                      #  u[-1,t] ==0,
                      #  cp.norm(u[:,t],1) <=9000/400,
                      #  u[:,t]@np.ones(self.num_of_state) >=-900/400,
                       ]

            obj += self.gamma[1]*cp.norm(cp.multiply(x[:, t],self.acmap) - x_desired*self.acmap, 2)+self.gamma[0]*24*cp.norm(u[:, t],2)
            #obj += cp.norm(u[:,t],2)
        # constr+=[u[5,9]<=0.1]
        # constr += [u <= x[:, :-1]]
        prob = cp.Problem(cp.Minimize(obj), constr)
        # prob.solve()
        prob.solve(solver='ECOS_BB')
        # print('??',cp.norm(u[:, 1],2).value*400)
        # print('?2?',x[:, 1].value)
        state=x[:, 1].value
        action=u.value[:,0]

        return action,state
