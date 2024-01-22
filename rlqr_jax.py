import jax
import numpy as np
import jax.numpy as jnp
from jax import debug
from jax import grad, jit, lax, random, jacfwd, jacrev
from jax.numpy.linalg import cholesky as chol
from jax.numpy.linalg import inv, eigh
import time
import numpy as np
from functools import partial
from tabulate import tabulate
import pyqtgraph as pg

from jax import config
config.update("jax_enable_x64", True) # double precision
jnp.set_printoptions(precision=6) # print options
cpu_device = jax.devices('cpu')[0]

## Switch to using white background and black foreground
pg.setConfigOption("background", "w")
pg.setConfigOption("foreground", "k")


class RobustLQRSolver:
    def __init__(self,input_mats,gens):
        A,B1,B2,C,D1,D2,P_N,Sigma,Qk,Rk = input_mats
        self.A = A
        self.B1 = B1
        self.B2= B2
        self.C = C
        self.D1 = D1
        self.D2 = D2
        self.P_N = P_N
        self.Sigma = Sigma
        self.Qk = Qk
        self.Rk = Rk
        self.N_gen,self.M_tilde_gen = gens
 
        self.N = A.shape[0]
        self.n = A[0].shape[0]
        self.m = B1[0].shape[1]
        self.z = C[0].shape[0]
        self.p = self.z
        self.w = D2[0].shape[1]
        #Reshaping the matrix blocks
        self.AB1B2 = jnp.concatenate((A,B1,B2),axis=2)
        self.CD1D2 = jnp.concatenate((C,D1,D2),axis=2)
        Z_Z_I = jnp.zeros((N,z,n+m+w))
        Z_Z_I = Z_Z_I.at[:,:,n+m:].set(jnp.eye(m))
        self.CD1D2 = jnp.concatenate((self.CD1D2,Z_Z_I),axis=1)
        
        #solver options
        self.tmax = 1e10
        self.alpha_min = 1e-12
        self.mu = 10
        self.tol_rel_gap = 1e-8
        
        #auto-diff functions
        self.grad_v_t = jit(jacfwd(self.backward_riccati_recursion,has_aux=True),device=cpu_device)
        self.H_v_t = jit(jacfwd(self.grad_v_t,has_aux=True),device=cpu_device)

        self.init_pyqtsubplots(4)

    @partial(jit, static_argnums=(0,))
    def is_neg_def(self, x):
        return ~jnp.all(jnp.isnan(jnp.diagonal(x)))
    
    #PHASE-1 EVALUATION FUNCTION
    def evalPhase1(self,lmbda,G_gen,N_gen_k,t):
        feasible = True
        N = jnp.tensordot(N_gen_k,lmbda[:p],axes=([0],[0]))
        if(jnp.min(jnp.diagonal(N)) < 0):
            feasible = False
        G = G_gen[0]
        G += jnp.tensordot(G_gen[1:],lmbda,axes=([0],[0]))
        cholG = chol(G)
        if(jnp.any(jnp.isnan(jnp.diagonal(cholG)))):
            feasible = False

        return jnp.asarray([t*lmbda[p],
                            -2*jnp.sum(jnp.log(jnp.diagonal(cholG))) - jnp.sum(jnp.log(jnp.diagonal(N)))]), feasible

    # PHASE-1 SOLVER LOOP
    def solve_phase1(self,lmbda,G_gen,N_gen_k):
        n = self.n
        w = self.w
        z = self.z
        p = self.p
        keep_optimizing = True
        t = 1

        grad_ph1 = jacfwd(self.evalPhase1,has_aux=True)
        H_ph1 = jacfwd(jacfwd(self.evalPhase1,has_aux=True),has_aux=True)

        v, feasible = self.evalPhase1(lmbda,G_gen,N_gen_k,t)
        v = jnp.sum(v)

        if not feasible:
            keep_optimizing = False

        gradients,_ = grad_ph1(lmbda,G_gen,N_gen_k,t)
        dv_add = gradients[0]
        dv = gradients[1]
        #initial t calculation by gradients of cost and bariers
        t = jnp.linalg.norm(dv) / jnp.linalg.norm(dv_add)
        i=0
        v, feasible = self.evalPhase1(lmbda,G_gen,N_gen_k,t)
        v = jnp.sum(v)

        while(keep_optimizing):

            gradients,_ = grad_ph1(lmbda,G_gen,N_gen_k,t)
            dv = jnp.sum(gradients,axis=0)
            hessians,_, = H_ph1(lmbda,G_gen,N_gen_k,t)
            d2v = jnp.sum(hessians,axis=0)

            Newton_dec = jnp.sqrt(dv @ inv(d2v) @ dv)
            accuracy = 0.0
            alpha = 1.0
            if Newton_dec < 0.25:
                accuracy = Newton_dec**2 + (n+z+p)/ t

            line_search_success = False
            while True:
                step = -alpha*(inv(d2v) @ dv)
                lmbda_alpha= lmbda + step

                v_alpha,feasible = self.evalPhase1(lmbda_alpha,G_gen,N_gen_k,t)
                v_alpha = jnp.sum(v_alpha)

                if feasible and (v_alpha <= v + (step.T.dot(dv)) * (alpha * 0.25)):
                    v= v_alpha.copy()
                    lmbda= lmbda_alpha.copy()
                    line_search_success = True
                    break
                elif abs(step.T.dot(dv)) <= 100 * 1e-16:
                    break
                else:
                    alpha = min(alpha / 2, jnp.sqrt(alpha) / jnp.sqrt(1 + Newton_dec))

            if line_search_success and (lmbda[p] < 0):
                return lmbda,True
            
            elif lmbda[p] - accuracy > 0 and accuracy > 0:
                keep_optimizing = False

            elif Newton_dec**2 < 0.25 * accuracy or not line_search_success or dv[p] < 0:
                if t < self.tmax:
                    v = v - t * lmbda[p]
                    t = self.mu * t
                    v = v + t * lmbda[p]
                elif not line_search_success:
                    keep_optimizing = False
                    return lmbda,False
            i += 1
        
        return lmbda,feasible

    def is_feasible(self):
        N = self.N
        n = self.n
        z = self.z
        m =self.m
        nw = self.w
        p = self.p
        P = jnp.zeros((N+1,n,n))
        P = P.at[N].set(self.P_N)
        lmbda = jnp.ones(p+1)
        lmbda_feas = jnp.zeros(N*p)

        QR_diag = jnp.zeros((N,n+m+w,n+m+w))
        QR_diag = QR_diag.at[:,:n,:n].set(self.Qk)
        QR_diag = QR_diag.at[:,n:n+m,n:n+m].set(self.Rk)
        dimG = n + z
        G_gen = jnp.zeros((self.p+2,dimG,dimG))
        I_0_D2_B2 = jnp.zeros((n+z,z+nw))
        I_0_D2_B2 = I_0_D2_B2.at[:z,:z].set(-jnp.eye(z))
        for k in range(self.N-1,-1,-1):
            I_0_D2_B2 = I_0_D2_B2.at[:z,z:].set(self.D2[k])
            I_0_D2_B2 = I_0_D2_B2.at[z:,z:].set(self.B2[k])
            G_gen = G_gen.at[0,z:,z:].set(inv(P[k+1]))
            N_gen_k = jnp.zeros((p,p,p))
            for i in range(p):
                N_gen_k = N_gen_k.at[i,i,i].set(1)
            G_gen = G_gen.at[1:p+1].set(I_0_D2_B2 @ self.M_tilde_gen[k][:] @ I_0_D2_B2.T)
            G = G_gen[0]
            G += jnp.tensordot(G_gen[1:p+1],lmbda[:p],axes=([0],[0]))
            G_gen = G_gen.at[p+1].set(jnp.eye(dimG))
            lmbda_k = lmbda.copy()

            if(jnp.all(jnp.isnan(jnp.diagonal(chol(G))))):
                eigs,_ =  eigh(G)
                min_eigval = min(eigs)
                min_eigval = min(min_eigval*1.1,min_eigval-0.1)
                lmbda = lmbda.at[p].set(-min_eigval)
                lmbda_k,feasible_k = self.solve_phase1(lmbda,G_gen,N_gen_k)

                if not feasible_k:
                    return lmbda_k,False
                print('Phase-I for time-step {k} is completed'.format(k=k))

            lmbda_feas = lmbda_feas.at[k*p:k*p+p].set(lmbda_k[:p])
            QSR = QR_diag[k] + self.AB1B2[k].T @ P[k+1] @ self.AB1B2[k]
            Mk = jnp.tensordot(M_tilde_gen[k][:],lmbda_k[:p],axes=([0],[0]))
            Mk = jnp.linalg.inv(Mk)
            QSR += self.CD1D2[k].T @ Mk @ self.CD1D2[k]
            Q = QSR[:n,:n]
            S = QSR[:n,n:]
            R = QSR[n:,n:]
            P = P.at[k].set(Q - S @ jnp.linalg.inv(R) @ S.T)
        print('The initial feasible point:\n',lmbda_feas)
        return lmbda_feas
    
    @partial(jit, static_argnums=(0,))
    def backward_riccati_recursion(self,lmbda,t):
        N = self.N
        n = self.n
        m = self.m
        w = self.w
        z = self.z
        p = self.p
        QR_diag = jnp.zeros((N,n+m+w,n+m+w))
        QR_diag = QR_diag.at[:,:n,:n].set(self.Qk)
        QR_diag = QR_diag.at[:,n:n+m,n:n+m].set(self.Rk)
        
        #checks neg. def. in the for loop
        def cond_fun(carry):
            k,_, cholR22,_ = carry
            return (k >= 0) & (self.is_neg_def(cholR22[k+1]))
        
        #carries out the loop if condition is true
        def body_fun(carry):
            k,P,cholR22,K = carry
            s_i = k*p
            e_i = p
            QSR = QR_diag[k] + self.AB1B2[k].T @ P[k+1] @ self.AB1B2[k]
            M = jnp.tensordot(self.M_tilde_gen[k],lax.dynamic_slice_in_dim(lmbda,s_i,e_i),axes=([0],[0]))
            M = jnp.linalg.inv(M)
            QSR += self.CD1D2[k].T @ M @ self.CD1D2[k]
            Q = QSR[:n,:n]
            S = QSR[:n,n:]
            R = QSR[n:,n:]
            R22 = R[m:,m:]
            cholR22 = cholR22.at[k].set(chol(-R22))
            L = jnp.linalg.inv(R) @ S.T
            K = K.at[k].set(L[:m,:])
            P = P.at[k].set(Q - S @ L)
            return k-1, P, cholR22,K

        P = jnp.zeros((N+1,n,n))
        K = jnp.zeros((N,m,n))
        P = P.at[-1].set(self.P_N)
        cholR22 = jnp.zeros((N,w,w))
        cholR22 = cholR22.at[-1].set(jnp.eye(w))
        N_tilde = jnp.tensordot(self.N_gen,lmbda,axes=([0],[0]))
        k, final_P,cholR22,K = lax.while_loop(
            cond_fun=cond_fun,
            body_fun=body_fun,
            init_val=(N-1, P, cholR22,K)
        )
        return jnp.asarray([t*jnp.trace(self.Sigma @ final_P[0]),
                            -2*jnp.sum(jnp.log(jnp.diagonal(cholR22,axis1=1,axis2=2))),
                            - jnp.sum(jnp.log(N_tilde.diagonal()))]), (final_P[0],K,k==-1) 
    
    def solve(self):
        N = self.N
        n = self.n
        w = self.w
        p = self.p
        keep_optimizing = True
        t = 1
        lmbda = self.is_feasible()
        v, output = self.backward_riccati_recursion(lmbda,t)
        v = jnp.sum(v)
        P0,K,feasible = output
        print("Feasibility of the problem:", feasible)
        if not feasible:
            keep_optimizing = False

        gradients,_ = self.grad_v_t(lmbda,t)
        dv_add = gradients[0]
        dv_R22 = gradients[1]
        dv_N = gradients[2]
        dv = dv_R22 + dv_N
        t = jnp.linalg.norm(dv) / jnp.linalg.norm(dv_add) / 10
        v, output = self.backward_riccati_recursion(lmbda,t)
        v = jnp.sum(v)
        P0,K,feasible = output
        i=0
        while(keep_optimizing):
            gradients,_ = self.grad_v_t(lmbda,t)
            dv = jnp.sum(gradients,axis=0)

            hessians,_, = self.H_v_t(lmbda,t)
            d2v = jnp.sum(hessians,axis=0)

            step_dir = inv(d2v) @ dv
            Newton_dec = jnp.sqrt(dv @ step_dir)
            accuracy = 0.0
            alpha = 1.0
            if Newton_dec < 0.25:
                accuracy = Newton_dec**2 + N*(w + p)/t
           
            line_search_success = False
            while True:
                step = -alpha*step_dir
                lmbda_alpha= lmbda + step

                v_alpha,output_linesearch = self.backward_riccati_recursion(lmbda_alpha,t)
                v_alpha = jnp.sum(v_alpha)
                P0_alpha,K_alpha,feasible = output_linesearch

                if feasible and (v_alpha <= v + (step.T.dot(dv)) * (alpha * 0.25)):
                    v= v_alpha.copy()
                    P0 = P0_alpha.copy()
                    lmbda= lmbda_alpha.copy()
                    K = K_alpha.copy()
                    line_search_success = True
                    break
                elif abs(step.T.dot(dv)) <= 100 * 1e-16:
                    break
                elif alpha < self.alpha_min:
                    break
                else:
                    alpha = min(alpha / 2, jnp.sqrt(alpha) / jnp.sqrt(1 + Newton_dec))
            if Newton_dec**2 < 0.25 * accuracy or not line_search_success:
                if accuracy < (1e-8) * abs(v):
                    keep_optimizing = False
                elif t < self.tmax:
                    v = v - t * jnp.trace(self.Sigma * P0)
                    t = self.mu * t
                    v = v + t * jnp.trace(self.Sigma * P0)
                elif not line_search_success:
                    keep_optimizing = False
            print("\n",
                    tabulate(
                        [
                            [
                                i,
                                v,
                                jnp.trace( self.Sigma @ P0),
                                Newton_dec,
                                alpha,
                                accuracy,
                                t,
                            ]
                        ],
                        headers=[
                            "iteration",
                            "cost",
                            "primal obj",
                            "newton dec",
                            "alpha",
                            "accuracy",
                            "t"
                        ],
                    ),
                )
            self.update_pyqtsubplots(i, v, jnp.trace( self.Sigma @ P0), alpha, Newton_dec)
            i += 1
        return lmbda, jnp.trace(self.Sigma * P0), K
    
    
    def init_pyqtsubplots(self, num_plots):
        self.win = pg.GraphicsLayoutWidget(show=True,size=(1080,360))
        self.win.setWindowTitle("Robust LQR Solver")
        pg.setConfigOptions(antialias=True)
        self.plts = [self.win.addPlot(row=0,col=i) for i in range(num_plots)]

        # plt.setAutoVisibleOnly(y=True)
        colors = ["b", "r", "k","g"]
        self.curves = [self.plts[i].plot(pen=pg.mkPen(colors[i],width=2)) for i in range(num_plots)]

        labels = ["v_t", "trace(Sigma0*P_0)", "alpha","Newton Decrement"]
        for i, lbl in enumerate(labels):
            self.plts[i].setLabel("left", lbl)
            self.plts[i].setLabel("bottom", "Iteration")
        self.data = [[] for i in range(num_plots)]

    def update_pyqtsubplots(self, i, v_t, trSP, alpha,newton_dec):

        yy = [v_t, trSP, alpha,newton_dec]
        titles = [
            "Cost with bariers",
            "Primal cost",
            "alpha (Line-Search Param.)", "Newton_dec"
        ]
        for k, y in enumerate(yy):
            self.data[k].append(yy[k])
            self.curves[k].setData(jnp.hstack(self.data[k]))
        pg.QtWidgets.QApplication.processEvents()

# Example usage
#==========================SOLVER INPUT============================

N = 10 #Horizon length
# UNCOMMENT TO SEE WHAT HAPPENS TO NEWTON DECREMENT
# N = 50 
# N = 100 

n = 4 # State dimension
m = 2  # Control input dimension
z = m
w = m
p = z

A = np.loadtxt('Ad.txt',usecols=range(n))
A = jnp.array(A[:N*n].reshape(N,n,n))

B1 = np.loadtxt('Bd.txt',usecols=range(m))
B1 = jnp.array(B1[:N*n].reshape(N,n,m))

B2 = B1

C = jnp.zeros((N,z,n))
D1 = jnp.zeros((N,z,m))
D1 = D1.at[:,:,].set(jnp.eye(z))
D2 = jnp.zeros((N,z,w))

P_N = 10*jnp.eye(n)
Sigma = 0.01*jnp.eye(n)
Qk = jnp.eye(n)
Rk = jnp.eye(m)

N_gen = jnp.zeros((N*p,N*p,N*p))
for i in range(N*p):
    N_gen = N_gen.at[i,i,i].set(1)

gamma = 0.25
M_tilde_gen = jnp.zeros((N,p, w + z, w + z))
for t in range(N):
    for i in range(p):
        M_tilde_gen = M_tilde_gen.at[t,i, i, i].set(1/(gamma**2))
        M_tilde_gen = M_tilde_gen.at[t,i, w + i, w + i].set(-1)

#==========================SOLVER INPUT============================

#Running the example
rlqr_solver = RobustLQRSolver([A,B1,B2,C,D1,D2,P_N,Sigma,Qk,Rk],[N_gen,M_tilde_gen])
lmbda_sol,optimal_cost, K = rlqr_solver.solve()
print("Optimal lambda sequence solution:\n",lmbda_sol)
print("Optimal primal cost: %0.9f" % optimal_cost)

#saving the optimal state-feedback matrix sequence
K_np = np.asarray(K)
with open('Kk.txt', 'w') as outfile:
    for Kk in K_np:
        np.savetxt(outfile,Kk,delimiter=' ')
