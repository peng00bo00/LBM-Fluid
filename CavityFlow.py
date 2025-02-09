import warp as wp
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as anim

## fluid grid
H = wp.constant(256)
W = wp.constant(256)
L = H - 1.0
shape = (H, W)

## constants
Re = wp.constant(1000.0)
U  = wp.constant(0.1)
nu = U * L / Re
tau = 3.0 * nu + 0.5
omega = 1.0 / tau

## D2Q9 model
e = wp.array([[0, 0], 
              [1, 0], [0, 1], [-1, 0], [0, -1], 
              [1, 1], [-1, 1], [-1, -1], [1, -1]], 
              dtype=wp.vec2i)

w = wp.array([4/9, 
              1/9, 1/9, 1/9, 1/9, 
              1/36, 1/36, 1/36, 1/36], 
              dtype=float)

## rho and velocity
rho = wp.ones(shape=shape, dtype=float)
vel = wp.zeros(shape=shape, dtype=wp.vec2f)

## distribution function
fold = wp.ones(shape=(H, W, 9), dtype=float)
fnew = wp.ones(shape=(H, W, 9), dtype=float)
feqb = wp.ones(shape=(H, W, 9), dtype=float)

## warp functions
@wp.func
def atGridBoundary(i: int, j: int) -> bool:
    """Check if [i, j] is at the grid boundary.
    """

    if i == 0 or i == H - 1:
        return True
    if j == 0 or j == W - 1:
        return True

    return False

@wp.func
def feq(w: float, rho: float, e: wp.vec2f, u: wp.vec2f) -> float:
    eu = wp.dot(e, u)
    uv = wp.dot(u, u)
    f = w * rho * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * uv)

    return f


## warp kernels
@wp.kernel
def initFluid(rho: wp.array2d(dtype=float), 
              vel: wp.array2d(dtype=wp.vec2f),
              e: wp.array1d(dtype=wp.vec2i),
              w: wp.array1d(dtype=float),
              fold: wp.array3d(dtype=float),
              fnew: wp.array3d(dtype=float),
              feqb: wp.array3d(dtype=float)):
    
    i, j = wp.tid()

    vel[i, j][0] = 0.0
    vel[i, j][1] = 0.0
    rho[i, j] = 1.0

    for k in range(9):
        f = feq(w[k], rho[i, j], wp.vec2f(e[k]), vel[i, j])
        fold[i, j, k] = f
        fnew[i, j, k] = f
        feqb[i, j, k] = f

@wp.kernel
def solveEquilibrium(rho: wp.array2d(dtype=float), 
                     vel: wp.array2d(dtype=wp.vec2f),
                     e: wp.array1d(dtype=wp.vec2i),
                     w: wp.array1d(dtype=float),
                     f: wp.array3d(dtype=float)):
    
    i, j = wp.tid()

    for k in range(9):
        f[i, j, k] = feq(w[k], rho[i, j], wp.vec2f(e[k]), vel[i, j])


@wp.kernel
def collideStream(e: wp.array1d(dtype=wp.vec2i),
                  fold: wp.array3d(dtype=float),
                  feqb: wp.array3d(dtype=float),
                  fnew: wp.array3d(dtype=float)):

    i, j = wp.tid()

    ## skip grid boundary
    if atGridBoundary(i, j):
        return

    for k in range(9):
        ip = i - e[k][0]
        jp = j - e[k][1]
        
        fnew[i, j, k] = (1.0 - omega) * fold[ip, jp, k] + omega * feqb[ip, jp, k]

@wp.kernel
def updateFluid(rho: wp.array2d(dtype=float), 
                vel: wp.array2d(dtype=wp.vec2f),
                e: wp.array1d(dtype=wp.vec2i),
                fnew: wp.array3d(dtype=float)):
    
    i, j = wp.tid()

    ## skip grid boundary
    if atGridBoundary(i, j):
        return
    
    ## reset rho and velocity
    rho[i, j] = 0.
    vel[i, j][0] = 0.
    vel[i, j][1] = 0.

    for k in range(9):
        rho[i, j] += fnew[i, j, k]
        vel[i, j] += fnew[i, j, k] * wp.vec2f(e[k])
    
    vel[i, j] /= rho[i, j]


@wp.kernel
def boundaryCondition(rho: wp.array2d(dtype=float), 
                      vel: wp.array2d(dtype=wp.vec2f),
                      e: wp.array1d(dtype=wp.vec2i),
                      w: wp.array1d(dtype=float),
                      fold: wp.array3d(dtype=float)):
    """Apply boundary conditions on grid.
    """
    i, j = wp.tid()

    if not atGridBoundary(i, j):
        return
    
    inb = i
    jnb = j

    ## update top boundary
    if i == 0:
        vel[i, j][0] = 0.0
        vel[i, j][1] = U

        inb = i+1

    ## update bottom boundary
    elif i == H-1:
        vel[i, j][0] = 0.0
        vel[i, j][1] = 0.0
        
        inb = i-1

    ## update left boundary
    elif j == 0:
        vel[i, j][0] = 0.0
        vel[i, j][1] = 0.0

        jnb = j+1

    ## update right boundary
    elif j == W - 1:
        vel[i, j][0] = 0.0
        vel[i, j][1] = 0.0

        jnb = j-1
    
    ## update rho
    rho[i, j] = rho[inb, jnb]
    
    ## update equilibrium
    for k in range(9):
        fold[i, j, k] = feq(w[k], rho[i, j], wp.vec2f(e[k]), vel[i, j]) \
                        - feq(w[k], rho[inb, jnb], wp.vec2f(e[k]), vel[inb, jnb]) \
                        + fold[inb, jnb, k]

## rendering
NUM_STEP   = 100000
SUB_STEP   = 500
NUM_FRAMES = NUM_STEP // SUB_STEP

def step():
    global fnew, fold, feqb

    with wp.ScopedTimer("Solve Equilibrium", active=False):
        wp.launch(kernel=solveEquilibrium,
                dim=shape,
                inputs=[rho, vel, e, w, feqb])
    
    with wp.ScopedTimer("Collide and Stream", active=False):
        wp.launch(kernel=collideStream,
                    dim=shape,
                    inputs=[e, fold, feqb, fnew])
    
    with wp.ScopedTimer("Update Fluid", active=False):
        wp.launch(kernel=updateFluid,
                    dim=shape,
                    inputs=[rho, vel, e, fnew])
        
    ## swap fold and fnew
    fold, fnew = fnew, fold
    
    with wp.ScopedTimer("Boundary Condition", active=False):
        wp.launch(kernel=boundaryCondition,
                    dim=shape,
                    inputs=[rho, vel, e, w, fold])

def renderFrame(num_frame=None, frame=None):
    ## sub-steps simulation
    for _ in range(SUB_STEP):
        step()
    
    ## update frame
    vel_numpy = vel.numpy()
    vel_mag = np.sum(vel_numpy**2, axis=2)
    vel_mag = np.sqrt(vel_mag)

    frame.set_array(vel_mag)
    
    return (frame,)


## launch kernel
wp.launch(kernel=initFluid,
          dim=shape,
          inputs=[rho, vel, e, w, fold, fnew, feqb])

## start rendering
fig = plt.figure()
frame = plt.imshow(wp.zeros_like(rho).numpy(), 
                    animated=True,
                    interpolation="antialiased")

plt.xticks([])
plt.yticks([])
fig.tight_layout()

frame.set_norm(matplotlib.colors.Normalize(0.0, 0.1))

seq = anim.FuncAnimation(
    fig,
    renderFrame,
    fargs=(frame,),
    frames=NUM_FRAMES,
    blit=True,
    interval=50,
    repeat=False,
)

# plt.show()
seq.save("cavity.gif", writer="ffmpeg", fps=20, dpi=100)