import warp as wp
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as anim


## fluid grid
H = wp.constant(101)
W = wp.constant(401)
shape = (H, W)

## cylindar
ci = wp.constant(50.0)
cj = wp.constant(80.0)
r  = wp.constant(10.0)

## constants
Re = wp.constant(50000.0)
U  = wp.constant(0.05)
D  = 2 * r
nu = U * D / Re

tau  = 3.0 * nu + 0.5
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

bi = wp.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=int)

## MRT
vec9f = wp.vec(length=9, dtype=float)
d = vec9f(1./9, 1./36, 1./36, 1./6,
          1./12, 1./6, 1./12, 1./4, 1./4)
d = wp.constant(d)

s = vec9f(1.0, 1.63, 1.14, 1.0, 1.92, 0.0, 1.92, omega, omega)
s = wp.constant(s)

mat99f = wp.mat(shape=(9, 9), dtype=float)
M = mat99f( 
            1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,
           -4.0, -1.0, -1.0, -1.0, -1.0,  2.0,  2.0,  2.0,  2.0,
            4.0, -2.0, -2.0, -2.0, -2.0,  1.0,  1.0,  1.0,  1.0,
            0.0,  1.0,  0.0, -1.0,  0.0,  1.0, -1.0, -1.0,  1.0,
            0.0, -2.0,  0.0,  2.0,  0.0,  1.0, -1.0, -1.0,  1.0,
            0.0,  0.0,  1.0,  0.0, -1.0,  1.0,  1.0, -1.0, -1.0,
            0.0,  0.0, -2.0,  0.0,  2.0,  1.0,  1.0, -1.0, -1.0,
            0.0,  1.0, -1.0,  1.0, -1.0,  0.0,  0.0,  0.0,  0.0,
            0.0,  0.0,  0.0,  0.0,  0.0,  1.0, -1.0,  1.0, -1.0
        )

Minv = mat99f( 
            1.0, -4.0,  4.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,
            1.0, -1.0, -2.0,  1.0, -2.0,  0.0,  0.0,  1.0,  0.0,
            1.0, -1.0, -2.0,  0.0,  0.0,  1.0, -2.0, -1.0,  0.0,
            1.0, -1.0, -2.0, -1.0,  2.0,  0.0,  0.0,  1.0,  0.0,
            1.0, -1.0, -2.0,  0.0,  0.0, -1.0,  2.0, -1.0,  0.0,
            1.0,  2.0,  1.0,  1.0,  1.0,  1.0,  1.0,  0.0,  1.0,
            1.0,  2.0,  1.0, -1.0, -1.0,  1.0,  1.0,  0.0, -1.0,
            1.0,  2.0,  1.0, -1.0, -1.0, -1.0, -1.0,  0.0,  1.0,
            1.0,  2.0,  1.0,  1.0,  1.0, -1.0, -1.0,  0.0, -1.0,
        )

M = wp.constant(M)
Minv = wp.constant(Minv)

## initialize rho and velocity
rho  = wp.ones(shape=shape, dtype=float)
vel  = wp.zeros(shape=shape, dtype=wp.vec2f)

fold = wp.ones(shape=(H, W, 9), dtype=float)
fnew = wp.ones(shape=(H, W, 9), dtype=float)
feqb = wp.ones(shape=(H, W, 9), dtype=float)

mask = wp.zeros(shape=(H, W), dtype=wp.bool)

## warp functions
@wp.func
def dot(x: vec9f, y: vec9f) -> float:
    ret = 0.

    for i in range(9):
        ret += x[i] * y[i]

    return ret

@wp.func
def dot(A: mat99f, x: vec9f) -> vec9f:
    y = vec9f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    for i in range(9):
        for j in range(9):
            y[i] += A[i, j] * x[j]

    return y

@wp.func
def dot(A: mat99f, B: mat99f) -> mat99f:
    C = mat99f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               )

    for i in range(9):
        for j in range(9):
            s = 0.
            s+= A[i, 0] * B[0, j]
            s+= A[i, 1] * B[1, j]
            s+= A[i, 2] * B[2, j]
            s+= A[i, 3] * B[3, j]
            s+= A[i, 4] * B[4, j]
            s+= A[i, 5] * B[5, j]
            s+= A[i, 6] * B[6, j]
            s+= A[i, 7] * B[7, j]
            s+= A[i, 8] * B[8, j]

            C[i, j] = s

    return C

@wp.func
def mul(x: vec9f, y: vec9f) -> vec9f:
    z = vec9f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    for i in range(9):
        z[i] = x[i] * y[i]

    return z

@wp.func
def add(x: vec9f, y: vec9f) -> vec9f:
    z = vec9f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    for i in range(9):
        z[i] = x[i] + y[i]

    return z

@wp.func
def sub(x: vec9f, y: vec9f) -> vec9f:
    z = vec9f(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    for i in range(9):
        z[i] = x[i] - y[i]

    return z

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

@wp.func
def toMoment(f: vec9f) -> vec9f:
    return dot(M, f)

@wp.func
def fromMoment(m: vec9f) -> vec9f:
    dm = mul(d, m)
    return dot(Minv, dm)


## warp kernels
@wp.kernel
def createCylindar(mask: wp.array2d(dtype=wp.bool),
                   ci: float,
                   cj: float,
                   r: float):
    i, j = wp.tid()

    fi = float(i)
    fj = float(j)
    d = wp.sqrt((fi - ci)*(fi - ci) + (fj - cj)*(fj - cj))

    if d <= r:
        mask[i, j] = True
    else:
        mask[i, j] = False

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
def collide(feqb: wp.array3d(dtype=float),
            fold: wp.array3d(dtype=float)):

    i, j = wp.tid()
    
    fVec = vec9f()
    feqbVec = vec9f()
    
    for k in range(9):
        fVec[k] = fold[i, j, k]
        feqbVec[k] = feqb[i, j, k]
    
    ## moment space
    m    = toMoment(fVec)
    meq  = toMoment(feqbVec)

    for k in range(9):
        m[k] = (1.0 - s[k]) * m[k] + s[k] * meq[k]
    
    ## back to fluid
    fm = fromMoment(m)

    for k in range(9):
        fold[i, j, k] = fm[k]

@wp.kernel
def stream(e: wp.array1d(dtype=wp.vec2i),
           fold: wp.array3d(dtype=float),
           fnew: wp.array3d(dtype=float)):

    i, j = wp.tid()

    ## skip grid boundary
    if atGridBoundary(i, j):
        return
    
    for k in range(9):
        ip = i - e[k][0]
        jp = j - e[k][1]
        
        fnew[i, j, k] = fold[ip, jp, k]

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
                      mask: wp.array2d(dtype=wp.bool),
                      feqb: wp.array3d(dtype=float),
                      fold: wp.array3d(dtype=float)):
    """Apply boundary conditions on grid.
    """    
    i, j = wp.tid()

    if not atGridBoundary(i, j) and not mask[i, j]:
        ## update equilibrium
        for k in range(9):
            feqb[i, j, k] = feq(w[k], rho[i, j], wp.vec2f(e[k]), vel[i, j])
        
        return

    inb = i
    jnb = j

    if atGridBoundary(i, j):
        ## update top boundary
        if i == 0:
            vel[i, j][0] = 0.0
            vel[i, j][1] = 0.0

            inb = i+1

            fold[i, j, 4] = fold[i, j, 2]
            fold[i, j, 7] = fold[i, j, 5]
            fold[i, j, 8] = fold[i, j, 6]
        
        ## update bottom boundary
        elif i == H-1:
            vel[i, j][0] = 0.0
            vel[i, j][1] = 0.0

            inb = i-1       

            fold[i, j, 2] = fold[i, j, 4]
            fold[i, j, 5] = fold[i, j, 7]
            fold[i, j, 6] = fold[i, j, 8]     

        ## update left boundary
        elif j == 0:
            vel[i, j][0] = 0.0
            vel[i, j][1] = U

            jnb = j+1            

        ## update right boundary
        elif j == W - 1:
            vel[i, j][0] = vel[i, j-1][0]
            vel[i, j][1] = vel[i, j-1][1]

            jnb = j-1

    ## update cylinder obstacle
    elif mask[i, j]:
        vel[i, j][0] = 0.0
        vel[i, j][1] = 0.0

        if i <= ci and j <= cj:
            inb = i-1
            jnb = j-1
        
        elif i <= ci and j > cj:
            inb = i-1
            jnb = j+1
        
        elif i > ci and j <= cj:
            inb = i+1
            jnb = j-1
                
        else:
            inb = i+1
            jnb = j+1

    ## update rho
    rho[i, j] = rho[inb, jnb]

    ## update equilibrium
    for k in range(9):
        f = feq(w[k], rho[i, j], wp.vec2f(e[k]), vel[i, j])
        fold[i, j, k] = f
        feqb[i, j, k] = f

def step():
    global fnew, fold, feqb
    
    with wp.ScopedTimer("Collide and Stream", active=False):
        wp.launch(kernel=collide,
                    dim=shape,
                    inputs=[feqb, fold])
        
        wp.launch(kernel=stream,
                    dim=shape,
                    inputs=[e, fold, fnew])
    
    with wp.ScopedTimer("Update Fluid", active=False):
        wp.launch(kernel=updateFluid,
                    dim=shape,
                    inputs=[rho, vel, e, fnew])
        
    ## swap fold and fnew
    fold, fnew = fnew, fold
    
    with wp.ScopedTimer("Boundary Condition", active=False):
        wp.launch(kernel=boundaryCondition,
                    dim=shape,
                    inputs=[rho, vel, e, w, mask, feqb, fold])

## rendering
NUM_STEP   = 80000
SUB_STEP   = 80
NUM_FRAMES = NUM_STEP // SUB_STEP

def renderFrame(num_frame=None, frame=None):
    ## sub-steps simulation
    with wp.ScopedTimer("Rendering Frame", active=True):
        for _ in range(SUB_STEP):
            step()
    
    ## update frame
    vel_numpy = vel.numpy()
    vel_mag = np.sum(vel_numpy**2, axis=2)
    vel_mag = np.sqrt(vel_mag)

    frame.set_array(vel_mag)
    fig.tight_layout()
    
    return (frame,)

## launch kernel
wp.launch(kernel=createCylindar,
          dim=shape,
          inputs=[mask, ci, cj, r])

wp.launch(kernel=initFluid,
          dim=shape,
          inputs=[rho, vel, e, w, fold, fnew, feqb])

## start rendering
fig = plt.figure(figsize=(8,2))
ax  = plt.gca()
ax.set_aspect('equal')

frame = plt.imshow(wp.zeros_like(rho).numpy(), 
                    animated=True,
                    interpolation="antialiased")

plt.xticks([])
plt.yticks([])
fig.tight_layout()

frame.set_norm(matplotlib.colors.Normalize(0.0, 0.15))

seq = anim.FuncAnimation(
    fig,
    renderFrame,
    fargs=(frame,),
    frames=NUM_FRAMES,
    blit=True,
    interval=2,
    repeat=False,
)

plt.show()
# seq.save("turbulence.gif", writer="ffmpeg", fps=20, dpi=100)