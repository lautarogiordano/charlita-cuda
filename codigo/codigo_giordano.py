import numpy as np
import time
import warnings
import os
from numba import cuda
from numba.core.errors import (
    NumbaPerformanceWarning,
    NumbaDeprecationWarning,
    NumbaPendingDeprecationWarning,
)

# Filtro algunos warnings que tira numba
warnings.simplefilter("ignore", category=NumbaPerformanceWarning)
warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)

from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float64


# La parte más importante del código, estas funciones bloquean/desbloquean
# el acceso a memoria de las componentes i, j del array w
# Notar que primero se bloquea la componente con índice menor, sin importar
# el orden en el que le paso i y j
@cuda.jit(device=True)
def double_lock(mutex, i, j):
    first, second = (i, j) if i < j else (j, i)

    while cuda.atomic.cas(mutex, first, 0, 1) != 0:
        pass
    while cuda.atomic.cas(mutex, second, 0, 1) != 0:
        pass

    cuda.threadfence()


@cuda.jit(device=True)
def double_unlock(mutex, i, j):
    cuda.threadfence()
    cuda.atomic.exch(mutex, j, 0)
    cuda.atomic.exch(mutex, i, 0)


#################
# Funciones GPU #
#################


# Kernel
@cuda.jit
def stream_exchange(w, r, mutex, n_agents, wmin, f, mcs, rng_state):
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)

    if tid >= n_agents:
        return

    # Evolución temporal dentro del kernel
    for t in range(mcs):
        # Esto es importante. No siempre vale que un hilo se encarga de un
        # agente. Yo puedo elegir arbitrariamente la cantidad de hilos en mi
        # grid. Si esta es menor que la cantidad de agentes, entonces algunos
        # hilos se van a encargar de más de un agente. En esos casos el hilo
        # tid se va a encargar del agente tid y de los agentes tid + n * stride,
        # donde stride es el tamaño de la grilla. Esto lo hago porque quizás
        # una grilla más chica corre más rápido que una adecuada al tamaño del
        # sistema
        for i in range(tid, n_agents, stride):
            # Elijo un oponente aleatorio para el agente i
            j = int(xoroshiro128p_uniform_float64(rng_state, i) * (n_agents))

            # Chequeo que ambos agentes tengan riqueza mayor que la mínima
            if j != i and w[i] > wmin and w[j] > wmin:
                # Bloqueo w[i] y w[j]
                double_lock(mutex, i, j)

                # Interacción de Yard-Sale
                dw = min(r[i] * w[i], r[j] * w[j])

                # Probabilidad -> número aleatorio
                prob_win = xoroshiro128p_uniform_float64(rng_state, i)
                # Probabilidad de que gane el agente i.
                p = 0.5 + f * (w[j] - w[i]) / (w[i] + w[j])
                # Intercambio
                dw = dw if prob_win <= p else -dw

                w[i] += dw
                w[j] -= dw

                # Desbloqueo
                double_unlock(mutex, i, j)

        cuda.syncthreads()


# Funcion que crea los datos iniciales, streams y semillas.
def run_gpu(
    n_streams, n_agents, tpb=32, bpg=512, wmin=1e-17, f=0.0, mcs=100, save=False
):
    with cuda.defer_cleanup():
        # Si pido un solo stream, uso el default
        if n_streams == 1:
            streams = [cuda.default_stream()]
        else:
            streams = [cuda.stream() for _ in range(n_streams)]

        # Inicializo riquezas, riesgos, mutexes y estados aleatorios
        wealths = [
            np.random.rand(n_agents).astype(np.float64) for _ in range(n_streams)
        ]

        risks = [np.random.rand(n_agents).astype(np.float64) for _ in range(n_streams)]

        mutexes = [np.zeros((n_agents,), dtype=np.int32) for _ in range(n_streams)]

        rng_states = [
            create_xoroshiro128p_states(n_agents, seed=time.time())
            for _ in range(n_streams)
        ]

        # Normalizo riquezas
        for wi in wealths:
            wi /= np.sum(wi)

        for i, (stream, wi, ri, mutex, rng_state) in enumerate(
            zip(streams, wealths, risks, mutexes, rng_states)
        ):
            with cuda.pinned(wi, ri, mutex):
                # Paso los datos a device en su correspondiente stream
                dev_w = cuda.to_device(wi, stream=stream)
                dev_r = cuda.to_device(ri, stream=stream)
                dev_m = cuda.to_device(mutex, stream=stream)

                stream_exchange[bpg, tpb, stream](
                    dev_w, dev_r, dev_m, n_agents, wmin, f, mcs, rng_state
                )

                dev_w.copy_to_host(wi, stream=stream)

            # Limpio memoria en el device
            del dev_w, dev_r, dev_m

        cuda.synchronize()

        if save:
            filepath = os.path.join(os.getcwd(), "data", f"w_f={f}")
            np.save(filepath, np.array(wealths))


#################
# Funciones CPU #
#################


# Función que devuelve un array de oponentes (no permite que oponente[i]==i)
def get_opps_cpu(n_agents):
    random_array = np.random.randint(0, n_agents, n_agents)
    indices = np.arange(0, n_agents)
    random_array = np.where(
        random_array == indices, (random_array + 1) % n_agents, random_array
    )
    return random_array


# Equivalente al kernel pero en cpu
def cpu_exchange(n_agents, w, r, wmin, f, mcs):
    opps = np.arange(0, n_agents)

    for t in range(mcs):
        opps = get_opps_cpu(n_agents)

        for i, j in enumerate(opps):
            if j != i and w[i] > wmin and w[j] > wmin:
                dw = min(r[i] * w[i], r[j] * w[j])

                prob_win = np.random.rand()
                p = 0.5 + f * (w[j] - w[i]) / (w[i] + w[j])
                dw = dw if prob_win <= p else -dw

                w[i] += dw
                w[j] -= dw


def run_cpu(n_agents, wmin=1e-17, f=0.0, mcs=100):
    w = np.random.rand(n_agents)
    w /= np.sum(w)
    r = np.random.rand(n_agents)

    cpu_exchange(n_agents, w, r, wmin, f, mcs)


########
# Main #
########

n_streams = 1000
n_agents = 10000
mcs = 5000
f = 0.2

t0 = time.time()

### Corrida de prueba (y guardamos datos al final)
print("Testing cpu: ")
run_cpu(n_agents, f=f, mcs=mcs)

t1 = time.time()
print(f"time: {t1-t0:.3f} s")

# warm up
run_gpu(n_streams=1000, n_agents=n_agents, f=f, mcs=10)

t2 = time.time()

print("Testing gpu 1 stream: ")
run_gpu(n_streams=1, n_agents=n_agents, f=f, mcs=mcs)

t3 = time.time()
print(f"time: {t3-t2:.3f} s")

print(f"Testing gpu {n_streams} streams: ")
run_gpu(n_streams=n_streams, n_agents=n_agents, f=f, mcs=mcs, save=False)
cuda.synchronize()

tf = time.time()
print(f"time: {tf-t3:.3f} s")

### Corrida para distintas configuraciones
# configs = [[32, 64], [32, 128], [32, 256], [32, 512], [32, 1024],
#            [64, 64], [64, 128], [64, 256], [64, 512], [64, 1024],
#            [128, 64], [128, 128], [128, 256], [128, 512], [128, 1024],
#            [256, 64], [256, 128], [256, 256], [256, 512], [256, 1024],
#            [512, 64], [512, 128], [512, 256], [512, 512], [512, 1024],
#            [1024, 64], [1024, 128], [1024, 256], [1024, 512], [1024, 1024],
#         ]
# mcs = 10000
# print("--------------------------------")
# print("Prueba de configuraciones")
# print("tpb\tbpg\tt_gpu\tt_gpu_streams (10000 agents, 5000 mcs)")
# print("--------------------------------")

# for tpb, bpg in configs:
#     t0 = time.time()
#     run_gpu(n_streams=1, n_agents=n_agents, tpb=tpb, bpg=bpg, f=f, mcs=mcs)

#     t1 = time.time()

#     run_gpu(n_streams=1000, n_agents=n_agents, tpb=tpb, bpg=bpg, f=f, mcs=mcs)
#     t2 = time.time()

#     print(f"{tpb},\t{bpg},\t{t1 - t0:.3f},\t{t2 - t1:.3f}")

### Corrida para distintos tamaños (n_agents)
agents_set = [10, 30, 50, 100, 300, 500, 1000, 3000, 5000, 10000, 30000, 50000, 100000]
mcs = 5000
print("--------------------------------")
print("Prueba de agentes")
print("n_agents\tt_cpu\tt_gpu\tt_gpu_streams (5000 mcs)")
print("--------------------------------")

for n_agents in agents_set:
    t0 = time.time()

    run_cpu(n_agents, f=f, mcs=mcs)

    t1 = time.time()

    run_gpu(n_streams=1, n_agents=n_agents, f=f, mcs=mcs)

    t2 = time.time()

    run_gpu(n_streams=n_streams, n_agents=n_agents, f=f, mcs=mcs, save=True)

    tf = time.time()

    print(f"{n_agents},\t{t1-t0:.3f},\t{t2-t1:.3f},\t{tf-t2:.3f}\t")

### Corrida para varios mcs
mcs_set = [10, 30, 100, 300, 500, 1000, 3000, 5000, 10000, 30000, 50000]
n_agents = 10000
print("--------------------------------")
print("Prueba de mcs")
print("MCS\tt_cpu\tt_gpu\tt_gpu_streams (10000 agents)")
print("--------------------------------")
for mcs in mcs_set:
    t0 = time.time()

    run_cpu(n_agents, f=f, mcs=mcs)

    t1 = time.time()

    run_gpu(n_streams=1, n_agents=n_agents, f=f, mcs=mcs)

    t2 = time.time()

    run_gpu(n_streams=n_streams, n_agents=n_agents, f=f, mcs=mcs, save=True)

    tf = time.time()

    print(f"{mcs},\t{t1-t0:.3f},\t{t2-t1:.3f},\t{tf-t2:.3f}")
