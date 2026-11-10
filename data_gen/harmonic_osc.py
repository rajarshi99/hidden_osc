import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

def create_expm_stepper(masses, K, dt):
    num_nodes = masses.shape[0]

    A = jnp.block([
        [jnp.zeros((num_nodes, num_nodes)), jnp.eye(num_nodes)],
        [-(1/masses)[:, None] * K, jnp.zeros((num_nodes, num_nodes))]
        ])

    time_step_mat = jax.scipy.linalg.expm(A * dt)

    def expm_step(carry, _):
        q, v = carry
        qv = jnp.concatenate([q, v], axis=-1)
        qv_next = time_step_mat @ qv

        q_next = qv_next[:num_nodes]
        v_next = qv_next[num_nodes:]

        return (q_next, v_next), (q_next, v_next)

    return expm_step

def rk4_step(carry, _):
    q, v, mass, K, dt = carry
    
    # Define the first-order system derivative: f(q, v) = (dq/dt, dv/dt)
    def system_derivatives(q_curr, v_curr):
        dq = v_curr
        dv = (-K @ q_curr) / mass
        return dq, dv

    # k1
    dq1, dv1 = system_derivatives(q, v)
    
    # k2
    q_k2 = q + 0.5 * dt * dq1
    v_k2 = v + 0.5 * dt * dv1
    dq2, dv2 = system_derivatives(q_k2, v_k2)
    
    # k3
    q_k3 = q + 0.5 * dt * dq2
    v_k3 = v + 0.5 * dt * dv2
    dq3, dv3 = system_derivatives(q_k3, v_k3)
    
    # k4
    q_k4 = q + dt * dq3
    v_k4 = v + dt * dv3
    dq4, dv4 = system_derivatives(q_k4, v_k4)
    
    # Combine stages
    q_next = q + (dt / 6.0) * (dq1 + 2.0 * dq2 + 2.0 * dq3 + dq4)
    v_next = v + (dt / 6.0) * (dv1 + 2.0 * dv2 + 2.0 * dv3 + dv4)
    
    return (q_next, v_next, mass, K, dt), (q_next, v_next)

def symplectic_step(carry, _):
    q, v, mass, K, dt = carry
    
    # Half-step velocity update
    acc = -K @ q / mass
    v_half = v + 0.5 * dt * acc

    # Full-step position update
    q_next = q + dt * v_half
    acc_next = -K @ q_next / mass

    # Second half-step velocity update
    v_next = v_half + 0.5 * dt * acc_next
    
    return (q_next, v_next, mass, K, dt), (q_next, v_next)

def main():
    masses = jnp.array([1.0, 2.5, 0.5, 1.8])
    spring_constants = [15.0, 30.0, 10.0]
    num_nodes = 4

    dt = 0.01
    num_steps = 5000
    t = dt * jnp.arange(num_steps)

    K = [[0]*num_nodes for _ in range(num_nodes)]
    for i in range(num_nodes - 1):
        k = spring_constants[i]
        K[i][i] += k
        K[i+1][i+1] += k
        K[i][i+1] -= k
        K[i+1][i] -= k
    K = jnp.array(K)
    K = K + jnp.eye(num_nodes) * 5.0

    # Initial Conditions (Displaced states, zero initial velocity)
    q0 = jnp.array([0.5, -0.8, 1.2, -0.3])
    v0 = jnp.zeros(num_nodes)
    init_carry = (q0, v0, masses, K, dt)
    _, (q_symp, v_symp) = jax.lax.scan(symplectic_step, init_carry, length=num_steps)
    _, (q_rk4, v_rk4) = jax.lax.scan(rk4_step, init_carry, length=num_steps)
    _, (q_expm, v_expm) = jax.lax.scan(create_expm_stepper(masses, K, dt), (q0, v0), length=num_steps)

    def total_energy(q, v):
        kinetic = 0.5 * jnp.sum(masses * (v**2), axis=-1)
        potential = 0.5 * jnp.sum(q * (q @ K), axis=-1)
        return kinetic + potential

    tot_energy_symp = total_energy(q_symp, v_symp)
    tot_energy_rk4 = total_energy(q_rk4, v_rk4)
    tot_energy_expm = total_energy(q_expm, v_expm)

    plt.plot(t, tot_energy_symp, label="symplectic step")
    plt.plot(t, tot_energy_rk4, label="rk4 step")
    plt.plot(t, tot_energy_expm, label="expm step")

    plt.xlabel("Time")
    plt.ylabel("Total Energy")

    plt.legend()
    plt.grid()
    plt.savefig("harmonic_osc.png")
    plt.close()

    return t, q_symp, q_rk4, q_expm, masses, K

if __name__ == "__main__":
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    t, q_symp, q_rk4, q_expm, M, K = main()

    data_fname = data_dir / f"harmonic_osc_coupled.npz"
    np.savez(data_fname, t=t, q=q_expm[:,1:])
    print("Saved incomplete simulation data @", data_fname)

    sys_params_fname = data_dir / f"harmonic_osc_coupled_sys_params.npz"
    np.savez(sys_params_fname, M=M, K=K)
    print("Saved sys params @", data_fname)

    print(q_expm[-1] - q_symp[-1], q_expm[-1] - q_rk4[-1])
