import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from model.hidden_harmonic_osc import HiddenHarmonicOsc

@eqx.filter_jit
def make_step(model, opt_state, optimizer, loss_fn):
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss

def train(model, train_data, epochs=1, lr=1e-3):
    inp = train_data["q"][:-1]
    out = train_data["q"][1:]

    def loss_fn(model):
        pred = jax.vmap(model)(inp)
        return jnp.mean((pred - out) ** 2)
    
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    for epoch in range(epochs):
        model, opt_state, loss = make_step(model, opt_state, optimizer, loss_fn)
        print(f"Epoch {epoch+1}/{epochs} | Loss {loss:.6f}")
    return model

def test(model, test_data):
    t_gt = test_data["t"]
    q_gt = test_data["q"]
    q_curr = q_gt[0]

    loss_list = []
    for q in q_gt:
        loss = jnp.linalg.norm(q_curr - q)
        q_curr = model(q_curr)
        loss_list.append(loss)

    plt.plot(t_gt, loss_list)
    plt.xlabel("Time")
    plt.ylabel("Loss")
    plt.grid()
    plt.show()

    print("Loss: ", loss)

    return 0

def get_data():
    data = np.load("data/harmonic_osc_coupled.npz")
    n_train = data["t"].shape[0] // 2

    train_data = {}
    test_data = {}
    for k,v in data.items():
        train_data[k] = v[:n_train]
        test_data[k] = v[n_train:]

    return train_data, test_data

def train_test_generic_mlp():
    train_data, test_data = get_data()
    num_dims = train_data["q"].shape[1]

    key = jax.random.PRNGKey(42)

    model = eqx.nn.MLP(
            in_size=num_dims,
            out_size=num_dims,
            width_size=64,
            depth=2,
            activation=jax.nn.tanh,
            key=key
            )

    train(model, train_data, epochs=500)
    test(model, train_data)
    test(model, test_data)

def main():
    train_data, test_data = get_data()
    num_tsteps, num_dims = train_data["q"].shape

    # Assume chain and one extra unknown osc connected to known
    edges = [[i,i+1] for i in range(num_dims-1)]
    for i in range(num_dims):
        edges.append([i, num_dims])
    edges = jnp.array(edges)

    key = jax.random.PRNGKey(69)
    amp_key, phase_key, omega_key = jax.random.split(key,3)
    num_modes = 10
    model = HiddenHarmonicOsc(
            amp=jax.random.normal(amp_key, (1,num_modes)),         # Only one hidden osc
            phase=jax.random.normal(phase_key, (1,num_modes)),         # Only one hidden osc
            omega=jax.random.normal(omega_key, num_modes),         # Only one hidden osc

            diag_params=jnp.ones(num_dims+1), # Only one hidden osc
            edge_params=jnp.ones(len(edges)),
            edges=edges
            )

    def loss_fn(model):
        return model.loss(train_data["t"], train_data["q"])

    schedule = optax.exponential_decay(
                init_value=1e-2,
                transition_steps=10000,
                decay_rate=0.9
                )
    optimizer = optax.adam(schedule)
    opt_state = optimizer.init(
            eqx.filter(model, eqx.is_array)
            )

    iters_per_log = 1000
    num_iters = 500*iters_per_log
    x_norm_list = []
    loss_list = []
    val_list = []
    for step in range(num_iters):
        model, opt_state, loss = make_step(model, opt_state, optimizer, loss_fn)
        if step % iters_per_log == 0:
            x_norm_list.append(jnp.linalg.norm(model.amp))

            loss_list.append(loss)

            num_test_steps = min(test_data["q"].shape[0], 100)
            q_pred = model.forward(train_data["t"][-2:], train_data["q"][-2:], num_test_steps)
            rmse = jnp.sqrt(jnp.mean(
                (q_pred[:,:-1] - test_data["q"][:num_test_steps,:])**2
                ))
            val_list.append(rmse)

            lr = schedule(step)
            print(f"step={step} | loss={loss:.6e} | val={rmse:.6e} | lr={float(lr):.3e}")

    print(model.get_osc_params())

    plt.plot(x_norm_list)
    plt.ylabel("Hidden Coord Norm")
    plt.yscale("log")
    plt.xlabel("Train Step")
    plt.savefig("hidden.png")
    plt.close()

    plt.plot(loss_list)
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.xlabel("Train Step")
    plt.savefig("loss.png")
    plt.close()

    plt.plot(val_list)
    plt.ylabel("RMSE on Prediction")
    plt.yscale("log")
    plt.xlabel("Train Step")
    plt.savefig("val.png")
    plt.close()

    q_pred = model.forward(train_data["t"][-2:], train_data["q"][-2:], num_test_steps)
    print("q_pred shape", q_pred.shape)
    print("omega shape", model.omega)

    err_wrt_time = jnp.linalg.norm(q_pred[:50,:-1] - test_data["q"][:50], axis=1)
    plt.plot(test_data["t"][:50], err_wrt_time)
    plt.ylabel("Error Norm")
    plt.yscale("log")
    plt.xlabel("Test Time")
    plt.grid()
    plt.savefig("test.png")
    plt.close()

if __name__ == "__main__":
    main()

