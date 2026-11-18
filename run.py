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

    model = HiddenHarmonicOsc(
            x=jnp.zeros((num_tsteps,1)),       # Only one hidden osc
            diag_params=jnp.zeros(num_dims+1), # Only one hidden osc
            # diag_params=jnp.array([0.1, 0.2, 0.9]),
            edge_params=jnp.zeros(len(edges)),
            edges=edges
            )

    def loss_fn(model):
        return model.loss(train_data["t"], train_data["q"])

    optimizer = optax.adam(1e-2)
    opt_state = optimizer.init(
            eqx.filter(model, eqx.is_array)
            )

    x_norm_list = []
    loss_list = []
    for step in range(10000):
        model, opt_state, loss = make_step(model, opt_state, optimizer, loss_fn)
        print(f"step={step} | loss={loss:.6e}")
        loss_list.append(loss)
        x_norm_list.append(jnp.linalg.norm(model.x))

    print(model.get_osc_params())

    plt.plot(loss_list)
    plt.ylabel("Loss")
    plt.xlabel("Train Step")
    plt.savefig("loss.png")
    plt.close()

    plt.plot(x_norm_list)
    plt.ylabel("Hidden Osc Norm")
    plt.xlabel("Train Step")
    plt.savefig("hidden.png")
    plt.close()

if __name__ == "__main__":
    main()

