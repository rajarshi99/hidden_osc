import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from model.hidden_harmonic_osc import HiddenHarmonicOsc

@eqx.filter_jit
def make_step(model, opt_state, optimizer, loss_fn):
    (loss, aux), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss, aux

def train(model, train_data, epochs=1, lr=1e-3):
    inp = train_data["q"][:-1]
    out = train_data["q"][1:]

    def loss_fn(model):
        pred = jax.vmap(model)(inp)
        return jnp.mean((pred - out) ** 2), None # Not tested after aux in make_step
    
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    for epoch in range(epochs):
        model, opt_state, loss, _ = make_step(model, opt_state, optimizer, loss_fn)
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
    model = HiddenHarmonicOsc(
            x=jax.random.normal(key, (num_tsteps,1)),         # Only one hidden osc

            diag_params=jnp.ones(num_dims+1), # Only one hidden osc
            edge_params=jnp.ones(len(edges)),
            edges=edges
            )
    num_test_steps = min(test_data["q"].shape[0], 100)

    def loss_fn(model):
        del_loss, forward_loss = model.loss(train_data["t"], train_data["q"])
        return del_loss + 1*forward_loss, (del_loss, forward_loss)

    schedule = optax.exponential_decay(
                init_value=1e-2,
                transition_steps=50000,
                decay_rate=0.9
                )
    optimizer = optax.adam(schedule)
    opt_state = optimizer.init(
            eqx.filter(model, eqx.is_array)
            )

    iters_per_log = 1000
    num_iters = 1000*iters_per_log
    x_norm_list = []
    loss_list = []
    loss_aux_list = []
    val_list = []
    for step in range(num_iters):
        model, opt_state, loss, aux = make_step(model, opt_state, optimizer, loss_fn)
        if step % iters_per_log == 0:
            x_norm_list.append(jnp.linalg.norm(model.x))

            loss_list.append(loss)
            loss_aux_list.append(aux)

            q_pred = model.forward(
                    q0=model.build_state(train_data["q"][-2], model.x[-2]),
                    q1=model.build_state(train_data["q"][-1], model.x[-1]),
                    h=train_data["t"][-1] - train_data["t"][-2],
                    num_steps=num_test_steps
                    )
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

    for i in range(len(loss_aux_list[0])):
        plt.plot([tup[i] for tup in loss_aux_list], label=f"Term {i}")
    plt.plot(loss_list, label="Total")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.xlabel("Train Step")
    plt.legend()
    plt.savefig("loss.png")
    plt.close()

    plt.plot(val_list)
    plt.ylabel("RMSE on Prediction")
    plt.yscale("log")
    plt.xlabel("Train Step")
    plt.savefig("val.png")
    plt.close()

    q_pred = model.forward(
            q0=model.build_state(train_data["q"][-2], model.x[-2]),
            q1=model.build_state(train_data["q"][-1], model.x[-1]),
            h=train_data["t"][-1] - train_data["t"][-2],
            num_steps=num_test_steps
            )
    err_wrt_time = jnp.linalg.norm(
            q_pred[:num_test_steps,:-1] - test_data["q"][:num_test_steps],
            axis=1
            )
    plt.plot(test_data["t"][:num_test_steps], err_wrt_time)
    plt.ylabel("Error Norm")
    plt.yscale("log")
    plt.xlabel("Test Time")
    plt.grid()
    plt.savefig("test.png")
    plt.close()

if __name__ == "__main__":
    main()

