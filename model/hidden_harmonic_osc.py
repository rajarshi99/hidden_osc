import jax
import jax.numpy as jnp
import equinox as eqx

class HiddenHarmonicOsc(eqx.Module):
    # trainable
    x: jax.Array            # (T,)
    diag_params: jax.Array 
    edge_params: jax.Array 

    # static
    edges: jax.Array = eqx.field(static=True)

    def get_osc_params(self):
        diag = jax.nn.softplus(self.diag_params)
        edge = jax.nn.softplus(self.edge_params)
        return diag, edge

    def potential(self, q):
        k_diag, k_edge = self.get_osc_params()

        V = 0.5 *  jnp.sum(k_diag * q**2, axis=-1)

        qi = q[..., self.edges[:,0]]
        qj = q[..., self.edges[:,1]]
        V += 0.5* jnp.sum(k_edge * (qi - qj)**2, axis=-1)

        return V

    def disc_langrangian(self, q0, q1, h):
        qdot = (q1 - q0) / h[:, None]
        qmid = 0.5 * (q0 +q1)
        kinetic = 0.5*jnp.sum(qdot**2, axis=-1)
        return h * (kinetic - self.potential(qmid))

    def loss(self, t_data, q_data):
        q = jnp.concatenate([q_data, self.x], axis=1)

        def action(q):
            return jnp.sum(
                    self.disc_langrangian(
                        q[:-1],
                        q[1:],
                        t_data[1:] - t_data[:-1]
                        )
                    )

        R = jax.grad(action)(q)
        _, k_edge = self.get_osc_params()

        return jnp.mean(R[1:-1]**2) + 1e-3*jnp.mean(k_edge**2)
