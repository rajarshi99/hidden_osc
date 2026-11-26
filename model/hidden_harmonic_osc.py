import jax
import jax.numpy as jnp
import equinox as eqx

class HiddenHarmonicOsc(eqx.Module):
    # trainable
    x: jax.Array   # (T, num_hidden_nodes)

    diag_params: jax.Array  # (num_nodes,)
    edge_params: jax.Array  # (num_edges,)

    # static
    edges: jax.Array = eqx.field(static=True)

    def get_osc_params(self):
        diag = jax.nn.softplus(self.diag_params)
        # edge = jax.nn.softplus(self.edge_params)
        return diag, self.edge_params

    def potential(self, q):
        k_diag, k_edge = self.get_osc_params()

        V = 0.5 *  jnp.sum(k_diag * q**2, axis=-1)

        qi = q[..., self.edges[:,0]]
        qj = q[..., self.edges[:,1]]
        V += jnp.sum(k_edge*qi*qj, axis=-1)

        return V

    def disc_langrangian(self, q0, q1, h):
        qdot = (q1 - q0) / h[:, None]
        qmid = 0.5 * (q0 +q1)
        kinetic = 0.5*jnp.sum(qdot**2, axis=-1)
        return h * (kinetic - self.potential(qmid))

    def forward(self, q0, q1, h, num_steps):
        force_fn = jax.grad(self.potential)

        def scan_fn(carry, _):
            q_prev, q_curr = carry
            q_next = 2.0*q_curr - q_prev - h**2 * force_fn(q_curr)
            return (q_curr, q_next), q_next

        _, q_future = jax.lax.scan(scan_fn, (q0, q1), length=num_steps)
        return q_future

    def build_state(self, q, x):
        # k_diag, _ = self.get_osc_params()
        # omega_min = jnp.sqrt(jnp.min(k_diag))
        # omega_max = jnp.sqrt(jnp.max(k_diag))
        # omega = jax.nn.sigmoid(self.omega) * (omega_max - omega_min) + omega_min
        #
        # phase_t = t[:, None, None] * omega[None, None, :] + self.phase[None, :]
        # x = jnp.sum(self.amp[None, :, :]*jnp.cos(phase_t), axis=-1)
        # q_full = jnp.concatenate([q, x], axis=1)
        q_full = jnp.concatenate([q, x], axis=-1)
        return q_full

    def loss(self, t_data, q_data):
        q = self.build_state(q_data, self.x)
        def action(q):
            return jnp.sum(
                    self.disc_langrangian(
                        q[:-1],
                        q[1:],
                        t_data[1:] - t_data[:-1]
                        )
                    )
        R = jax.grad(action)(q)
        del_loss = jnp.mean(R[1:-1]**2)

        num_fsteps = min(q_data.shape[0] // 2, 50)
        q0 = self.build_state(q_data[-num_fsteps-2], self.x[-num_fsteps-2])
        q1 = self.build_state(q_data[-num_fsteps-1], self.x[-num_fsteps-1])
        h = t_data[-num_fsteps-1] - t_data[-num_fsteps-2]
        q_end_pred = self.forward(q0, q1, h, num_fsteps)
        forward_loss = jnp.mean((q_end_pred[:,:-self.x.shape[1]] - q_data[-num_fsteps:])**2)

        return del_loss, forward_loss
