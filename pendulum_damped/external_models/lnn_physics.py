# Generalized Lagrangian Networks | 2020
# Miles Cranmer, Sam Greydanus, Stephan Hoyer (...)

import jax
import jax.numpy as jnp
from jax import jit

m = 1. # old was 0.1
l = 10. # old was 1.
g = 9.81
b = 0.5


@jit
def kinetic_energy(q, q_dot):
  T1 = 0.5 * m * (l * q_dot)**2
  T = T1
  return T

@jit
def potential_energy(q, q_dot):
  V = -m*g*jnp.cos(q)
  return V

# Double pendulum lagrangian
@jit
def lagrangian_fn(q, q_dot):
  T = kinetic_energy(q, q_dot)
  V = potential_energy(q, q_dot)
  return T - V

# Double pendulum lagrangian
@jit
def hamiltonian_fn(q, q_dot, m1=1, m2=1, l1=1, l2=1, g=9.8):
  (t1, t2), (w1, w2) = q, q_dot

  T = kinetic_energy(q, q_dot, m1=1, m2=1, l1=1, l2=1, g=9.8)
  V = potential_energy(q, q_dot, m1=1, m2=1, l1=1, l2=1, g=9.8)
  return T + V
  

# Double pendulum dynamics via analytical forces taken from Diego's blog
@jit
def analytical_fn(state):
  q, q_t = state
  q_tt = -(g/l)*jnp.sin(q)-(b/m)*q_t
  return jnp.stack([q_t, q_tt])