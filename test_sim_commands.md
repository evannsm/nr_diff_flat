# First non-pyjoules except for helix spin, then pyjoules except for helix spin.
# Separately after we'll do helix spin with and without pyjoules.

## Part 1: All Non-PyJoules Except Helix Spin (JAX)

```bash
ros2 run nr_diff_flat run_node --platform sim --trajectory hover      --double-speed --hover-mode 1 --ctrl-type jax --log
ros2 run nr_diff_flat run_node --platform sim --trajectory yaw_only   --double-speed --ctrl-type jax --log
ros2 run nr_diff_flat run_node --platform sim --trajectory circle_horz --double-speed --ctrl-type jax --log
ros2 run nr_diff_flat run_node --platform sim --trajectory circle_horz --double-speed --spin --ctrl-type jax --log
ros2 run nr_diff_flat run_node --platform sim --trajectory circle_vert --double-speed --ctrl-type jax --log
ros2 run nr_diff_flat run_node --platform sim --trajectory fig8_horz   --double-speed --ctrl-type jax --log
ros2 run nr_diff_flat run_node --platform sim --trajectory fig8_vert   --double-speed --ctrl-type jax --log
ros2 run nr_diff_flat run_node --platform sim --trajectory fig8_vert   --double-speed --short --ctrl-type jax --log
ros2 run nr_diff_flat run_node --platform sim --trajectory helix       --double-speed --ctrl-type jax --log
ros2 run nr_diff_flat run_node --platform sim --trajectory sawtooth    --double-speed --ctrl-type jax --log
ros2 run nr_diff_flat run_node --platform sim --trajectory triangle    --double-speed --ctrl-type jax --log
```

## Part 2: All PyJoules Except Helix Spin (JAX)

```bash
ros2 run nr_diff_flat run_node --platform sim --trajectory hover      --double-speed --hover-mode 1 --ctrl-type jax --pyjoules --log
ros2 run nr_diff_flat run_node --platform sim --trajectory yaw_only   --double-speed --ctrl-type jax --pyjoules --log
ros2 run nr_diff_flat run_node --platform sim --trajectory circle_horz --double-speed --ctrl-type jax --pyjoules --log
ros2 run nr_diff_flat run_node --platform sim --trajectory circle_horz --double-speed --spin --ctrl-type jax --pyjoules --log
ros2 run nr_diff_flat run_node --platform sim --trajectory circle_vert --double-speed --ctrl-type jax --pyjoules --log
ros2 run nr_diff_flat run_node --platform sim --trajectory fig8_horz   --double-speed --ctrl-type jax --pyjoules --log
ros2 run nr_diff_flat run_node --platform sim --trajectory fig8_vert   --double-speed --ctrl-type jax --pyjoules --log
ros2 run nr_diff_flat run_node --platform sim --trajectory fig8_vert   --double-speed --short --ctrl-type jax --pyjoules --log
ros2 run nr_diff_flat run_node --platform sim --trajectory helix       --double-speed --ctrl-type jax --pyjoules --log
ros2 run nr_diff_flat run_node --platform sim --trajectory sawtooth    --double-speed --ctrl-type jax --pyjoules --log
ros2 run nr_diff_flat run_node --platform sim --trajectory triangle    --double-speed --ctrl-type jax --pyjoules --log
```

## Part 3: Helix Spin with and without PyJoules (JAX)

```bash
ros2 run nr_diff_flat run_node --platform sim --trajectory helix --double-speed --spin --ctrl-type jax --log
ros2 run nr_diff_flat run_node --platform sim --trajectory helix --double-speed --spin --ctrl-type jax --pyjoules --log
```

## Part 4: NumPy Controller Variants (Non-PyJoules)

```bash
ros2 run nr_diff_flat run_node --platform sim --trajectory hover      --double-speed --hover-mode 1 --ctrl-type numpy --log
ros2 run nr_diff_flat run_node --platform sim --trajectory yaw_only   --double-speed --ctrl-type numpy --log
ros2 run nr_diff_flat run_node --platform sim --trajectory circle_horz --double-speed --ctrl-type numpy --log
ros2 run nr_diff_flat run_node --platform sim --trajectory circle_horz --double-speed --spin --ctrl-type numpy --log
ros2 run nr_diff_flat run_node --platform sim --trajectory circle_vert --double-speed --ctrl-type numpy --log
ros2 run nr_diff_flat run_node --platform sim --trajectory fig8_horz   --double-speed --ctrl-type numpy --log
ros2 run nr_diff_flat run_node --platform sim --trajectory fig8_vert   --double-speed --ctrl-type numpy --log
ros2 run nr_diff_flat run_node --platform sim --trajectory fig8_vert   --double-speed --short --ctrl-type numpy --log
ros2 run nr_diff_flat run_node --platform sim --trajectory helix       --double-speed --ctrl-type numpy --log
ros2 run nr_diff_flat run_node --platform sim --trajectory sawtooth    --double-speed --ctrl-type numpy --log
ros2 run nr_diff_flat run_node --platform sim --trajectory triangle    --double-speed --ctrl-type numpy --log
```

## Part 5: NumPy Controller Variants (PyJoules)

```bash
ros2 run nr_diff_flat run_node --platform sim --trajectory hover      --double-speed --hover-mode 1 --ctrl-type numpy --pyjoules --log
ros2 run nr_diff_flat run_node --platform sim --trajectory yaw_only   --double-speed --ctrl-type numpy --pyjoules --log
ros2 run nr_diff_flat run_node --platform sim --trajectory circle_horz --double-speed --ctrl-type numpy --pyjoules --log
ros2 run nr_diff_flat run_node --platform sim --trajectory circle_horz --double-speed --spin --ctrl-type numpy --pyjoules --log
ros2 run nr_diff_flat run_node --platform sim --trajectory circle_vert --double-speed --ctrl-type numpy --pyjoules --log
ros2 run nr_diff_flat run_node --platform sim --trajectory fig8_horz   --double-speed --ctrl-type numpy --pyjoules --log
ros2 run nr_diff_flat run_node --platform sim --trajectory fig8_vert   --double-speed --ctrl-type numpy --pyjoules --log
ros2 run nr_diff_flat run_node --platform sim --trajectory fig8_vert   --double-speed --short --ctrl-type numpy --pyjoules --log
ros2 run nr_diff_flat run_node --platform sim --trajectory helix       --double-speed --ctrl-type numpy --pyjoules --log
ros2 run nr_diff_flat run_node --platform sim --trajectory sawtooth    --double-speed --ctrl-type numpy --pyjoules --log
ros2 run nr_diff_flat run_node --platform sim --trajectory triangle    --double-speed --ctrl-type numpy --pyjoules --log
```
