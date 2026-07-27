# AGHRI Online FPS Benchmark Final Report

Generated: 2026-07-26T18:04:50.998723+00:00

The benchmark runner is `tools/run_aghri_online_fps_matrix.py`.
It launches `launch/aghri_online_late_fusion.launch.py` with `inference_mode:=live`, `play_bag:=false`, `enable_visualisation:=false`, and `launch_rviz:=false`, waits for `worker_ready` and `LIVE LiDAR inference enabled`, then starts `ros2 bag play` separately.

Detector CSV caches and cached detector publishers are not used by this benchmark.

Rows collected: 48
Dry run: False

See `per_run_results.csv`, `summary_results.csv`, `summary_results.md`, `benchmark_config.json`, `environment.json`, and `raw_runs/`.
