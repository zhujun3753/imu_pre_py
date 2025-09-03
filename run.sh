# evo_traj euroc data/state_groundtruth_estimate0/data.csv --save_as_kitti
# evo_traj euroc data/state_groundtruth_estimate0/data.csv --save_plot ape_tum_kitti.png

# 进入容器后
python main.py --euroc_root /app/data --kf_hz 10 --scale_drop 0.3 --iters 8 --sigma_g 0.1

