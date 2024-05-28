# Paper scripts
python test_yaw_constraints.py -cf ./test_configs/reachavoid/test_config_cbf_reachavoid_circle_config_easy.yaml -pt reachavoidcf0_easy -rb 1.2
python test_yaw_constraints.py -cf ./test_configs/reachavoid/test_config_cbf_reachavoid_circle_config1.yaml --naive_task -pt reachavoidcf1_naive_task -rb 1.2
python test_yaw_constraints.py -cf ./test_configs/reachavoid/test_config_cbf_reachavoid_circle_config1.yaml --naive_task -pt reachavoidcf1_naive_task_squeeze -rb 1.0
python test_yaw_constraints.py -cf ./test_configs/reachavoid/test_config_cbf_reachavoid_circle_config1.yaml -pt reachavoidcf1_lqr_task -rb 1.2
python test_yaw_constraints.py -cf ./test_configs/reachavoid/test_config_cbf_reachavoid_circle_config1.yaml -pt reachavoidcf1_lqr_task_squeeze -rb 1.0

# Robustness checks and staggered obstacles
python test_yaw_constraints.py -cf ./test_configs/reachavoid/test_config_cbf_reachavoid_circle_config2.yaml -pt reachavoidcf2_lqr_task -rb 1.2
python test_yaw_constraints.py -cf ./test_configs/reachavoid/test_config_cbf_reachavoid_circle_config2.yaml --naive_task -pt reachavoidcf2_naive_task -rb 1.2
python test_yaw_constraints.py -cf ./test_configs/reachavoid/test_config_cbf_reachavoid_circle_config3.yaml -pt reachavoidcf3_lqr_task -rb 1.2

# Different wheelbase
python test_yaw_constraints.py -cf ./test_configs/reachavoid/test_config_cbf_reachavoid_circle_config4.yaml -pt reachavoidcf4_lqr_task -rb 1.2

# Check whether everything stops
python test_yaw_constraints.py -cf ./test_configs/reachavoid/test_config_cbf_reachavoid_circle_config2.yaml --naive_task -pt reachavoidcf2_naive_task_squeeze -rb 1.0
python test_yaw_constraints.py -cf ./test_configs/reachavoid/test_config_cbf_reachavoid_circle_config2.yaml -pt reachavoidcf2_lqr_task_squeeze -rb 1.0