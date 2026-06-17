from .yacs import CfgNode as CN
import argparse
import os
import numpy as np

from lib.utils.cfg_utils import make_cfg

cfg = CN()

cfg.workspace = os.environ.get('PWD', os.getcwd())
cfg.loaded_iter = -1
cfg.ip = '127.0.0.1'
cfg.port = 6009
cfg.data_device = 'cuda'
cfg.mode = 'train' 
cfg.task = 'hello' # task folder name
cfg.exp_name = 'test' # experiment folder name
cfg.gpus = [0] # list of gpus to use 
cfg.debug = False
cfg.resume = True # If set to True, resume training from the last checkpoint.
cfg.to_cuda = False # higher GPU utilization with larger memory required

cfg.source_path = ''
cfg.model_path = ''
cfg.record_dir = None
cfg.resolution = -1
cfg.resolution_scales = [1]

cfg.eval = CN()
cfg.eval.skip_train = False 
cfg.eval.skip_test = False 
cfg.eval.eval_train = False
cfg.eval.eval_test = True
cfg.eval.quiet = False

cfg.train = CN()
cfg.train.debug_from = -1
cfg.train.detect_anomaly = False
cfg.train.test_iterations = [1000, 3000, 5000, 10000, 15000, 20000, 25000, 30000]
cfg.train.save_iterations = [1000, 3000, 5000, 10000, 15000, 20000, 25000, 30000]
cfg.train.iterations = 30000
cfg.train.quiet = False
cfg.train.checkpoint_iterations = [1000, 3000, 5000, 10000, 15000, 20000, 25000, 30000]
cfg.train.start_checkpoint = None
cfg.train.importance_sampling = False
cfg.train.disable_structure_updates = False
cfg.train.scalar_trace_path = ''
cfg.train.save_visuals = False
cfg.train.log_image_interval = 0
cfg.train.log_fixed_view = True
cfg.train.log_fixed_view_ids = []
cfg.train.eval_sampled_interval = 1000
cfg.train.eval_full_interval = 5000
cfg.train.eval_panel_interval = 1000
cfg.train.eval_sampled_train_view_count = 15
cfg.train.eval_sampled_test_view_count = 15
cfg.train.eval_full_train_enabled = True
cfg.train.eval_full_test_enabled = True
cfg.train.periodic_eval_enabled = True
cfg.train.periodic_eval_interval = 1000
cfg.train.periodic_eval_view_ids = []
cfg.train.periodic_eval_views_per_camera = 3
cfg.train.periodic_eval_max_views = 15
cfg.train.full_snapshot_iterations = [1000, 3000, 5000, 10000, 15000, 20000, 25000, 30000]
cfg.train.full_snapshot_splits = ['train', 'test']
cfg.train.psnr_drop_warn_threshold = 5.0
cfg.train.guided_feedback = CN()
cfg.train.guided_feedback.enabled = False
cfg.train.guided_feedback.signal_path = ''
cfg.train.guided_feedback.region_weight = 2.0
cfg.train.guided_feedback.geometry_weight = 1.0
cfg.train.guided_feedback.min_valid_pixels = 1
cfg.train.guided_feedback.supervision_mode = 'lidar_supervised'  # lidar_supervised | da3_unsupervised | hybrid_reference
cfg.train.guided_feedback.use_lidar_depth = True
cfg.train.guided_feedback.feedback_mode = 'region'
cfg.train.guided_feedback.bad_pixel_weight = 3.0
cfg.train.guided_feedback.good_pixel_weight = 0.25
cfg.train.guided_feedback.pixel_radius = 2
cfg.train.guided_feedback.use_da3_structure = False
cfg.train.guided_feedback.assert_no_lidar_supervision = False
cfg.train.guided_feedback.da3_edge_weight = 1.0
cfg.train.guided_feedback.da3_ranking_weight = 1.0
cfg.train.guided_feedback.da3_side_weight = 0.5
cfg.train.guided_feedback.da3_edge_margin = 0.05
cfg.train.guided_feedback.da3_ranking_margin = 0.02
cfg.train.feedback_controller = CN()
cfg.train.feedback_controller.enabled = False
cfg.train.feedback_controller.start_iter = 0
cfg.train.feedback_controller.interval = 1000
cfg.train.feedback_controller.max_triggers = -1
cfg.train.feedback_controller.mode = 'off'  # off | diagnose_only | feedback_update | repair_dryrun | repair_apply
cfg.train.feedback_controller.risk_source = 'da3_boundary'  # lidar_error | da3_boundary | da3_structure
cfg.train.feedback_controller.supervision_mode = 'da3_unsupervised'  # lidar_supervised | da3_unsupervised | hybrid_reference
cfg.train.feedback_controller.feedback_mode = 'none'  # none | region | pixel | softpatch | contribution_softpatch
cfg.train.feedback_controller.repair_mode = 'none'  # none | dryrun | opacity_regularization | prune
cfg.train.feedback_controller.output_dir = ''
cfg.train.feedback_controller.selected_views = []
cfg.train.feedback_controller.trigger_views = []
cfg.train.feedback_controller.max_regions = 30
cfg.train.feedback_controller.max_pixels_per_region = 64
cfg.train.feedback_controller.top_contributors = 5
cfg.train.feedback_controller.signal_path = ''
cfg.train.feedback_controller.contribution_summary_path = ''
cfg.train.feedback_controller.contribution_source = 'cached_summary'  # cached_summary | live_current_model
cfg.train.feedback_controller.dryrun_scorer_path = ''
cfg.train.feedback_controller.dryrun_extra_args = ''
cfg.train.feedback_controller.run_dryrun_scorer = False
cfg.train.feedback_controller.use_cached_if_exists = True
cfg.train.feedback_controller.recompute_risk = False
cfg.train.feedback_controller.recompute_contribution = False
cfg.train.feedback_controller.recompute_softpatch = False
cfg.train.feedback_controller.recompute_responsible_groups = False
cfg.train.feedback_controller.run_counterfactual = False
cfg.train.feedback_controller.run_candidate_tagging = False
cfg.train.feedback_controller.run_group_counterfactual_every_n_triggers = 0
cfg.train.feedback_controller.run_candidate_tagging_every_n_triggers = 0
cfg.train.feedback_controller.fail_policy = 'warn'  # skip | warn | stop
cfg.train.feedback_controller.allow_parameter_modification = False
cfg.train.feedback_controller.skip_existing = True
cfg.train.gaussian_control = CN()
cfg.train.gaussian_control.enabled = False
cfg.train.gaussian_control.control_mode = 'off'  # off | protect_only | opacity_regularization | opacity_decay_apply | repair_dryrun | repair_apply
cfg.train.gaussian_control.evidence_source = 'auto'  # auto | lidar | da3
cfg.train.gaussian_control.risk_source = 'da3_boundary'  # lidar_error | da3_boundary | da3_structure
cfg.train.gaussian_control.supervision_mode = 'da3_unsupervised'  # lidar_supervised | da3_unsupervised | hybrid_reference
cfg.train.gaussian_control.counterfactual_objective = 'da3_structure'  # lidar_depth_error | da3_structure | hybrid
cfg.train.gaussian_control.group_source = 'feedback_controller'  # feedback_controller | cached_group_table
cfg.train.gaussian_control.cached_group_table = ''
cfg.train.gaussian_control.allow_parameter_modification = False
cfg.train.gaussian_control.allow_real_prune = False
cfg.train.gaussian_control.allow_real_split = False
cfg.train.gaussian_control.allow_real_shrink = False
cfg.train.gaussian_control.opacity_reg_weight = 0.0
cfg.train.gaussian_control.opacity_decay_factor = 0.95
cfg.train.gaussian_control.max_decay_gaussians_per_trigger = 10
cfg.train.gaussian_control.max_decay_ratio = 0.00005
cfg.train.gaussian_control.max_controlled_gaussians = 256
cfg.train.gaussian_control.min_group_confidence = 0.0
cfg.train.gaussian_control.min_multiview_support = 1
cfg.train.gaussian_control.min_support_pixels = 8
cfg.train.gaussian_control.rgb_safety_threshold = 0.02
cfg.train.gaussian_control.protect_good_groups = True
cfg.train.gaussian_control.skip_low_evidence = True

cfg.geovit = CN()
cfg.geovit.enabled = False
cfg.geovit.model_name = 'depth-anything/DA3-LARGE-1.1'
cfg.geovit.model_dir = ''
cfg.geovit.device = 'cuda'
cfg.geovit.use_cache = True
cfg.geovit.detach_cache = True
cfg.geovit.include_tokens = False
cfg.geovit.include_confidence = True
cfg.geovit.local_files_only = True
cfg.geovit.process_res = 128
cfg.geovit.process_res_method = 'upper_bound_resize'
cfg.geovit.confidence_threshold = 0.0

cfg.optim = CN()
# learning rate
cfg.optim.position_lr_init = 0.00016 # position_lr_init_{bkgd, obj ...}, similar to the following
cfg.optim.position_lr_final = 0.0000016
cfg.optim.position_lr_delay_mult = 0.01
cfg.optim.position_lr_max_steps = 30000
cfg.optim.feature_lr = 0.0025
cfg.optim.opacity_lr = 0.05
cfg.optim.scaling_lr = 0.005
cfg.optim.rotation_lr = 0.001
# densification and pruning
cfg.optim.percent_dense = 0.01 
cfg.optim.densification_interval = 100
cfg.optim.opacity_reset_interval = 3000
cfg.optim.densify_from_iter = 500
cfg.optim.densify_until_iter = 15000
cfg.optim.densify_grad_threshold = 0.0002 # densify_grad_threshold_{bkgd, obj ...}
cfg.optim.densify_grad_abs_bkgd = False # densification strategy from AbsGS
cfg.optim.densify_grad_abs_obj = False 
cfg.optim.max_screen_size = 20
cfg.optim.min_opacity = 0.005
cfg.optim.percent_big_ws = 0.1
# loss weight
cfg.optim.lambda_l1 = 1.
cfg.optim.lambda_dssim = 0.2
cfg.optim.lambda_sky = 0.
cfg.optim.lambda_sky_scale = []
cfg.optim.lambda_semantic = 0.
cfg.optim.lambda_reg = 0.
cfg.optim.lambda_depth_lidar = 0.
cfg.optim.lambda_depth_mono = 0.
cfg.optim.lambda_normal_mono = 0.
cfg.optim.lambda_color_correction = 0.
cfg.optim.lambda_pose_correction = 0.
cfg.optim.lambda_scale_flatten = 0.
cfg.optim.lambda_opacity_sparse = 0.


cfg.model = CN()
cfg.model.gaussian = CN()
cfg.model.gaussian.sh_degree = 3
cfg.model.gaussian.fourier_dim = 1 # fourier spherical harmonics dimension
cfg.model.gaussian.fourier_scale = 1.
cfg.model.gaussian.flip_prob = 0. # symmetry prior for rigid objects, flip gaussians with this probability during training
cfg.model.gaussian.semantic_mode = 'logits'

cfg.model.nsg = CN()
cfg.model.nsg.include_bkgd = True # include background
cfg.model.nsg.include_obj = True # include object
cfg.model.nsg.include_sky = False # include sky cubemap
cfg.model.nsg.opt_track = True # tracklets optimization
cfg.model.sky = CN()
cfg.model.sky.resolution = 1024
cfg.model.sky.white_background = True


#### Note: We have not fully tested this.
cfg.model.use_color_correction = False # If set to True, learn transformation matrixs for appearance embedding
cfg.model.color_correction = CN() 
cfg.model.color_correction.mode = 'image' # If set to 'image', learn separate embedding for each image. If set to 'sensor', learn a single embedding for all images captured by one camera senosor. 
cfg.model.color_correction.use_mlp = False # If set to True, regress embedding from extrinsic by a mlp. Otherwise, define the embedding explicitly.
cfg.model.color_correction.use_sky = False # If set to True, using spparate embedding for background and sky
# Alternative choice from GOF: https://github.com/autonomousvision/gaussian-opacity-fields/blob/main/scene/appearance_network.py

cfg.model.use_pose_correction = False # If set to True, use pose correction for camera poses. 
cfg.model.pose_correction = CN()
cfg.model.pose_correction.mode = 'image' # If set to 'image', learn separate correction matrix for each image. If set to 'frame', learn a single correction matrix for all images corresponding to the same frame timestamp. 
####

cfg.data = CN()
cfg.data.white_background = False # If set to True, use white background. Should be False when using sky cubemap.
cfg.data.use_colmap_pose = False # If set to True, use colmap to recalibrate camera poses as input (rigid bundle adjustment now).
cfg.data.filter_colmap = False # If set to True, filter out SfM points by camera poses.
cfg.data.box_scale = 1.0 # Scale the bounding box by this factor.
cfg.data.split_test = -1 
cfg.data.shuffle = True
cfg.data.eval = True
cfg.data.type = 'Colmap'
cfg.data.images = 'images'
cfg.data.use_semantic = False
cfg.data.use_mono_depth = False
cfg.data.use_mono_normal = False
cfg.data.use_colmap = True
cfg.data.colmap_executable = ""
cfg.data.initialization_note = ""
cfg.data.allow_lidar_initialization = True
cfg.data.require_no_lidar_initialization = False
# data.load_pcd_from: Load the initialization point cloud from a previous experiment without generation.
# data.extent: radius of the scene, we recommend 10 - 20 meters.
# data.sphere_scale: Scale the sphere radius by this factor.
# data.regenerate_pcd: Regenerate the initialization point cloud.

cfg.render = CN()
cfg.render.convert_SHs_python = False
cfg.render.compute_cov3D_python = False
cfg.render.debug = False
cfg.render.scaling_modifier = 1.0
cfg.render.fps = 24
cfg.render.render_normal = False
cfg.render.save_video = True
cfg.render.save_image = True
cfg.render.coord = 'world' # ['world', 'vehicle']
cfg.render.concat_cameras = []
cfg.render.camera_transform = CN()
cfg.render.camera_transform.enabled = False
cfg.render.camera_transform.shift = [0.0, 0.0, 0.0]
cfg.render.camera_transform.rotation_deg = [0.0, 0.0, 0.0]
cfg.render.camera_transform.mode = 'camera'
cfg.viewer = CN()
cfg.viewer.frame_id = 0 # Select the frame_id (start from 0) to save for viewer


parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/default.yaml", type=str)
parser.add_argument("--mode", type=str, default="")
parser.add_argument('--det', type=str, default='')
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

args = parser.parse_args()
cfg = make_cfg(cfg, args)

