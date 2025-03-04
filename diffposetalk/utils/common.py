from functools import reduce
from pathlib import Path

import torch
import torch.nn.functional as F


class NullableArgs:
    def __init__(self, namespace):
        for key, value in namespace.__dict__.items():
            setattr(self, key, value)

    def __getattr__(self, key):
        # when an attribute lookup has not found the attribute
        if key == 'align_mask_width':
            if 'use_alignment_mask' in self.__dict__:
                return 1 if self.use_alignment_mask else 0
            else:
                return 0
        if key == 'no_head_pose':
            return not self.predict_head_pose
        if key == 'no_use_learnable_pe':
            return not self.use_learnable_pe

        return None


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_option_text(args, parser):
    message = ''
    for k, v in sorted(vars(args).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = f'\t[default: {str(default)}]'
        message += f'{str(k):>30}: {str(v):<30}{comment}\n'
    return message


def get_model_path(exp_name, iteration, model_type='DPT'):
    exp_root_dir = Path(__file__).parent.parent / 'experiments' / model_type
    exp_dir = exp_root_dir / exp_name
    if not exp_dir.exists():
        exp_dir = next(exp_root_dir.glob(f'{exp_name}*'))
    model_path = exp_dir / f'checkpoints/iter_{iteration:07}.pt'
    return model_path, exp_dir.relative_to(exp_root_dir)


def get_pose_input(coef_dict, rot_repr, with_global_pose):
    if rot_repr == 'aa':
        pose_input = coef_dict['pose'] if with_global_pose else coef_dict['pose'][..., -3:]
        # Remove mouth rotation round y, z axis
        pose_input = pose_input[..., :-2]
    else:
        raise ValueError(f'Unknown rotation representation: {rot_repr}')
    return pose_input


def get_motion_coef(coef_dict, rot_repr, with_global_pose=False, norm_stats=None):
    if norm_stats is not None:
        if rot_repr == 'aa':
            keys = ['exp', 'pose']
        else:
            raise ValueError(f'Unknown rotation representation {rot_repr}!')

        coef_dict = {k: (coef_dict[k] - norm_stats[f'{k}_mean']) / norm_stats[f'{k}_std'] for k in keys}
    pose_coef = get_pose_input(coef_dict, rot_repr, with_global_pose)
    return torch.cat([coef_dict['exp'], pose_coef], dim=-1)


def get_coef_dict(motion_coef, shape_coef=None, denorm_stats=None, with_global_pose=False, rot_repr='aa'):
    coef_dict = {
        'exp': motion_coef[..., :50]
    }
    if rot_repr == 'aa':
        if with_global_pose:
            coef_dict['pose'] = motion_coef[..., 50:]
        else:
            placeholder = torch.zeros_like(motion_coef[..., :3])
            coef_dict['pose'] = torch.cat([placeholder, motion_coef[..., -1:]], dim=-1)
        # Add back rotation around y, z axis
        coef_dict['pose'] = torch.cat([coef_dict['pose'], torch.zeros_like(motion_coef[..., :2])], dim=-1)
    else:
        raise ValueError(f'Unknown rotation representation {rot_repr}!')

    if shape_coef is not None:
        if motion_coef.ndim == 3:
            if shape_coef.ndim == 2:
                shape_coef = shape_coef.unsqueeze(1)
            if shape_coef.shape[1] == 1:
                shape_coef = shape_coef.expand(-1, motion_coef.shape[1], -1)

        coef_dict['shape'] = shape_coef

    if denorm_stats is not None:
        coef_dict = {k: coef_dict[k] * denorm_stats[f'{k}_std'] + denorm_stats[f'{k}_mean'] for k in coef_dict}

    if not with_global_pose:
        if rot_repr == 'aa':
            coef_dict['pose'][..., :3] = 0
        else:
            raise ValueError(f'Unknown rotation representation {rot_repr}!')

    return coef_dict


def coef_dict_to_vertices(coef_dict, flame, rot_repr='aa', ignore_global_rot=False, flame_batch_size=512):
    shape = coef_dict['exp'].shape[:-1]
    coef_dict = {k: v.view(-1, v.shape[-1]) for k, v in coef_dict.items()}
    n_samples = reduce(lambda x, y: x * y, shape, 1)

    # Convert to vertices
    vert_list = []
    for i in range(0, n_samples, flame_batch_size):
        batch_coef_dict = {k: v[i:i + flame_batch_size] for k, v in coef_dict.items()}
        if rot_repr == 'aa':
            vert, _, _ = flame(
                batch_coef_dict['shape'], batch_coef_dict['exp'], batch_coef_dict['pose'],
                pose2rot=True, ignore_global_rot=ignore_global_rot, return_lm2d=False, return_lm3d=False)
        else:
            raise ValueError(f'Unknown rot_repr: {rot_repr}')
        vert_list.append(vert)

    vert_list = torch.cat(vert_list, dim=0)  # (n_samples, 5023, 3)
    vert_list = vert_list.view(*shape, -1, 3)  # (..., 5023, 3)

    return vert_list


def compute_loss(args, is_starting_sample, shape_coef, motion_coef_gt, noise, target, prev_motion_coef, coef_stats,
                 flame, end_idx=None):
    if args.criterion.lower() == 'l2':
        criterion_func = F.mse_loss
    elif args.criterion.lower() == 'l1':
        criterion_func = F.l1_loss
    else:
        raise NotImplementedError(f'Criterion {args.criterion} not implemented.')

    loss_vert = None
    loss_vel = None
    loss_smooth = None
    loss_head_angle = None
    loss_head_vel = None
    loss_head_smooth = None
    loss_head_trans_vel = None
    loss_head_trans_accel = None
    loss_head_trans = None
    if args.target == 'noise':
        loss_noise = criterion_func(noise, target[:, args.n_prev_motions:], reduction='none')
    elif args.target == 'sample':
        if is_starting_sample:
            target = target[:, args.n_prev_motions:]
        else:
            motion_coef_gt = torch.cat([prev_motion_coef, motion_coef_gt], dim=1)
            if args.no_constrain_prev:
                target = torch.cat([prev_motion_coef, target[:, args.n_prev_motions:]], dim=1)

        loss_noise = criterion_func(motion_coef_gt, target, reduction='none')

        if args.l_vert > 0 or args.l_vel > 0:
            coef_gt = get_coef_dict(motion_coef_gt, shape_coef, coef_stats, with_global_pose=False,
                                    rot_repr=args.rot_repr)
            coef_pred = get_coef_dict(target, shape_coef, coef_stats, with_global_pose=False,
                                      rot_repr=args.rot_repr)
            seq_len = target.shape[1]

            if args.rot_repr == 'aa':
                verts_gt, _, _ = flame(coef_gt['shape'].view(-1, 100), coef_gt['exp'].view(-1, 50),
                                       coef_gt['pose'].view(-1, 6), return_lm2d=False, return_lm3d=False)
                verts_pred, _, _ = flame(coef_pred['shape'].view(-1, 100), coef_pred['exp'].view(-1, 50),
                                         coef_pred['pose'].view(-1, 6), return_lm2d=False, return_lm3d=False)
            else:
                raise ValueError(f'Unknown rotation representation {args.rot_repr}!')
            verts_gt = verts_gt.view(-1, seq_len, 5023, 3)
            verts_pred = verts_pred.view(-1, seq_len, 5023, 3)

            if args.l_vert > 0:
                loss_vert = criterion_func(verts_gt, verts_pred, reduction='none')

            if args.l_vel > 0:
                vel_gt = verts_gt[:, 1:] - verts_gt[:, :-1]
                vel_pred = verts_pred[:, 1:] - verts_pred[:, :-1]
                loss_vel = criterion_func(vel_gt, vel_pred, reduction='none')

            if args.l_smooth > 0:
                vel_pred = verts_pred[:, 1:] - verts_pred[:, :-1]
                loss_smooth = criterion_func(vel_pred[:, 1:], vel_pred[:, :-1], reduction='none')

        # head pose
        if not args.no_head_pose:
            if args.rot_repr == 'aa':
                head_pose_gt = motion_coef_gt[:, :, 50:53]
                head_pose_pred = target[:, :, 50:53]
            else:
                raise ValueError(f'Unknown rotation representation {args.rot_repr}!')

            if args.l_head_angle > 0:
                loss_head_angle = criterion_func(head_pose_gt, head_pose_pred, reduction='none')

            if args.l_head_vel > 0:
                head_vel_gt = head_pose_gt[:, 1:] - head_pose_gt[:, :-1]
                head_vel_pred = head_pose_pred[:, 1:] - head_pose_pred[:, :-1]
                loss_head_vel = criterion_func(head_vel_gt, head_vel_pred, reduction='none')

            if args.l_head_smooth > 0:
                head_vel_pred = head_pose_pred[:, 1:] - head_pose_pred[:, :-1]
                loss_head_smooth = criterion_func(head_vel_pred[:, 1:], head_vel_pred[:, :-1], reduction='none')

            if not is_starting_sample and args.l_head_trans > 0:
                # # version 1: constrain both the predicted previous and current motions (x_{-3} ~ x_{2})
                # head_pose_trans = head_pose_pred[:, args.n_prev_motions - 3:args.n_prev_motions + 3]
                # head_vel_pred = head_pose_trans[:, 1:] - head_pose_trans[:, :-1]
                # head_accel_pred = head_vel_pred[:, 1:] - head_vel_pred[:, :-1]

                # version 2: constrain only the predicted current motions (x_{0} ~ x_{2})
                head_pose_trans = torch.cat([head_pose_gt[:, args.n_prev_motions - 3:args.n_prev_motions],
                                             head_pose_pred[:, args.n_prev_motions:args.n_prev_motions + 3]], dim=1)
                head_vel_pred = head_pose_trans[:, 1:] - head_pose_trans[:, :-1]
                head_accel_pred = head_vel_pred[:, 1:] - head_vel_pred[:, :-1]

                # will constrain x_{-2|0} ~ x_{1}
                loss_head_trans_vel = criterion_func(head_vel_pred[:, 2:4], head_vel_pred[:, 1:3], reduction='none')
                # will constrain x_{-3|0} ~ x_{2}
                loss_head_trans_accel = criterion_func(head_accel_pred[:, 1:], head_accel_pred[:, :-1],
                                                       reduction='none')
    else:
        raise ValueError(f'Unknown diffusion target: {args.target}')

    if end_idx is None:
        mask = torch.ones((target.shape[0], args.n_motions), dtype=torch.bool, device=target.device)
    else:
        mask = torch.arange(args.n_motions, device=target.device).expand(target.shape[0], -1) < end_idx.unsqueeze(1)

    if args.target == 'sample' and not is_starting_sample:
        if args.no_constrain_prev:
            # Warning: this option will be deprecated in the future
            mask = torch.cat([torch.zeros_like(mask[:, :args.n_prev_motions]), mask], dim=1)
        else:
            mask = torch.cat([torch.ones_like(mask[:, :args.n_prev_motions]), mask], dim=1)

    loss_noise = loss_noise[mask].mean()
    if loss_vert is not None:
        loss_vert = loss_vert[mask].mean()
    if loss_vel is not None:
        loss_vel = loss_vel[mask[:, 1:]]
        loss_vel = loss_vel.mean() if torch.numel(loss_vel) > 0 else None
    if loss_smooth is not None:
        loss_smooth = loss_smooth[mask[:, 2:]]
        loss_smooth = loss_smooth.mean() if torch.numel(loss_smooth) > 0 else None
    if loss_head_angle is not None:
        loss_head_angle = loss_head_angle[mask].mean()
    if loss_head_vel is not None:
        loss_head_vel = loss_head_vel[mask[:, 1:]]
        loss_head_vel = loss_head_vel.mean() if torch.numel(loss_head_vel) > 0 else None
    if loss_head_smooth is not None:
        loss_head_smooth = loss_head_smooth[mask[:, 2:]]
        loss_head_smooth = loss_head_smooth.mean() if torch.numel(loss_head_smooth) > 0 else None
    if loss_head_trans_vel is not None:
        vel_mask = mask[:, args.n_prev_motions:args.n_prev_motions + 2]
        accel_mask = mask[:, args.n_prev_motions:args.n_prev_motions + 3]
        loss_head_trans_vel = loss_head_trans_vel[vel_mask].mean()
        loss_head_trans_accel = loss_head_trans_accel[accel_mask].mean()
        loss_head_trans = loss_head_trans_vel + loss_head_trans_accel

    return loss_noise, loss_vert, loss_vel, loss_smooth, loss_head_angle, loss_head_vel, loss_head_smooth, \
           loss_head_trans


def _truncate_audio(audio, end_idx, pad_mode='zero'):
    batch_size = audio.shape[0]
    audio_trunc = audio.clone()
    if pad_mode == 'replicate':
        for i in range(batch_size):
            audio_trunc[i, end_idx[i]:] = audio_trunc[i, end_idx[i] - 1]
    elif pad_mode == 'zero':
        for i in range(batch_size):
            audio_trunc[i, end_idx[i]:] = 0
    else:
        raise ValueError(f'Unknown pad mode {pad_mode}!')

    return audio_trunc


def _truncate_coef_dict(coef_dict, end_idx, pad_mode='zero'):
    batch_size = coef_dict['exp'].shape[0]
    coef_dict_trunc = {k: v.clone() for k, v in coef_dict.items()}
    if pad_mode == 'replicate':
        for i in range(batch_size):
            for k in coef_dict_trunc:
                coef_dict_trunc[k][i, end_idx[i]:] = coef_dict_trunc[k][i, end_idx[i] - 1]
    elif pad_mode == 'zero':
        for i in range(batch_size):
            for k in coef_dict:
                coef_dict_trunc[k][i, end_idx[i]:] = 0
    else:
        raise ValueError(f'Unknown pad mode: {pad_mode}!')

    return coef_dict_trunc


def truncate_coef_dict_and_audio(audio, coef_dict, n_motions, audio_unit=640, pad_mode='zero'):
    batch_size = audio.shape[0]
    end_idx = torch.randint(1, n_motions, (batch_size,), device=audio.device)
    audio_end_idx = (end_idx * audio_unit).long()
    # mask = torch.arange(n_motions, device=audio.device).expand(batch_size, -1) < end_idx.unsqueeze(1)

    # truncate audio
    audio_trunc = _truncate_audio(audio, audio_end_idx, pad_mode=pad_mode)

    # truncate coef dict
    coef_dict_trunc = _truncate_coef_dict(coef_dict, end_idx, pad_mode=pad_mode)

    return audio_trunc, coef_dict_trunc, end_idx


def truncate_motion_coef_and_audio(audio, motion_coef, n_motions, audio_unit=640, pad_mode='zero'):
    batch_size = audio.shape[0]
    end_idx = torch.randint(1, n_motions, (batch_size,), device=audio.device)
    audio_end_idx = (end_idx * audio_unit).long()
    # mask = torch.arange(n_motions, device=audio.device).expand(batch_size, -1) < end_idx.unsqueeze(1)

    # truncate audio
    audio_trunc = _truncate_audio(audio, audio_end_idx, pad_mode=pad_mode)

    # prepare coef dict and stats
    coef_dict = {'exp': motion_coef[..., :50], 'pose_any': motion_coef[..., 50:]}

    # truncate coef dict
    coef_dict_trunc = _truncate_coef_dict(coef_dict, end_idx, pad_mode=pad_mode)
    motion_coef_trunc = torch.cat([coef_dict_trunc['exp'], coef_dict_trunc['pose_any']], dim=-1)

    return audio_trunc, motion_coef_trunc, end_idx


def nt_xent_loss(feature_a, feature_b, temperature):
    """
    Normalized temperature-scaled cross entropy loss.

    (Adapted from https://github.com/sthalles/SimCLR/blob/master/simclr.py)

    Args:
        feature_a (torch.Tensor): shape (batch_size, feature_dim)
        feature_b (torch.Tensor): shape (batch_size, feature_dim)
        temperature (float): temperature scaling factor

    Returns:
        torch.Tensor: scalar
    """
    batch_size = feature_a.shape[0]
    device = feature_a.device

    features = torch.cat([feature_a, feature_b], dim=0)

    labels = torch.cat([torch.arange(batch_size), torch.arange(batch_size)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1))
    labels = labels.to(device)

    features = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T)

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(labels.shape[0], -1)

    # select the positives and negatives
    positives = similarity_matrix[labels].view(labels.shape[0], -1)
    negatives = similarity_matrix[~labels].view(labels.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    logits = logits / temperature
    labels = torch.zeros(labels.shape[0], dtype=torch.long).to(device)

    loss = F.cross_entropy(logits, labels)
    return loss
