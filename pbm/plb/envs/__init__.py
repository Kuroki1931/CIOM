import gym
from .env import PlasticineEnv
from gym import register

ENVS = []
for env_name in ['push_box', 'two_roomba_deform_one_rope', 'two_roomba_wrap_rope', 'three_roomba_deform_one_rope', "spell_out", "two_roomba_rotate_one_box", "four_roomba_transfer_one_box", "multi_bc", 'multi_bc_rope', 'multi_bc_rope_6', 'multi_bc_rope_5_side', 'multi_bc_rope_5_center', 'multi_bc_rope_4_side', 'multi_bc_rope_4_center', 'multi_bc_rope_4_side_center', 'multi_bc_rope_3']:
    for id in range(2000):
        register(
            id = f'{env_name}-v{id+1}',
            entry_point=f"plb.envs.env:PlasticineEnv",
            kwargs={'cfg_path': f"{env_name.lower()}.yml", "version": id+1},
            max_episode_steps=50
        )


def make(env_name, nn=False, sdf_loss=10, density_loss=10, contact_loss=1, soft_contact_loss=False):
    env: PlasticineEnv = gym.make(env_name, nn=nn)
    env.taichi_env.loss.set_weights(sdf=sdf_loss, density=density_loss,
                                    contact=contact_loss, is_soft_contact=soft_contact_loss)
    return env