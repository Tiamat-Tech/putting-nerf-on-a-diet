import jax

from jaxnerf.nerf import clip_utils
from jaxnerf.nerf import utils


def render_fn(model, variables, key_0, key_1, rays):
    return jax.lax.all_gather(model.apply(variables, key_0, key_1, rays, False), axis_name="batch")


render_pfn = jax.pmap(render_fn, in_axes=(None, None, None, None, 0),
                      donate_argnums=3, axis_name="batch")


def render_image_from_pose(theta, phi, radius, ):
    camtoworld = clip_utils.pose_spherical(theta, phi, radius)
    # TODO @Alex: use render_image to render image from rays
    return None

