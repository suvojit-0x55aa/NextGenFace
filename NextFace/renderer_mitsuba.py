"""Mitsuba 3 Renderer class with API-compatible interface.

Drop-in replacement for the original PyRedner-based Renderer class.
Same constructor signature, same public attributes, same method signatures.

When torch gradients are enabled, rendering uses a DrJit-PyTorch gradient
bridge so that loss.backward() propagates through the Mitsuba render back
to the input scene parameter tensors (vertices, textures, envmap).
"""

import torch
import math
import mitsuba as mi

from mitsuba_variant import ensure_variant
from scene_mitsuba import build_scenes
from render_mitsuba import render_scenes, render_albedo
from gradient_bridge import differentiable_render


class Renderer:

    def __init__(self, samples, bounces, device):
        ensure_variant()
        self.samples = samples
        self.bounces = bounces
        self.device = torch.device(device)
        self.clip_near = 10.0
        self.upVector = torch.tensor([0.0, -1.0, 0.0])
        self.counter = 0
        self.screenWidth = 256
        self.screenHeight = 256

    def setupCamera(self, focal, image_width, image_height):
        """Compute FOV from focal length. Returns FOV tensor for compatibility.

        Note: In the Mitsuba pipeline, camera construction happens inside
        buildScenes() via build_camera(). This method is kept for API
        compatibility and returns the FOV value.
        """
        fov = torch.tensor(
            [360.0 * math.atan(image_width / (2.0 * focal)) / math.pi]
        )
        return fov

    def buildScenes(self, vertices, indices, normal, uv, diffuse, specular,
                    roughness, focal, envMap):
        """Build Mitsuba scenes from batched parameters.

        Args:
            vertices: [N, V, 3] float32 torch tensor
            indices: [F, 3] int32 torch tensor
            normal: [N, V, 3] float32 torch tensor
            uv: [V, 2] float32 torch tensor
            diffuse: [N, H, W, 3] or [1, H, W, 3]
            specular: [N, H, W, 3] or [1, H, W, 3]
            roughness: [N, H, W, 1] or [1, H, W, 1]
            focal: [N] torch tensor
            envMap: [N, H, W, 3] torch tensor

        Returns:
            list of Mitsuba scenes
        """
        # Cache parameters for potential albedo rebuild
        self._last_build_params = (
            vertices, indices, normal, uv, diffuse, specular,
            roughness, focal, envMap,
        )
        # Use differentiable integrator when gradients are enabled,
        # high-quality path integrator otherwise (final output renders).
        differentiable = torch.is_grad_enabled()
        return build_scenes(
            vertices, indices, normal, uv, diffuse, specular,
            roughness, focal, envMap,
            screen_width=self.screenWidth,
            screen_height=self.screenHeight,
            samples=self.samples,
            bounces=self.bounces,
            differentiable=differentiable,
        )

    def renderAlbedo(self, scenes):
        """Render albedo of given scenes.

        Rebuilds scenes in albedo mode (AOV integrator) using the cached
        parameters from the last buildScenes() call, then renders albedo.

        Args:
            scenes: list of Mitsuba scenes (from buildScenes, used for
                    API compatibility but rebuilt internally for albedo)

        Returns:
            albedo images [N, H, W, 4]
        """
        if hasattr(self, '_last_build_params'):
            (vertices, indices, normal, uv, diffuse, specular,
             roughness, focal, envMap) = self._last_build_params
            albedo_scenes = build_scenes(
                vertices, indices, normal, uv, diffuse, specular,
                roughness, focal, envMap,
                screen_width=self.screenWidth,
                screen_height=self.screenHeight,
                samples=self.samples,
                bounces=self.bounces,
                albedo_mode=True,
            )
            return render_albedo(albedo_scenes, spp=self.samples,
                                 device=self.device)
        return render_albedo(scenes, spp=self.samples, device=self.device)

    def render(self, scenes):
        """Render scenes with path tracing.

        When torch gradients are enabled and cached build params exist,
        uses the DrJit-PyTorch gradient bridge so that loss.backward()
        propagates gradients through rendering to the input tensors.

        Args:
            scenes: list of Mitsuba scenes (from buildScenes)

        Returns:
            ray traced images [N, H, W, 4]
        """
        if torch.is_grad_enabled() and hasattr(self, '_last_build_params'):
            images = self._differentiable_render(scenes)
        else:
            images = render_scenes(scenes, spp=self.samples, device=self.device)
        self.counter += 1
        return images

    def _differentiable_render(self, scenes):
        """Render scenes with gradient flow via the DrJit-PyTorch bridge.

        For each scene, maps the cached input torch tensors to their
        corresponding Mitsuba scene parameter paths, then renders using
        differentiable_render() which bridges DrJit AD with PyTorch autograd.
        """
        (vertices, indices, normal, uv, diffuse, specular,
         roughness, focal, envMap) = self._last_build_params

        shared_texture = diffuse.shape[0] == 1
        n_frames = vertices.shape[0]

        images = []
        for i in range(n_frames):
            tex_idx = 0 if shared_texture else i

            # Discover available scene parameters
            params = mi.traverse(scenes[i])

            # Map torch tensors to scene parameter paths.
            # Only include paths that actually exist in the scene.
            torch_params = {}

            if "face_mesh.vertex_positions" in params:
                torch_params["face_mesh.vertex_positions"] = vertices[i].reshape(-1)

            if "face_mesh.vertex_normals" in params:
                torch_params["face_mesh.vertex_normals"] = normal[i].reshape(-1)

            if "face_mesh.bsdf.base_color.data" in params:
                torch_params["face_mesh.bsdf.base_color.data"] = diffuse[tex_idx]

            if "face_mesh.bsdf.roughness.data" in params:
                torch_params["face_mesh.bsdf.roughness.data"] = roughness[tex_idx]

            if "envmap.data" in params:
                envmap_tensor = envMap[i]
                # Mitsuba envmap adds a wrap-around column internally
                # (e.g., [H, W, 3] input becomes [H, W+1, 3] in scene).
                # Pad input tensor to match by repeating first column.
                envmap_scene = params["envmap.data"]
                if isinstance(envmap_scene, mi.TensorXf):
                    scene_shape = envmap_scene.shape
                    if (len(scene_shape) == 3
                            and scene_shape[1] == envmap_tensor.shape[0] + 1):
                        # envmap_tensor is [H, W, 3], scene is [H, W+1, 3]
                        envmap_tensor = torch.cat(
                            [envmap_tensor, envmap_tensor[:, :1, :]], dim=1
                        )
                    elif (len(scene_shape) == 3
                            and scene_shape[0] == envmap_tensor.shape[0]
                            and scene_shape[1] == envmap_tensor.shape[1] + 1):
                        envmap_tensor = torch.cat(
                            [envmap_tensor, envmap_tensor[:, :1, :]], dim=1
                        )
                torch_params["envmap.data"] = envmap_tensor

            img = differentiable_render(scenes[i], torch_params, spp=self.samples)

            # Ensure RGBA (4 channels)
            if img.shape[-1] == 3:
                alpha = torch.ones(*img.shape[:-1], 1)
                img = torch.cat([img, alpha], dim=-1)
            elif img.shape[-1] > 4:
                img = img[..., :4]

            images.append(img.to(self.device))

        return torch.stack(images, dim=0)
