"""Mitsuba 3 Renderer class with API-compatible interface.

Drop-in replacement for the original PyRedner-based Renderer class.
Same constructor signature, same public attributes, same method signatures.
"""

import torch
import math

from mitsuba_variant import ensure_variant
from scene_mitsuba import build_scenes
from render_mitsuba import render_scenes, render_albedo


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
        return build_scenes(
            vertices, indices, normal, uv, diffuse, specular,
            roughness, focal, envMap,
            screen_width=self.screenWidth,
            screen_height=self.screenHeight,
            samples=self.samples,
            bounces=self.bounces,
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

        Args:
            scenes: list of Mitsuba scenes (from buildScenes)

        Returns:
            ray traced images [N, H, W, 4]
        """
        images = render_scenes(scenes, spp=self.samples, device=self.device)
        self.counter += 1
        return images
