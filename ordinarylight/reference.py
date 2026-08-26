"""Small CPU reference path tracer used to define backend behavior."""

import math

import numpy as np

from .cameras import OrthographicCamera, PanoramicCamera
from .volume import integrate_volumes, intersect_unit_boxes


class ReferencePathTracer:
    """A deterministic, intentionally simple diffuse path tracer."""

    def __init__(self, seed=1):
        self.seed = seed

    def render(self, scene, camera, width, height, samples=1, max_bounces=3):
        """Render a display-ready RGBA8 reference image."""
        hdr = self.render_hdr(
            scene, camera, width, height,
            samples=samples, max_bounces=max_bounces,
        )
        rgb = hdr[..., :3]
        rgb = rgb / (1.0 + rgb)  # Reinhard tone mapping
        rgb = np.power(np.clip(rgb, 0.0, 1.0), 1.0 / 2.2)
        rgba = np.empty((height, width, 4), dtype=np.uint8)
        rgba[..., :3] = np.rint(rgb * 255.0).astype(np.uint8)
        rgba[..., 3] = 255
        return rgba

    def render_hdr(self, scene, camera, width, height, samples=1, max_bounces=3):
        """Render linear HDR float32 radiance for backend-neutral consumers."""
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")
        if samples <= 0 or max_bounces <= 0:
            raise ValueError("samples and max_bounces must be positive")

        triangles, colors, emissions = scene.triangles()
        rng = np.random.default_rng(self.seed)
        accumulated = np.zeros((height * width, 3), dtype=np.float32)

        for _ in range(samples):
            origins, directions = self._camera_rays(camera, width, height, rng)
            radiance = np.zeros_like(origins)
            throughput = np.ones_like(origins)
            active = np.ones(len(origins), dtype=bool)

            for _bounce in range(max_bounces):
                active_indices = np.flatnonzero(active)
                if not len(active_indices):
                    break
                hit_t, hit_triangle = self._intersect(
                    origins[active_indices], directions[active_indices], triangles
                )
                volumes = scene.visible_volumes
                if volumes:
                    entries, exits = intersect_unit_boxes(
                        origins[active_indices], directions[active_indices], volumes
                    )
                    volume_radiance, volume_transmittance = integrate_volumes(
                        volumes, origins[active_indices], directions[active_indices],
                        entries, exits, max_distance=hit_t, lights=scene.lights,
                    )
                    ray_ids = active_indices
                    radiance[ray_ids] += throughput[ray_ids] * volume_radiance
                    throughput[ray_ids] *= volume_transmittance[:, None]
                missed = hit_triangle < 0
                if np.any(missed):
                    ray_ids = active_indices[missed]
                    sky = self._sky(directions[ray_ids], scene.environment)
                    radiance[ray_ids] += throughput[ray_ids] * sky
                    active[ray_ids] = False

                hit = ~missed
                if not np.any(hit):
                    continue
                ray_ids = active_indices[hit]
                triangle_ids = hit_triangle[hit]
                hit_points = origins[ray_ids] + directions[ray_ids] * hit_t[hit, None]
                normals = self._normals(triangles[triangle_ids])
                flip = np.sum(normals * directions[ray_ids], axis=1) > 0.0
                normals[flip] *= -1.0

                radiance[ray_ids] += throughput[ray_ids] * emissions[triangle_ids]
                throughput[ray_ids] *= colors[triangle_ids]
                origins[ray_ids] = hit_points + normals * 1e-4
                directions[ray_ids] = self._cosine_hemisphere(normals, rng)

            accumulated += radiance

        rgb = accumulated.reshape((height, width, 3)) / samples
        rgba = np.empty((height, width, 4), dtype=np.float32)
        rgba[..., :3] = rgb
        rgba[..., 3] = 0.0
        return rgba

    def render_to(self, scene, camera, surface, samples=1, max_bounces=3):
        """Render directly into a RenderSurface and return its presentation result."""
        rgba = self.render(
            scene,
            camera,
            surface.width,
            surface.height,
            samples=samples,
            max_bounces=max_bounces,
        )
        return surface.present(rgba)

    @staticmethod
    def _camera_rays(camera, width, height, rng):
        position = np.asarray(camera.position, dtype=np.float32)
        forward = np.asarray(camera.target, dtype=np.float32) - position
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, np.asarray(camera.up, dtype=np.float32))
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        y, x = np.mgrid[0:height, 0:width]
        jitter_x = rng.random((height, width), dtype=np.float32)
        jitter_y = rng.random((height, width), dtype=np.float32)
        ndc_x = (x + jitter_x) / width * 2.0 - 1.0
        ndc_y = 1.0 - (y + jitter_y) / height * 2.0
        aspect = width / height
        if isinstance(camera, PanoramicCamera):
            yaw = ndc_x * math.radians(camera.horizontal_fov_degrees) * 0.5
            pitch = ndc_y * math.radians(camera.vertical_fov_degrees) * 0.5
            directions = (
                np.cos(pitch)[..., None] * np.cos(yaw)[..., None] * forward
                + np.cos(pitch)[..., None] * np.sin(yaw)[..., None] * right
                + np.sin(pitch)[..., None] * up
            ).reshape((-1, 3))
            origins = np.repeat(position[None, :], len(directions), axis=0)
        elif isinstance(camera, OrthographicCamera):
            half_height = camera.vertical_size * 0.5
            origins = (
                position[None, None, :]
                + ndc_x[..., None] * (aspect * half_height) * right
                + ndc_y[..., None] * half_height * up
            ).reshape((-1, 3))
            directions = np.repeat(forward[None, :], len(origins), axis=0)
        else:
            half_height = math.tan(
                math.radians(camera.vertical_fov_degrees) / 2.0
            )
            directions = (
                forward[None, None, :]
                + ndc_x[..., None] * (aspect * half_height) * right
                + ndc_y[..., None] * half_height * up
            ).reshape((-1, 3))
            origins = np.repeat(position[None, :], len(directions), axis=0)
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        return origins, directions

    @staticmethod
    def _intersect(origins, directions, triangles):
        count = len(origins)
        best_t = np.full(count, np.inf, dtype=np.float32)
        best_triangle = np.full(count, -1, dtype=np.int32)
        epsilon = 1e-7
        for triangle_id, triangle in enumerate(triangles):
            edge1 = triangle[1] - triangle[0]
            edge2 = triangle[2] - triangle[0]
            p = np.cross(directions, edge2)
            determinant = p @ edge1
            valid = np.abs(determinant) > epsilon
            inverse = np.zeros_like(determinant)
            inverse[valid] = 1.0 / determinant[valid]
            tvec = origins - triangle[0]
            u = np.sum(tvec * p, axis=1) * inverse
            q = np.cross(tvec, edge1)
            v = np.sum(directions * q, axis=1) * inverse
            t = (q @ edge2) * inverse
            hit = valid & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (t > 1e-4)
            nearer = hit & (t < best_t)
            best_t[nearer] = t[nearer]
            best_triangle[nearer] = triangle_id
        return best_t, best_triangle

    @staticmethod
    def _normals(triangles):
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        return normals

    @staticmethod
    def _sky(directions, environment=None):
        if environment is not None:
            radiance = np.ones((len(directions), 3), np.float32)
            if environment.image is not None:
                longitude = np.arctan2(directions[:, 2], directions[:, 0])
                longitude += environment.rotation
                u = np.mod(longitude / (2.0 * np.pi) + 0.5, 1.0)
                v = np.arccos(np.clip(directions[:, 1], -1.0, 1.0)) / np.pi
                height, width, _channels = environment.image.shape
                x = np.minimum((u * width).astype(np.int64), width - 1)
                y = np.minimum((v * height).astype(np.int64), height - 1)
                radiance = environment.image[y, x]
            return (
                radiance * np.asarray(environment.color, np.float32)
                * float(environment.intensity)
            )
        blend = np.clip(0.5 * (directions[:, 1] + 1.0), 0.0, 1.0)[:, None]
        return (1.0 - blend) * np.array((0.8, 0.8, 0.8)) + blend * np.array((0.2, 0.4, 0.9))

    @staticmethod
    def _cosine_hemisphere(normals, rng):
        random_vectors = rng.normal(size=normals.shape).astype(np.float32)
        random_vectors /= np.linalg.norm(random_vectors, axis=1, keepdims=True)
        random_vectors[np.sum(random_vectors * normals, axis=1) < 0.0] *= -1.0
        directions = normals + random_vectors
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        return directions
