"""Hunyuan3D-2.1 teacher adapter.

Speaks the exact HTTP contract already implemented and debugged in
ml/cad/kaggle/hunyuan3d_api_server.ipynb:

    POST {endpoint}/generate {"prompt": str, "mode": "direct"|"image"}
      -> {"job_id": str}
    GET  {endpoint}/jobs/{job_id}
      -> {"status": "pending"} | {"status": "done", "result": {..., "mesh_url": str}}
         | {"status": "error", "error": str}
    GET  {endpoint}{mesh_url}
      -> raw mesh bytes (.stl)

That notebook is the reference implementation of the server side of this
contract (it also documents why generation is async/polled rather than a
single blocking request -- public tunnels enforce timeouts a real generation
can exceed). This file is only the client.

Configuration is never hardcoded -- no default endpoint is assumed. Set via
constructor args or environment variables:
    THREEDIMEN_HUNYUAN3D_ENDPOINT   e.g. https://xxxx.trycloudflare.com
    THREEDIMEN_HUNYUAN3D_MODE       "direct" (Shap-E, fast) or "image" (Hunyuan3D, slower/more detail)
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import requests

from .base import MeshArtifact, TeacherModel


class Hunyuan3DConfigError(RuntimeError):
    pass


class Hunyuan3DTeacher(TeacherModel):
    name = "hunyuan3d-2.1"

    def __init__(
        self,
        endpoint: str | None = None,
        mode: str | None = None,
        *,
        poll_interval_seconds: float = 2.0,
        timeout_seconds: float = 600.0,
        cache_dir: str | Path | None = None,
    ):
        self.endpoint = (endpoint or os.environ.get("THREEDIMEN_HUNYUAN3D_ENDPOINT") or "").rstrip("/")
        self.mode = mode or os.environ.get("THREEDIMEN_HUNYUAN3D_MODE", "direct")
        if self.mode not in ("direct", "image"):
            raise Hunyuan3DConfigError(f"mode must be 'direct' or 'image', got {self.mode!r}")
        self.poll_interval_seconds = poll_interval_seconds
        self.timeout_seconds = timeout_seconds
        self.cache_dir = Path(cache_dir) if cache_dir else Path(tempfile.gettempdir()) / "threedimen_hunyuan3d_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, prompt: str) -> MeshArtifact:
        if not self.endpoint:
            raise Hunyuan3DConfigError(
                "no Hunyuan3D endpoint configured. Set THREEDIMEN_HUNYUAN3D_ENDPOINT to the "
                "tunnel URL printed by ml/cad/kaggle/hunyuan3d_api_server.ipynb, or pass "
                "endpoint=... to Hunyuan3DTeacher()."
            )
        t0 = time.perf_counter()

        start = requests.post(
            f"{self.endpoint}/generate", json={"prompt": prompt, "mode": self.mode}, timeout=30
        )
        start.raise_for_status()
        job_id = start.json()["job_id"]

        result = self._poll_until_done(job_id)

        mesh_url = result["mesh_url"]
        mesh_response = requests.get(f"{self.endpoint}{mesh_url}", timeout=60)
        mesh_response.raise_for_status()
        mesh_path = self.cache_dir / Path(mesh_url).name
        mesh_path.write_bytes(mesh_response.content)

        # Trust the server's own report of which model actually generated this
        # mesh (result["part_type"] is "shap-e" for mode=direct, "hunyuan3d"
        # for mode=image -- see hunyuan3d_api_server.ipynb's _run_job). Using
        # self.name here instead would mislabel every direct-mode (Shap-E)
        # sample as Hunyuan3D in the dataset's provenance metadata.
        actual_model = result.get("part_type") or self.name
        return MeshArtifact(
            mesh_path=mesh_path,
            prompt=prompt,
            source_model=actual_model,
            generation_time_seconds=time.perf_counter() - t0,
            metadata={
                "endpoint": self.endpoint,
                "requested_mode": self.mode,
                "resolved_description": result.get("resolved_description"),
                "server_generation_time_seconds": result.get("generation_time_seconds"),
            },
        )

    def _poll_until_done(self, job_id: str) -> dict:
        deadline = time.perf_counter() + self.timeout_seconds
        while time.perf_counter() < deadline:
            time.sleep(self.poll_interval_seconds)
            job = requests.get(f"{self.endpoint}/jobs/{job_id}", timeout=30)
            job.raise_for_status()
            job_data = job.json()
            if job_data["status"] == "done":
                return job_data["result"]
            if job_data["status"] == "error":
                raise RuntimeError(f"Hunyuan3D generation failed server-side: {job_data.get('error')}")
        raise TimeoutError(f"Hunyuan3D generation did not finish within {self.timeout_seconds}s")
