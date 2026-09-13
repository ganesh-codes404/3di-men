from .base import MeshArtifact, TeacherModel
from .hunyuan3d import Hunyuan3DConfigError, Hunyuan3DTeacher
from .mock import MockTeacher

TEACHERS: dict[str, type[TeacherModel]] = {
    "hunyuan3d": Hunyuan3DTeacher,
    "mock": MockTeacher,
}

__all__ = [
    "MeshArtifact",
    "TeacherModel",
    "Hunyuan3DTeacher",
    "Hunyuan3DConfigError",
    "MockTeacher",
    "TEACHERS",
]
