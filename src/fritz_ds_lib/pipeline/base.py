from sklearn.pipeline import Pipeline

from fritz_ds_lib.core.base import ProjectBaseModel
from fritz_ds_lib.core.cereal import import_object


class PipelineConfig(ProjectBaseModel):
    pipeline_obj_path: str

    @property
    def pipeline(self) -> Pipeline:
        make_pipeline = import_object(self.pipeline_obj_path)
        return make_pipeline(self)
