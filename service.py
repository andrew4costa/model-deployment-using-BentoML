import bentoml
import numpy as np
from bentoml.io import NumpyNdarray

# get runner
xgb_runner = bentoml.models.get("xgb_booster:latest").to_runner()

# create service object
svc = bentoml.Service("xgb_classifier", runners=[xgb_runner])

# create endpoint named classify
@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def classify(input_series) -> np.ndarray:
    label = xgb_runner.predict.run(input_series)

    return label 