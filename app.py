from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
import shutil
import os
import tempfile
import pysd

app = FastAPI(
    title="PySD Simulation API",
    description="Upload a Vensim or XMILE model, run simulations with custom parameters, and retrieve results.",
    version="1.0.0"
)

def run_model(model_path, parameters=None, return_vars=None, initial_time=None, final_time=None):
    # Detect file type and load model
    if model_path.endswith(".mdl"):
        model = pysd.read_vensim(model_path)
    elif model_path.endswith(".xmile"):
        model = pysd.read_xmile(model_path)
    else:
        raise ValueError("Unsupported file format")

    # Set parameter values if provided
    if parameters:
        model.set_components(parameters)

    # Set simulation start time
    initial_condition = (initial_time, {}) if initial_time is not None else 'original'

    # Run simulation
    result = model.run(
        params=parameters,
        return_columns=return_vars,
        initial_condition=initial_condition,
        final_time=final_time
    )

    return result


@app.post("/simulate")
async def simulate(
    file: UploadFile,
    initial_time: float = Form(None),
    final_time: float = Form(None),
    return_vars: str = Form(None),
    parameters: str = Form(None)
):
    try:
        # Save uploaded model file to temp location
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Parse parameters
        return_vars_list = return_vars.split(",") if return_vars else None
        parameters_dict = eval(parameters) if parameters else None

        # Run model
        result_df = run_model(
            model_path=file_path,
            parameters=parameters_dict,
            return_vars=return_vars_list,
            initial_time=initial_time,
            final_time=final_time
        )

        return JSONResponse(content=result_df.to_dict())

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

