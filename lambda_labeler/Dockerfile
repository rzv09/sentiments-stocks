# Base AWS Lambda Python image
FROM public.ecr.aws/lambda/python:3.11

# Install deps into the Lambda task root
COPY lambda_labeler/requirements.txt  .
RUN pip install --no-cache-dir -r requirements.txt -t ${LAMBDA_TASK_ROOT}

# Copy app + model (paths are relative to the build context root, i.e. the repo).
# The model comes from ml/models/baseline/ — where scripts/train_baseline.py
# writes it. This previously read lambda_labeler/ml/models/baseline/, a path the
# trainer never writes to: the build either failed outright, or silently shipped
# whatever stale artifact happened to be sitting there. Run the trainer first;
# the .joblib is gitignored and is not in a fresh clone.
COPY lambda_labeler/app.py ${LAMBDA_TASK_ROOT}/
COPY ml/models/baseline/ ${LAMBDA_TASK_ROOT}/ml/models/baseline/

# Set the Lambda handler
CMD ["app.handler"]
