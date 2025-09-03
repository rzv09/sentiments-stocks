# Base AWS Lambda Python image
FROM public.ecr.aws/lambda/python:3.11

# Install deps into the Lambda task root
COPY lambda_labeler/requirements.txt  .
RUN pip install --no-cache-dir -r requirements.txt -t ${LAMBDA_TASK_ROOT}

# Copy code + model into the image
COPY lambda_labeler/app.py                          ${LAMBDA_TASK_ROOT}/
COPY lambda_labeler/ml/models/baseline/tfidf_logreg.joblib ${LAMBDA_TASK_ROOT}/ml/models/baseline/

# Set the Lambda handler
CMD ["app.handler"]
