# Create a ZIP file for Lambda deployment
data "archive_file" "lambda_zip" {
  type = "zip"
  dynamic "source" {
    for_each = var.lambda_sources
    content {
      content  = file(source.value.path)
      filename = source.value.filename
    }
  }

  output_path = "${path.module}/lambda_function.zip"
}

# Lambda function
resource "aws_lambda_function" "batch_launcher" {
  filename         = data.archive_file.lambda_zip.output_path
  function_name    = "${var.environment}-${var.system_name}-batch-launcher"
  role             = var.execution_role_arn
  handler          = var.handler
  source_code_hash = data.archive_file.lambda_zip.output_base64sha256
  runtime          = "python3.13"
  timeout          = var.timeout
  memory_size      = var.memory_size

  environment {
    variables = {
      LEROBOT_BUCKET_NAME  = var.lerobot_bucket_name
      SECRET_NAME          = var.secret_name
      BATCH_JOB_QUEUE      = var.batch_job_queue
      BATCH_JOB_DEFINITION = var.batch_job_definition
    }
  }

  tags = merge(var.common_tags, {
    Name = "${var.environment}-${var.system_name}-batch-launcher"
  })
}
